"""Self-RAG (Retrieval, Generation, and Critique through Self-Reflection) implementation.

Self-RAG is a framework that enhances LLM responses by combining retrieval with self-reflection.
The model determines when to retrieve, critique its own output for factuality, and adopts a
reflection-driven approach to improve response quality.

Reference: https://arxiv.org/abs/2310.11511
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Tuple, override

from pydantic import BaseModel, Field

from encourage.llm import BatchInferenceRunner, Response, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)

# Structured outputs (Pydantic)


class RelevanceResponse(BaseModel):
    """Response model for relevance classification."""

    response: str = Field(..., pattern="^(Relevant|Irrelevant)$")


class GenerationResponse(BaseModel):
    """Response model for generation."""

    response: str


class SupportResponse(BaseModel):
    """Response model for support classification."""

    response: str = Field(..., pattern="^(Fully supported|Partially supported|No support)$")


class UtilityResponse(BaseModel):
    """Response model for utility rating."""

    response: int = Field(..., ge=1, le=5)


class SelfRAG(BaseRAG):
    """Self‑RAG pipeline (retrieval ➜ relevance check ➜ generation ➜ support check ➜ utility)."""

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int = 3,
        device: str = "cuda",
        runner: BatchInferenceRunner | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            context_collection=context_collection,
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=top_k,
            device=device,
            runner=runner,
        )

        # System prompts for each step
        self._relevance_sys = """
        You are a critical evaluator employed by a RAG system. Analyze the retrieved context for
        factuality, relevance, coherence, and information completeness. Identify any hallucinations
        or missing important information from the retrieved context.
        Relevant or Irrelevant? Answer with one of those two words only.
        """
        self._support_sys = """
        Given the response and the context, determine if the response is supported by the context.
        Label as Fully supported, Partially supported or No support – output exactly one of
        those only.
        """
        self._utility_sys = """
        Given the query and the response, rate the utility of the response from 1 to 5. Output the
        number only.
        """

    @override
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] | None = None,
        meta_datas: list[MetaData] | None = None,
        retrieval_queries: list[str] | None = None,
        response_format: type[BaseModel] | str | None = None,  # kept for API parity
    ) -> ResponseWrapper:
        user_prompts = [(up or "") for up in (user_prompts or [])]

        if not user_prompts:
            return create_mock_response_wrapper(PromptCollection(prompts=[]))

        # 1) Retrieve + relevance filter (for every query)
        retrieved = self._retrieve_and_filter(
            runner,
            user_prompts,
            retrieval_queries=retrieval_queries,
        )

        self._generation_sys = sys_prompt
        self.response_format = response_format

        # 2) Generate / score / pick best (for every retrieved *relevant* document)
        answers: List[str] = []
        for q, docs in zip(user_prompts, retrieved):
            if not docs:
                answers.append(self._generate_no_ctx(runner, q))
                continue
            cands = self._generate_candidates(runner, q, docs)
            sup = self._assess_support(runner, cands, docs)
            util = self._rate_utility(runner, q, cands)
            answers.append(self._select(cands, sup, util))

        # 3) Wrap
        responses: List[Response] = []
        for i, (q, a, docs) in enumerate(zip(user_prompts, answers, retrieved)):
            if docs:
                ctx = Context.from_documents(docs)
            else:  # supply an empty context to avoid metric errors
                ctx = Context.from_documents([Document(content="No context retrieved", score=0.0)])
            responses.append(
                Response(
                    request_id=f"selfrag-{i}",
                    prompt_id="",
                    conversation_id=i,
                    sys_prompt=sys_prompt,
                    user_prompt=q,
                    response=a,
                    context=ctx,
                    meta_data=(meta_datas[i] if meta_datas and i < len(meta_datas) else MetaData()),
                    arrival_time=0.0,
                    finished_time=0.0,
                )
            )
        return ResponseWrapper(responses)

    def _retrieve_and_filter(
        self,
        runner: BatchInferenceRunner,
        queries: List[str],
        retrieval_queries: list[str] | None = None,
    ) -> List[List[Document]]:
        """Retrieve contexts *once* for the full `queries` list and then filter.

        This leverages `BaseRAG.retrieve_contexts`, which already batches the
        VectorStore calls internally.  We simply iterate over the returned
        document lists and discard those marked *Irrelevant*.
        """
        # 1. Retrieve *top_k* docs for **all** queries in one go
        if retrieval_queries and len(retrieval_queries) == len(queries):
            lookup_strings = retrieval_queries
        else:
            # Fall back to the user queries when nothing (usable) was supplied
            lookup_strings = queries

        all_raw_docs = self.retrieve_contexts(lookup_strings)

        # 2. Build a single relevance‑classification batch
        user_prompts: List[str] = []
        mapping: List[tuple[int, int]] = []  # (query_idx, doc_idx)
        # Ensure retrieval_queries is an iterable before zipping
        retrieval_queries_iter = retrieval_queries or []
        for q_idx, (retrieval_query, _, docs) in enumerate(
            zip(retrieval_queries_iter, queries, all_raw_docs)
        ):
            for d_idx, doc in enumerate(docs):
                mapping.append((q_idx, d_idx))
                user_prompts.append(
                    f"RETRIEVAL QUERY: {retrieval_query} CONTEXT: {doc.content}"
                )  # QUERY: {q}

        if not user_prompts:
            return [[] for _ in queries]

        pcs = PromptCollection.create_prompts(
            sys_prompts=self._relevance_sys,
            user_prompts=user_prompts,
        )
        relevance_preds = runner.run(pcs, response_format=RelevanceResponse)
        # 3. Distribute filtered docs back to their queries
        docs_per_q: List[List[Document]] = [[] for _ in queries]
        for (q_idx, d_idx), pred in zip(mapping, relevance_preds):
            try:
                pred = json.loads(pred.response)
                tag = pred["response"]  # type: ignore
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {pred.response}")
                tag = "Irrelevant"  # Default to Irrelevant if decoding fails
                continue

            if tag == "Relevant":
                docs_per_q[q_idx].append(all_raw_docs[q_idx][d_idx])

        return docs_per_q

    def _generate_candidates(
        self, runner: BatchInferenceRunner, q: str, docs: List[Document]
    ) -> List[str]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._generation_sys,
            user_prompts=[f"QUERY: {q}\nCONTEXT: {d.content}" for d in docs],
        )
        return [g.response for g in runner.run(pcs, response_format=self.response_format)]

    def _assess_support(
        self, runner: BatchInferenceRunner, ans: List[str], docs: List[Document]
    ) -> List[str]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._support_sys,
            user_prompts=[f"RESPONSE: {a}\nCONTEXT: {d.content}" for a, d in zip(ans, docs)],
        )
        return [s.response for s in runner.run(pcs, response_format=SupportResponse)]

    def _rate_utility(self, runner: BatchInferenceRunner, q: str, ans: List[str]) -> List[int]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._utility_sys,
            user_prompts=[f"QUERY: {q}\nRESPONSE: {a}" for a in ans],
        )
        return [u.response for u in runner.run(pcs, response_format=UtilityResponse)]

    @staticmethod
    def _select(ans: List[str], sup: List[str], util: List[int]) -> str:
        ranking: List[Tuple[int, int, str]] = []
        for a, s, u in zip(ans, sup, util):
            s_json = json.loads(s)["response"]
            # Convert utility value to int if it's a string
            u_value = u if isinstance(u, int) else json.loads(u)["response"]
            s_rank = 2 if s_json.startswith("Fully") else 1 if s_json.startswith("Partially") else 0
            ranking.append((s_rank, u_value, a))
        return max(ranking)[2]

    def _generate_no_ctx(self, runner: BatchInferenceRunner, q: str) -> str:
        pc = PromptCollection.create_prompts(
            sys_prompts=self._generation_sys,
            user_prompts=[f"QUERY: {q}\nCONTEXT: No additional context"],
        )
        response = runner.run(pc, response_format=self.response_format)[0].response
        return str(response) if response is not None else ""
