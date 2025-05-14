#self_rag.py
"""Self‑RAG implementation – **now with real structured outputs.**

This revision wires the Pydantic schemas declared at the top (_BinaryResponse,
_RelevanceResponse …) directly into every LLM call via
``BatchInferenceRunner.run(..., response_format=<YourSchema>)``. Doing so lets
Encourage parse the LLM‑returned JSON and gives us strong typing instead of the
previous string‑based heuristics.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from pydantic import BaseModel, Field

from encourage.llm import BatchInferenceRunner, Response, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured outputs (Pydantic) ---------------------------------------------

class BinaryResponse(BaseModel):
    """Yes/No response used for the retrieval decision."""

    response: str = Field(..., pattern="^(Yes|No)$")

class RelevanceResponse(BaseModel):
    """Relevant/Irrelevant response for context filtering."""

    response: str = Field(..., pattern="^(Relevant|Irrelevant)$")

class GenerationResponse(BaseModel):
    response: str

class SupportResponse(BaseModel):
    response: str = Field(..., pattern="^(Fully supported|Partially supported|No support)$")

class UtilityResponse(BaseModel):
    response: int = Field(..., ge=1, le=5)

# ---------------------------------------------------------------------------
#  SelfRAG -----------------------------------------------------------

class SelfRAG(BaseRAG):
    """Self‑RAG with structured‑output powered stages."""

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int = 3,
        device: str = "cuda",
        runner: BatchInferenceRunner | None = None,
        reflection_rounds: int = 1,
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
        self.reflection_rounds = reflection_rounds
        # --- System prompts (shortened for brevity) ----------------------
        self._decision_sys = "Given a user query, answer only 'Yes' or 'No' – do we need external retrieval?"
        self._relevance_sys = "Relevant or Irrelevant? Answer with one of those two words only."
        self._generation_sys = "Answer helpfully using the supplied context."
        self._support_sys = "Label as Fully supported, Partially supported or No support – exactly one."
        self._utility_sys = "Rate utility on a 1‑5 integer scale. Output the number only."
        self._reflection_sys = (
            "Critique the answer for factual accuracy, completeness and relevance."
        )
        self._refine_sys = "Improve the answer based on the critique above."

    # -------------------------------------------------------------------
    # Pipeline entry ------------------------------------------------------

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: List[str],
        **_,
    ) -> ResponseWrapper:  # type: ignore[override]

        if not user_prompts:
            return create_mock_response_wrapper(PromptCollection(prompts=[]))

        # 1) Retrieval decision ------------------------------------------
        needs_retrieval = self._decide_retrieval(runner, user_prompts)

        # 2) Retrieve + relevance filter ----------------------------------
        retrieved = self._retrieve_and_filter(runner, user_prompts, needs_retrieval)

        # 3) Generate / score / pick best ---------------------------------
        answers = []
        for q, docs in zip(user_prompts, retrieved):
            if not docs:
                answers.append(self._generate_no_ctx(runner, q))
                continue
            cands = self._generate_candidates(runner, q, docs)
            sup = self._assess_support(runner, cands, docs)
            util = self._rate_utility(runner, q, cands)
            answers.append(self._select(cands, sup, util))

        # 4) Optional reflection loop ------------------------------------
        if self.reflection_rounds:
            answers = self._reflect_and_refine(runner, user_prompts, answers, retrieved)

        # 5) Wrap ---------------------------------------------------------
        responses: List[Response] = []
        for i, (q, a, docs) in enumerate(zip(user_prompts, answers, retrieved)):
            ctx = Context.from_documents(docs) if docs else Context()
            responses.append(
                Response(
                    request_id=f"esrag-{i}",
                    prompt_id="",
                    conversation_id=i,
                    sys_prompt=sys_prompt,
                    user_prompt=q,
                    response=a,
                    context=ctx,
                    meta_data=MetaData(),
                    arrival_time=0.0,
                    finished_time=0.0,
                )
            )
        return ResponseWrapper(responses)

    # -------------------------------------------------------------------
    # Helper stages -------------------------------------------------------

    def _decide_retrieval(self, runner: BatchInferenceRunner, queries: List[str]) -> List[bool]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._decision_sys,
            user_prompts=[f"{q}" for q in queries],
        )
        preds = runner.run(pcs, response_format=BinaryResponse).get_parsed()  # type: ignore[attr-defined]
        return [p.response == "Yes" for p in preds]

    def _retrieve_and_filter(
        self,
        runner: BatchInferenceRunner,
        queries: List[str],
        needs: List[bool],
    ) -> List[List[Document]]:
        docs_per_q: List[List[Document]] = [[] for _ in queries]
        for idx, (q, need) in enumerate(zip(queries, needs)):
            if need:
                raw_docs = self.retrieve_contexts([q])[0]
                # Relevance filtering
                pcs = PromptCollection.create_prompts(
                    sys_prompts=self._relevance_sys,
                    user_prompts=[
                        f"QUERY: {q}\nCONTEXT: {d.page_content}" for d in raw_docs
                    ],
                )
                rels = runner.run(pcs, response_format=RelevanceResponse).get_parsed()  # type: ignore[attr-defined]
                docs_per_q[idx] = [d for d, r in zip(raw_docs, rels) if r.response == "Relevant"]
        return docs_per_q

    def _generate_candidates(self, runner: BatchInferenceRunner, q: str, docs: List[Document]) -> List[str]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._generation_sys,
            user_prompts=[f"QUERY: {q}\nCONTEXT: {d.page_content}" for d in docs],
        )
        return [g.response for g in runner.run(pcs, response_format=GenerationResponse).get_parsed()]  # type: ignore[attr-defined]

    def _assess_support(self, runner: BatchInferenceRunner, ans: List[str], docs: List[Document]) -> List[str]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._support_sys,
            user_prompts=[
                f"RESPONSE: {a}\nCONTEXT: {d.page_content}" for a, d in zip(ans, docs)
            ],
        )
        return [s.response for s in runner.run(pcs, response_format=SupportResponse).get_parsed()]  # type: ignore[attr-defined]

    def _rate_utility(self, runner: BatchInferenceRunner, q: str, ans: List[str]) -> List[int]:
        pcs = PromptCollection.create_prompts(
            sys_prompts=self._utility_sys,
            user_prompts=[f"QUERY: {q}\nRESPONSE: {a}" for a in ans],
        )
        return [u.response for u in runner.run(pcs, response_format=UtilityResponse).get_parsed()]  # type: ignore[attr-defined]

    @staticmethod
    def _select(ans: List[str], sup: List[str], util: List[int]) -> str:
        ranking: List[Tuple[int, int, str]] = []
        for a, s, u in zip(ans, sup, util):
            s_rank = 2 if s.startswith("Fully") else 1 if s.startswith("Partially") else 0
            ranking.append((s_rank, u, a))
        return max(ranking)[2]

    def _generate_no_ctx(self, runner: BatchInferenceRunner, q: str) -> str:
        pc = PromptCollection.create_prompts(
            sys_prompts=self._generation_sys,
            user_prompts=[f"QUERY: {q}\nCONTEXT: No additional context"],
        )
        return runner.run(pc, response_format=GenerationResponse).get_parsed()[0].response  # type: ignore[attr-defined]

    # Reflection loop (unchanged except structured outputs not required) ---
    def _reflect_and_refine(
        self,
        runner: BatchInferenceRunner,
        q: List[str],
        ans: List[str],
        docs: List[List[Document]],
    ) -> List[str]:
        current = ans
        for _ in range(self.reflection_rounds):
            # Reflection
            pcs = PromptCollection.create_prompts(
                sys_prompts=self._reflection_sys,
                user_prompts=[
                    f"QUERY: {qq}\nRESPONSE: {aa}\nCONTEXT: {Context.from_documents(dd).to_string() if dd else 'No context'}"
                    for qq, aa, dd in zip(q, current, docs)
                ],
            )
            crit = [r.response for r in runner.run(pcs, response_format=GenerationResponse).get_parsed()]  # reuse GenerationResponse to hold string
            # Refinement
            pcs2 = PromptCollection.create_prompts(
                sys_prompts=self._refine_sys,
                user_prompts=[
                    f"QUERY: {qq}\nINITIAL RESPONSE: {aa}\nCRITIQUE: {cc}\nCONTEXT: {Context.from_documents(dd).to_string() if dd else 'No context'}"
                    for qq, aa, cc, dd in zip(q, current, crit, docs)
                ],
            )
            current = [r.response for r in runner.run(pcs2, response_format=GenerationResponse).get_parsed()]
        return current
