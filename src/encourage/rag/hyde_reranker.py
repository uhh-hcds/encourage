"""Module implementing a RAG method that combines HYDE and Reranker approaches.

This implementation first generates hypothetical answers using HYDE's approach,
then retrieves documents using those hypothetical answers as queries,
and finally reranks the retrieved documents using a cross-encoder model.

It benefits from both:
- HYDE's ability to improve retrieval by using hypothetical documents
- Reranker's ability to improve precision through cross-encoder scoring
"""

import logging
from typing import Any, override

from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.config import HydeRerankerConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.hyde import HydeRAG
from encourage.rag.rerank.factory import RerankerFactory
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


@RAGFactory.register(RAGMethod.HydeReranker, HydeRerankerConfig)
class HydeRerankerRAG(HydeRAG):
    """RAG implementation combining HYDE (hypothetical document embeddings) with reranking.

    This RAG method enhances retrieval in two stages:
    1. Generates hypothetical answers to improve vector search (HYDE approach)
    2. Reranks retrieved documents using a cross-encoder for better precision

    Attributes:
        reranker_instance (Reranker): Instance of the Reranker class
        initial_top_k (int): Number of documents to retrieve initially before reranking

    """

    def __init__(
        self,
        config: HydeRerankerConfig,
        **kwargs: Any,
    ) -> None:
        """Initialize HydeRerankerRAG with configuration."""
        super().__init__(config=config)

        # Create reranker instance
        self.reranker_instance = RerankerFactory.create(
            rerank_ratio=config.rerank_ratio,
            model=config.reranker_model,
            device=config.device,
        )

        # Calculate how many docs to initially retrieve for reranking
        self.initial_top_k = self.reranker_instance.calculate_initial_top_k(config.top_k)

    @override
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts using HYDE approach and apply reranking.

        First generates hypothetical answers, then retrieves documents using those,
        and finally applies reranking to improve relevance.

        Args:
            query_list: List of queries for retrieval
            **kwargs: Additional arguments

        Returns:
            List of lists of reranked documents, ordered by relevance

        """
        original_top_k = self.top_k
        self.top_k = self.initial_top_k

        # Use the parent class's retrieve_contexts to get documents via HYDE
        self.hypothetical_answers = self.generate_hypothetical_answer(query_list)
        # Store the hypothetical answers for later use in metadata
        responses = self.hypothetical_answers.get_responses()
        initial_results = super().retrieve_contexts(responses, **kwargs)

        # Apply reranking to improve rankings
        reranked_results: list[list[Document]] = []
        for _, (query, documents) in enumerate(zip(query_list, initial_results)):
            # Rerank documents using the reranker, passing the original query
            reranked_documents = self.reranker_instance.rerank_documents(
                query=query, documents=documents, top_k=original_top_k
            )
            reranked_results.append(reranked_documents)

        return reranked_results

    @override
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        meta_datas: list[MetaData] = [],
        retrieval_queries: list[str] = [],
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Execute the complete HYDE+Reranker RAG pipeline and return responses.

        Args:
            runner: LLM runner to use for final answer generation
            sys_prompt: System prompt for the final answer generation
            user_prompts: List of user prompts (questions)
            meta_datas: List of metadata for the prompts
            retrieval_queries: Optional retrieval queries
            response_format: Optional response format for structured output

        Returns:
            ResponseWrapper containing the responses from the LLM

        """
        # If specific retrieval queries are provided, use them; otherwise use user_prompts
        if retrieval_queries:
            user_prompts = retrieval_queries
            logger.info(f"Generating {len(retrieval_queries)} retrieval queries.")
        else:
            logger.info(f"Using {len(user_prompts)} user prompts for retrieval.")

        retrieved_documents = self.retrieve_contexts(user_prompts)
        contexts = [Context.from_documents(documents) for documents in retrieved_documents]

        # Add hypothetical answers to metadata if available
        if self.hypothetical_answers:
            hypothetical_responses = self.hypothetical_answers.get_responses()
            for i, meta_data in enumerate(meta_datas):
                if i < len(hypothetical_responses):
                    meta_data["hypothetical_answer"] = hypothetical_responses[i]

        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=contexts,
            meta_datas=meta_datas,
            template_name=self.template_name,
        )

        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            return create_mock_response_wrapper(prompt_collection)
        else:
            # Run inference with the LLM
            return runner.run(prompt_collection, response_format=response_format)
