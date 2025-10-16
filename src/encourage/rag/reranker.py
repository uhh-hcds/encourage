"""Module implementing a RAG method using a reranker to refine retrieval results."""

import logging
from typing import Any, override

from pydantic import BaseModel
from tqdm import tqdm

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.config import RerankerRAGConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base_impl import BaseRAG
from encourage.rag.rerank.factory import RerankerFactory
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


@RAGFactory.register(RAGMethod.Reranker, RerankerRAGConfig)
class RerankerRAG(BaseRAG):
    """RAG implementation using a reranker model to improve retrieval quality."""

    def __init__(
        self,
        config: RerankerRAGConfig,
        **kwargs: Any,
    ) -> None:
        """Initialize RerankerRAG using BaseRAGConfig and reranker-specific params.

        Args:
            config: BaseRAGConfig instance with core configuration.
            **kwargs: Additional arguments.

        """
        # Initialize parent with config parameters via dict unpacking
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
        """Retrieve contexts and apply reranking to improve relevance.

        Args:
            query_list: List of queries for retrieval
            **kwargs: Additional arguments

        Returns:
            List of lists of documents, reranked by relevance

        """
        # Override top_k for this retrieval only
        original_top_k = self.top_k
        self.top_k = self.initial_top_k

        try:
            # Get initial candidates using vector search
            initial_results = super().retrieve_contexts(query_list, **kwargs)
        finally:
            # Restore original top_k
            self.top_k = original_top_k

        reranked_results: list[list[Document]] = []

        for query, documents in tqdm(zip(query_list, initial_results), desc="Reranking", total=len(query_list)):
            # Rerank documents using the reranker
            print(len(documents))
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
        """Execute the complete RAG pipeline with reranking and return responses."""
        # Generate queries and retrieve contexts with reranking
        if retrieval_queries:
            user_prompts = retrieval_queries
            logger.info(f"Generating {len(retrieval_queries)} retrieval queries.")
        else:
            logger.info(f"Using {len(user_prompts)} user prompts for retrieval.")

        retrieved_documents = self.retrieve_contexts(user_prompts)
        self.contexts = [Context.from_documents(documents) for documents in retrieved_documents]

        # Create prompt collection with template_name from class
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=meta_datas,
            template_name=self.template_name,
        )

        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            return create_mock_response_wrapper(prompt_collection)
        else:
            # Run inference with the LLM
            return runner.run(prompt_collection, response_format=response_format)
