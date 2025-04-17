"""Module implementing a RAG method using a reranker to refine retrieval results."""

import logging
from typing import Any, override

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.rag.reranker_base import Reranker
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


class RerankerRAG(BaseRAG):
    """RAG implementation that uses a reranker model to improve retrieval quality.

    This implementation extends BaseRAG by adding a reranking step after the initial
    vector search retrieval. The reranker refines the documents based on relevance
    scoring using a cross-encoder model.
    """

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_ratio: float = 3.0,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        where: dict[str, str] | None = None,
        device: str = "cuda",
        template_name: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize RerankerRAG with configuration.

        Args:
            context_collection: Collection of documents to use as context
            collection_name: Name of the collection in the vector database
            embedding_function: Function to create embeddings for vector search
            top_k: Number of documents to return (same meaning as in BaseRAG)
            reranker_model: Name or path of the cross-encoder reranker model
            rerank_ratio: How many times more documents to retrieve initially for reranking
                (e.g., 3.0 means retrieve 3*top_k documents initially)
            retrieval_only: If True, only retrieval is performed (no LLM inference)
            runner: Inference runner for LLM
            additional_prompt: Additional text to add to the prompt
            where: Optional filter for vector search
            device: Device to use for embeddings and reranking
            template_name: Name of the prompt template to use
            **kwargs: Additional arguments

        """
        # Create the reranker
        self.reranker_instance = Reranker(
            reranker_model=reranker_model, rerank_ratio=rerank_ratio, device=device
        )

        # Calculate how many documents to retrieve for reranking
        self.initial_top_k = self.reranker_instance.calculate_initial_top_k(top_k)

        # Initialize the parent class with the same top_k as requested
        super().__init__(
            context_collection=context_collection,
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=top_k,  # Keep original top_k to maintain LSP
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
            where=where,
            device=device,
            template_name=template_name,
            **kwargs,
        )

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

        for _, (query, documents) in enumerate(zip(query_list, initial_results)):
            # Rerank documents using the reranker
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
        meta_data: list[MetaData] = [],
        retrieval_instruction: list[str] = [],
        template_name: str = "",
    ) -> ResponseWrapper:
        """Execute the complete RAG pipeline with reranking and return responses."""
        # Generate queries and retrieve contexts with reranking
        if retrieval_instruction:
            logger.info(f"Generating {len(retrieval_instruction)} retrieval queries.")
            retrieved_documents = self.retrieve_contexts(retrieval_instruction)
            self.contexts = [Context.from_documents(documents) for documents in retrieved_documents]
        else:
            logger.info("No context retrieval queries provided. Using no context.")
            self.contexts = []

        template_name = template_name if template_name else self.template_name
        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=meta_data,
            template_name=template_name,
        )

        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            return create_mock_response_wrapper(prompt_collection)
        else:
            # Run inference with the LLM
            return runner.run(prompt_collection)
