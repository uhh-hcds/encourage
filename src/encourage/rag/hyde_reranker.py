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
from encourage.rag.hyde import HydeRAG
from encourage.rag.reranker_base import Reranker
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


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
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_ratio: float = 3.0,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        template_name: str = "",
        where: dict[str, str] | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        """Initialize HydeRerankerRAG with configuration.

        Args:
            context_collection: List of Document objects for the context collection
            collection_name: Name of the vector store collection
            embedding_function: Function to compute embeddings
            top_k: Number of top documents to return after reranking
            reranker_model: Name or path of the cross-encoder reranker model
            rerank_ratio: How many times more documents to retrieve initially for reranking
            retrieval_only: If True, only perform retrieval (no LLM inference)
            runner: Optional LLM runner for generating hypothetical answers
            additional_prompt: Prompt text to guide hypothetical answer generation
            template_name: Name of the prompt template to use
            where: Optional filter for vector search
            device: Device to use for computations
            **kwargs: Additional parameters

        """
        self.reranker_instance = Reranker(
            reranker_model=reranker_model, rerank_ratio=rerank_ratio, device=device
        )
        self.initial_top_k = self.reranker_instance.calculate_initial_top_k(top_k)
        self.final_top_k = top_k

        # Initialize the parent class (HydeRAG) with the initial_top_k for retrieval
        super().__init__(
            context_collection=context_collection,
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=self.initial_top_k,  # Use initial_top_k for retrieval
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
        """Retrieve contexts using HYDE approach and apply reranking.

        First generates hypothetical answers, then retrieves documents using those,
        and finally applies reranking to improve relevance.

        Args:
            query_list: List of queries for retrieval
            **kwargs: Additional arguments

        Returns:
            List of lists of reranked documents, ordered by relevance

        """
        # Use the parent class's retrieve_contexts to get documents via HYDE
        initial_results = super().retrieve_contexts(query_list, **kwargs)

        # Apply reranking to improve rankings
        reranked_results: list[list[Document]] = []
        for _, (query, documents) in enumerate(zip(query_list, initial_results)):
            # Rerank documents using the reranker, passing the original query
            reranked_documents = self.reranker_instance.rerank_documents(
                query=query, documents=documents, top_k=self.final_top_k
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
            logger.info(f"Generating {len(retrieval_queries)} retrieval queries.")
            retrieved_documents = self.retrieve_contexts(retrieval_queries)
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
