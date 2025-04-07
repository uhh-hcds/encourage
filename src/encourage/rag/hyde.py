"""HYDE (Hypothetical Document Embeddings) RAG implementation.

HYDE generates hypothetical answers to queries and uses those answers as the search vector
for retrieving relevant documents, which can improve retrieval quality compared to using
the original query directly.

Reference: https://arxiv.org/abs/2212.10496
"""

import logging
from typing import Any, override

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


class HydeRAG(BaseRAG):
    """HYDE (Hypothetical Document Embeddings) implementation of RAG.

    HYDE generates hypothetical answers to queries and uses those answers as the search vector
    for retrieving relevant documents.
    """

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        retrieval_only: bool = False,
        device: str = "cuda",
        where: dict[str, str] | None = None,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        template_name: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize HYDE RAG method with configuration.

        Args:
            context_collection: List of Document objects representing the context collection
            template_name: Name of the template to use for prompt formatting
            collection_name: Name of the vector store collection
            embedding_function: Function to compute embeddings for queries and documents
            top_k: Number of top documents to retrieve
            device: Device to use for embedding computations ("cuda" or "cpu")
            where: Optional filter criteria for document retrieval
            retrieval_only: If True, only perform retrieval without LLM inference
            runner: Optional BatchInferenceRunner for generating hypothetical answers
            additional_prompt: Additional prompt text to include in hypothetical answers
            **kwargs: Additional parameters for customization

        """
        # Initialize the parent class first
        super().__init__(
            context_collection=context_collection,
            collection_name=collection_name,
            embedding_function=embedding_function,
            top_k=top_k,
            retrieval_only=retrieval_only,
            device=device,
            where=where,
            runner=runner,
            additional_prompt=additional_prompt,
            template_name=template_name,
        )

    def generate_hypothetical_answer(self, query_list: list[str]) -> ResponseWrapper:
        """Generate a hypothetical answer to the query using the LLM."""
        # Create prompt collection using the additional_prompt as system prompt
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=self.additional_prompt,
            user_prompts=query_list,
            template_name=self.template_name,
        )

        if not self.runner:
            raise ValueError("No LLM runner provided for generating hypothetical answers.")

        # Get the response from the LLM using the main runner
        return self.runner.run(prompt_collection)

    @override
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts from the database using hypothetical answers as search vectors."""
        # Generate hypothetical answers for all queries
        self.hypothetical_answers = self.generate_hypothetical_answer(query_list)
        # Store the hypothetical answers for later use in metadata
        responses = self.hypothetical_answers.get_responses()

        # Use hypothetical answers as search vectors instead of original queries
        return self.client.query(
            collection_name=self.collection_name,
            query=responses,
            top_k=self.top_k,
            embedding_function=self.embedding_function,
            where=self.where if self.where else None,
        )

    @override
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        meta_datas: list[MetaData] = [],
        retrieval_instruction: list[str] = [],
        template_name: str = "",
    ) -> ResponseWrapper:
        """Execute the HYDE RAG pipeline and return responses.

        Args:
            runner: LLM runner to use for final answer generation
            sys_prompt: System prompt for the final answer generation
            user_prompts: Optional list of user prompts (questions)
            meta_datas: Optional list of metadata for the prompts
            retrieval_instruction: Optional retrieval instructions
            template_name: Optional template name for prompt formatting

        Returns:
            ResponseWrapper containing the responses from the LLM

        """
        # Retrieve contexts using HYDE approach (hypothetical answers)
        retrieved_documents = self.retrieve_contexts(user_prompts)
        contexts = [Context.from_documents(documents) for documents in retrieved_documents]

        # Add hypothetical answers to metadata
        if self.hypothetical_answers:
            hypothetical_responses = self.hypothetical_answers.get_responses()
            for i, meta_data in enumerate(meta_datas):
                if i < len(hypothetical_responses):
                    meta_data["hypothetical_answer"] = hypothetical_responses[i]

        # Use provided template_name or fall back to self.template_name
        template_name = template_name if template_name else self.template_name

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
            # Run inference with the LLM using the class runner
            return runner.run(prompt_collection)
