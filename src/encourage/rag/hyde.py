"""HYDE (Hypothetical Document Embeddings) RAG implementation.

HYDE generates hypothetical answers to queries and uses those answers as the search vector
for retrieving relevant documents, which can improve retrieval quality compared to using
the original query directly.

Reference: https://arxiv.org/abs/2212.10496
"""

import logging
from typing import Any

import pandas as pd

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context
from encourage.rag.naive import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


class HydeRAG(BaseRAG):
    """HYDE (Hypothetical Document Embeddings) implementation of RAG.

    HYDE generates hypothetical answers to queries and uses those answers as the search vector
    for retrieving relevant documents.
    """

    def __init__(
        self,
        qa_dataset: pd.DataFrame,
        template_name: str,
        collection_name: str,
        top_k: int,
        embedding_function: Any,
        meta_data_keys: list[str],
        context_key: str = "context",
        question_key: str = "question",
        answer_key: str = "program_answer",
        device: str = "cuda",
        where: dict[str, str] | None = None,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        cache_hypothetical_answers: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize HYDE RAG method with configuration.

        Args:
            qa_dataset: DataFrame containing questions and contexts
            template_name: Name of the template to use for prompt formatting
            collection_name: Name of the vector store collection
            top_k: Number of documents to retrieve
            embedding_function: Embedding function to use
            meta_data_keys: list of metadata keys to include
            context_key: Key in the dataset containing the context
            question_key: Key in the dataset containing the question
            answer_key: Key in the dataset containing the reference answer
            device: Device to use for embedding ("cuda" or "cpu")
            where: Optional filter for document retrieval
            retrieval_only: If True, only perform retrieval without LLM inference
            runner: LLM runner to use for final answer generation and hypothetical answers
            additional_prompt: Additional prompt text to include in hypothetical answers
            cache_hypothetical_answers: Whe ofther to cache hypothetical answers
            **kwargs: Additional parameters

        """
        # Initialize the parent class first
        super().__init__(
            qa_dataset=qa_dataset,
            template_name=template_name,
            collection_name=collection_name,
            top_k=top_k,
            embedding_function=embedding_function,
            meta_data_keys=meta_data_keys,
            context_key=context_key,
            question_key=question_key,
            answer_key=answer_key,
            device=device,
            where=where,
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
            **kwargs,
        )
        # Store hypothetical answers
        self.hypothetical_answers = None

    def generate_hypothetical_answer(self, query_list: list[str]) -> ResponseWrapper:
        """Generate a hypothetical answer to the query using the LLM.

        Args:
            query_list: list of queries to generate hypothetical answers for

        Returns:
            A hypothetical answer to the query

        """
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

    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[Context]:
        """Retrieve contexts from the database using hypothetical answers as search vectors.

        Args:
            query_list: list of queries to retrieve contexts for
            kwargs: Additional parameters

        Returns:
            list of contexts retrieved from the database

        """
        # Generate hypothetical answers for all queries
        hypothetical_answers = self.generate_hypothetical_answer(query_list)
        # Store the hypothetical answers for later use in metadata
        self.hypothetical_answers = hypothetical_answers
        responses = hypothetical_answers.get_responses()

        # Use hypothetical answers as search vectors instead of original queries
        results = self.client.query(
            collection_name=self.collection_name,
            query=responses,
            top_k=self.top_k,
            embedding_function=self.embedding_function,
            where=self.where if self.where else None,
        )

        return [Context.from_documents(document_list) for document_list in results]

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        retrieval_instruction: list[str] = [],
    ) -> ResponseWrapper:
        """Execute the HYDE RAG pipeline and return responses.

        Args:
            runner: LLM runner to use for final answer generation
            sys_prompt: System prompt for the final answer generation
            user_prompts: Optional list of user prompts (questions)
            retrieval_instruction: Optional retrieval instructions

        Returns:
            ResponseWrapper containing the responses from the LLM

        """
        user_prompts = user_prompts if user_prompts else self.user_prompts

        # Retrieve contexts using HYDE approach (hypothetical answers)
        contexts = self.retrieve_contexts(user_prompts)

        # Add hypothetical answers to metadata
        meta_datas = self.metadata.copy()
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
            # Run inference with the LLM using the class runner
            return runner.run(prompt_collection)
