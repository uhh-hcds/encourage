"""HYDE (Hypothetical Document Embeddings) RAG implementation.

HYDE generates hypothetical answers to queries and uses those answers as the search vector
for retrieving relevant documents, which can improve retrieval quality compared to using
the original query directly.

Reference: https://arxiv.org/abs/2212.10496
"""

import logging
from typing import Any, override

from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.config import HydeRAGConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper

logger = logging.getLogger(__name__)


@RAGFactory.register(RAGMethod.Hyde, HydeRAGConfig)
class HydeRAG(BaseRAG):
    """HYDE (Hypothetical Document Embeddings) implementation of RAG.

    HYDE generates hypothetical answers to queries and uses those
    answers as the search vectors for retrieving relevant documents.

    Args:
        config (HydeRAGConfig): Configuration dataclass for HYDE RAG.
        **kwargs: Additional keyword arguments for BaseRAG.

    """

    def __init__(self, config: HydeRAGConfig) -> None:
        """Initialize HYDE RAG with config.

        Passes configuration parameters to BaseRAG initializer and performs
        any HYDE-specific initialization.
        """
        super().__init__(config)

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
        result = self.runner.run(prompt_collection)
        if not isinstance(result, ResponseWrapper):
            raise TypeError("Expected result to be a ResponseWrapper, got {}".format(type(result)))
        return result

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

        # Use the parent class's retrieve_contexts with the hypothetical answers as queries
        return super().retrieve_contexts(responses, **kwargs)

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
        """Execute the HYDE RAG pipeline and return responses.

        Args:
            runner: LLM runner to use for final answer generation
            sys_prompt: System prompt for the final answer generation
            user_prompts: Optional list of user prompts (questions)
            meta_datas: Optional list of metadata for the prompts
            retrieval_queries: Optional retrieval queries
            response_format: Optional response format for structured output

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
            return runner.run(prompt_collection, response_format=response_format)
