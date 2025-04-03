"""Module containing various RAG method implementations as classes."""

import logging
from typing import Any

import pandas as pd

from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Document
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)


class SummarizationRAG(BaseRAG):
    """Implementation of RAG for summarization."""

    def __init__(
        self,
        qa_dataset: pd.DataFrame,
        template_name: str,
        collection_name: str,
        top_k: int,
        embedding_function: str,
        meta_data_keys: list[str],
        runner: BatchInferenceRunner,
        context_key: str = "context",
        answer_key: str = "answer",
        additional_prompt: str = "",
        where: dict[str, str] | None = None,
        retrieval_only: bool = False,
    ):
        """Initialize RAG method with configuration."""
        self.template_name = template_name
        self.context_key = context_key
        qa_dataset = self.create_summaries(runner, additional_prompt, qa_dataset)
        super().__init__(
            qa_dataset=qa_dataset,
            template_name=template_name,
            collection_name=collection_name,
            top_k=top_k,
            embedding_function=embedding_function,
            meta_data_keys=meta_data_keys,
            context_key="summary",
            answer_key=answer_key,
            where=where,
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
        )

    def create_summaries(
        self,
        runner: BatchInferenceRunner,
        additional_prompt: str,
        qa_dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create summary from the QA dataset."""
        unique_contexts = list(qa_dataset[self.context_key].unique())
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=additional_prompt,
            user_prompts=unique_contexts,
            template_name=self.template_name,
        )
        responses = runner.run(prompt_collection)
        sum_mapping = {
            context: response.response for response, context in zip(responses, unique_contexts)
        }
        qa_dataset["summary"] = qa_dataset[self.context_key].map(sum_mapping)
        return qa_dataset


class SummarizationContextRAG(BaseRAG):
    """Implementation of RAG with context preserving summarization."""

    def __init__(
        self,
        qa_dataset: pd.DataFrame,
        template_name: str,
        collection_name: str,
        top_k: int,
        embedding_function: str,
        meta_data_keys: list[str],
        runner: BatchInferenceRunner,
        context_key: str = "context",
        answer_key: str = "answer",
        additional_prompt: str = "",
        where: dict[str, str] | None = None,
        retrieval_only: bool = False,
    ):
        """Initialize RAG method with configuration."""
        self.template_name = template_name
        self.context_key = context_key
        self.qa_dataset = self.create_context_id(qa_dataset, context_key)
        qa_dataset = self.create_summaries(runner, additional_prompt, qa_dataset)
        super().__init__(
            qa_dataset=qa_dataset,
            template_name=template_name,
            collection_name=collection_name,
            top_k=top_k,
            embedding_function=embedding_function,
            meta_data_keys=meta_data_keys,
            context_key="summary",
            answer_key=answer_key,
            where=where,
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
        )

    def create_summaries(
        self,
        runner: BatchInferenceRunner,
        additional_prompt: str,
        qa_dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create summary from the QA dataset while preserving original contexts."""
        unique_contexts = list(qa_dataset[self.context_key].unique())
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=additional_prompt,
            user_prompts=unique_contexts,
            template_name=self.template_name,
        )
        responses = runner.run(prompt_collection)
        sum_mapping = {
            context: response.response for response, context in zip(responses, unique_contexts)
        }
        qa_dataset["summary"] = qa_dataset[self.context_key].map(sum_mapping)

        # Preserve original contexts for later use
        self.original_contexts = dict(zip(qa_dataset["context_id"], qa_dataset[self.context_key]))

        return qa_dataset

    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts from the database with context preservation.

        This method overrides the parent implementation to ensure that
        the retrieved summaries are replaced with their original contexts.

        Args:
            query_list (list[str]): List of queries to retrieve contexts for.
            **kwargs (Any): Additional parameters for retrieval.

        Returns:
            list[list[Document]]: A list of lists containing documents with
            their original contexts restored.

        """
        # Get contexts using the parent implementation
        document_lists = super().retrieve_contexts(query_list, **kwargs)

        # Replace each summary with its original context for all retrieved documents
        for document_list in document_lists:
            if not document_list or not hasattr(self, "original_contexts"):
                continue
            for doc in document_list:
                original = self.original_contexts.get(str(doc.id))
                if original:
                    doc.content = original
        return document_lists
