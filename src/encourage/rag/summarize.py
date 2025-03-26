"""Module containing various RAG method implementations as classes."""

import logging
from typing import Optional

import pandas as pd

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.rag_enum import NaiveRAG

logger = logging.getLogger(__name__)


class SummarizationRAG(NaiveRAG):
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
        qa_dataset["summary"] = qa_dataset["context"].map(sum_mapping)
        return qa_dataset


class ContextPreservingSummarizationRAG(NaiveRAG):
    """Implementation of RAG that uses summaries that returns original contexts in results."""

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
    ):
        """Initialize RAG method with configuration."""
        self.template_name = template_name
        self.context_key = context_key

        # Create summaries
        qa_dataset = self.create_summaries(runner, additional_prompt, qa_dataset)

        # Ensure the original context is preserved in metadata
        if self.context_key not in meta_data_keys:
            meta_data_keys = meta_data_keys + [self.context_key]

        # Initialize parent with summaries for search
        super().__init__(
            qa_dataset=qa_dataset,
            template_name=template_name,
            collection_name=collection_name,
            top_k=top_k,
            embedding_function=embedding_function,
            meta_data_keys=meta_data_keys,
            context_key="summary",
            answer_key=answer_key,
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

    def _get_contexts_from_db(
        self,
        query_list: list[str],
        meta_datas: Optional[list[MetaData]] = None,
    ) -> list[Context]:
        """Override to retrieve contexts but replace summaries with original contexts."""
        # Get contexts (summaries) using the parent implementation
        contexts = super()._get_contexts_from_db(query_list)

        # Replace summary content with original context content
        for context in contexts:
            for document in context.documents:
                if document.meta_data and self.context_key in document.meta_data.tags:
                    # Replace summary with original context
                    document.content = document.meta_data.tags[self.context_key]
        return contexts

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        retrieval_instruction: list[str] = [],
    ) -> ResponseWrapper:
        """Execute the RAG pipeline with summaries for search but original contexts in responses."""
        return super().run(runner, sys_prompt, user_prompts, retrieval_instruction)
