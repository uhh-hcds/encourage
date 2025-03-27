"""Module containing various RAG method implementations as classes."""

import logging

import pandas as pd

from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context
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
        where: dict[str, str] = None,
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


class SummarizationContextRAG(NaiveRAG):
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
        where: dict[str, str] = None,
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

    def _get_contexts_from_db(
        self,
        query_list: list[str],
    ) -> list[Context]:
        """Get contexts from the database with context preservation.

        This overrides the parent method to replace the summaries with
        the original contexts after retrieval.
        """
        # Get contexts using the parent implementation
        contexts = super()._get_contexts_from_db(query_list)

        # Replace each summary with its original context
        for _, context in enumerate(contexts):
            if hasattr(self, "original_contexts") and context.documents:
                for j, doc in enumerate(context.documents):
                    if str(doc.id) in self.original_contexts:
                        # Replace the summary content with the original context
                        context.documents[j].content = self.original_contexts[str(doc.id)]

        return contexts
