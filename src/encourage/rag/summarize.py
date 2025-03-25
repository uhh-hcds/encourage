"""Module containing various RAG method implementations as classes."""

import logging

import pandas as pd

from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.rag.rag import NaiveRAG

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
        additional_prompt: str = "",
    ):
        """Initialize RAG method with configuration."""
        self.template_name = template_name
        self.context_key = context_key
        qa_dataset = self.create_summaries(runner, additional_prompt, qa_dataset)
        super().__init__(
            qa_dataset,
            template_name,
            collection_name,
            top_k,
            embedding_function,
            meta_data_keys,
            "summary",
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
