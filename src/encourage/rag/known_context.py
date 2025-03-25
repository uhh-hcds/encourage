"""Module containing various RAG method implementations as classes."""

from typing import Any

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.rag.naive import NaiveRAG


class KnownContext(NaiveRAG):
    """Class for known context."""

    def __init__(
        self,
        qa_dataset: Any,
        template_name: str,
        collection_name: str,
        top_k: int,
        embedding_function: Any,
        meta_data_keys: list[str],
        context_key: str = "context",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize known context with context and metadata."""
        super().__init__(
            qa_dataset,
            template_name,
            collection_name,
            top_k,
            embedding_function,
            meta_data_keys,
            context_key,
        )

    def get_ground_truth_context(self, context_key: str = "context") -> list[Context]:
        """Create contexts from dataset."""
        documents = [
            Context.from_documents([Document(content=row[context_key], id=row["context_id"])])
            for (_, row) in self.qa_dataset.iterrows()
        ]
        return documents

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        retrieval_instruction: list[str] = [],
        *args: Any,
        **kwargs: Any,
    ) -> ResponseWrapper:
        """Run inference on known context."""
        # Create prompt collection
        self.contexts = self.get_ground_truth_context(self.context_key)
        user_prompts = user_prompts if user_prompts else self.user_prompts

        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=self.metadata,
            template_name=self.template_name,
        )
        return runner.run(prompt_collection)
