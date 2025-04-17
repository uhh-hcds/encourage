"""Module containing various RAG method implementations as classes."""

from typing import Any, override

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper


class KnownContext(BaseRAG):
    """Class for known context."""

    def __init__(
        self,
        context_collection: list[Document],
        template_name: str,
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        device: str = "cuda",
        runner: BatchInferenceRunner | None = None,
        where: dict[str, str] | None = None,
        retrieval_only: bool = False,
        additional_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize known context with context and metadata."""
        # Call parent's init with interface parameters
        self.context_collection = context_collection
        self.template_name = template_name
        self.collection_name = collection_name
        self.retrieval_only = retrieval_only

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
        """Run inference on known context."""
        # For KnownContext, we always use the predefined context collection
        # instead of retrieving based on instructions
        self.contexts = [Context.from_documents([doc]) for doc in self.context_collection]

        # Use provided template_name or fall back to self.template_name
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
            return create_mock_response_wrapper(prompt_collection)
        else:
            return runner.run(prompt_collection)
