"""Module containing various RAG method implementations as classes."""

from typing import override

from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.config import KnownContextConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper


@RAGFactory.register(RAGMethod.KnownContext, KnownContextConfig)
class KnownContext(BaseRAG):
    """RAG implementation for known context retrieval."""

    def __init__(self, config: KnownContextConfig) -> None:
        """Initialize KnownContext with provided configuration.

        Args:
            config (KnownContextConfig): Configuration object with parameters.
            **kwargs: Additional arguments passed to BaseRAG.

        """
        self.context_collection = config.context_collection
        self.template_name = config.template_name
        self.collection_name = config.collection_name
        self.retrieval_only = config.retrieval_only

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
        """Run inference on known context."""
        # For KnownContext, we always use the predefined context collection
        if not self.context_collection:
            self.contexts = []
        elif not isinstance(self.context_collection[0], Context):
            self.contexts = [Context.from_documents([doc]) for doc in self.context_collection]
        else:
            self.contexts = self.context_collection

        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=meta_datas,
            template_name=self.template_name,
        )

        if self.retrieval_only:
            return create_mock_response_wrapper(prompt_collection)
        else:
            return runner.run(prompt_collection, response_format=response_format)
