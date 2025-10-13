"""Module containing various RAG method implementations as classes."""

from typing import override

from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.config import NoContextConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper


@RAGFactory.register(RAGMethod.NoContext, NoContextConfig)
class NoContext(BaseRAG):
    """Class for no context."""

    def __init__(self, config: NoContextConfig) -> None:
        """Initialize KnownContext with provided configuration.

        Args:
            config (KnownContextConfig): Configuration object with parameters.

        """
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
        """Run inference on no context."""
        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            meta_datas=meta_datas,
            template_name=self.template_name,
        )

        if self.retrieval_only:
            return create_mock_response_wrapper(prompt_collection)
        else:
            return runner.run(prompt_collection, response_format=response_format)
