"""Module containing various RAG method implementations as classes."""

from typing import Any, override

from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.utils.llm_mock import create_mock_response_wrapper


class NoContext(BaseRAG):
    """Class for no context."""

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
        """Initialize only with metadata."""
        # Call parent's init with interface parameters
        self.template_name = template_name
        self.collection_name = collection_name
        self.retrieval_only = retrieval_only

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
        # Use template_name from class instance

        template_name = self.template_name

        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            meta_datas=meta_datas,
            template_name=template_name,
        )

        if self.retrieval_only:
            return create_mock_response_wrapper(prompt_collection)
        else:
            return runner.run(prompt_collection, response_format=response_format)
