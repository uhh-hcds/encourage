"""Interface definitions for RAG methods."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.vector_store import VectorStore


class RAGMethodInterface(ABC):
    """Interface for RAG (Retrieval Augmented Generation) implementations."""

    @abstractmethod
    def __init__(
        self,
        config: BaseModel,
        **kwargs: Any,
    ) -> None:
        """Initialize RAG method with required configuration."""
        pass

    @abstractmethod
    def init_db(self) -> VectorStore:
        """Initialize the database with contexts."""
        pass

    @abstractmethod
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve relevant contexts from the database."""
        pass

    @abstractmethod
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        meta_datas: list[MetaData] = [],
        retrieval_queries: list[str] = [],
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Execute the RAG pipeline and return responses."""
        pass
