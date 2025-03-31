"""Interface definitions for RAG methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData
from encourage.vector_store import VectorStore


class RAGMethodInterface(ABC):
    """Interface for RAG (Retrieval Augmented Generation) implementations."""

    @abstractmethod
    def __init__(
        self,
        qa_dataset: pd.DataFrame,
        template_name: str,
        collection_name: str,
        embedding_function: Any,
        meta_data_keys: List[str],
        context_key: str = "context",
        question_key: str = "question",
        answer_key: str = "program_answer",
        device: str = "cuda",
        where: Dict[str, str] | None = None,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize RAG method with required configuration."""
        pass

    @abstractmethod
    def create_context_id(
        self, qa_dataset: pd.DataFrame, context_key: str = "context"
    ) -> pd.DataFrame:
        """Create context identifiers for documents in the dataset."""
        pass

    @abstractmethod
    def create_prompt_meta_data(self, answer_key: str = "program_answer") -> List[MetaData]:
        """Create prompt meta data that is used for reference matching for the metrics."""
        pass

    @abstractmethod
    def prepare_contexts_for_db(self, meta_data_keys: List[str]) -> Context:
        """Prepare contexts for the vector database."""
        pass

    @abstractmethod
    def init_db(self, context_collection: Context) -> VectorStore:
        """Initialize the database with contexts."""
        pass

    @abstractmethod
    def retrieve_contexts(
        self,
        query_list: List[str],
        **kwargs: Any,
    ) -> List[Context]:
        """Retrieve relevant contexts from the database."""
        pass

    @abstractmethod
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: List[str] = [],
        retrieval_instruction: List[str] = [],
    ) -> ResponseWrapper:
        """Execute the RAG pipeline and return responses."""
        pass
