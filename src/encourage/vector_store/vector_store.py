"""Abstract class for vector store implementations."""

import logging
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.utils.file_manager import FileManager

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreBatch:
    """Vector store batch class."""

    documents: list[Document]

    def __len__(self) -> int:
        return len(self.documents)

    @classmethod
    def from_json(
        cls: Type["VectorStoreBatch"],
        path: Path,
        doc_keys: Optional[list[str]] = None,
        meta_keys: Optional[list[str]] = None,
    ) -> "VectorStoreBatch":
        """Load documents and meta_data from a JSON file with specified keys."""
        json_data = FileManager(path).load_jsonlines()

        # Default to all keys if none are provided
        if doc_keys is None:
            doc_keys = list(json_data[0].keys()) if json_data else []
        if meta_keys is None:
            meta_keys = list(json_data[0].keys()) if json_data else []

        def flatten_dict(d: dict) -> str:
            return ", ".join(f"{k}: {v}" for k, v in d.items())

        # Create documents by pairing flattened documents with their respective meta_data
        documents = [
            Document(
                content=flatten_dict({key: entry.get(key, "") for key in doc_keys}),
                meta_data=MetaData(tags={key: entry.get(key, "") for key in meta_keys}),
            )
            for entry in json_data
        ]

        return cls(documents=documents)


class VectorStore(ABC):
    """Abstract class for vector store implementations."""

    def test_connection(self, url: str, port: int) -> None:
        """Test the connection to the vector store."""
        try:
            with socket.create_connection((url, port), timeout=5):
                logger.info(f"Port {port} is reachable.")
        except (socket.timeout, ConnectionRefusedError) as e:
            raise RuntimeError(
                f"Port {port} is NOT reachable on host: {url}. Docker Container is not reachable! Error: {e}"  # noqa: E501
            ) from e

    @abstractmethod
    def create_collection(
        self, collection_name: str, distance: Any, size: int, overwrite: bool
    ) -> None:
        """Create a collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    def insert_documents(
        self, collection_name: str, vector_store_document: VectorStoreBatch
    ) -> None:
        """Insert documents."""
        pass

    @abstractmethod
    def query(self, collection_name: str, query: list, top_k: int, **kwargs: Any) -> list | dict:
        """Query the collection."""
        pass

    @abstractmethod
    def get_llama_index_class(self, collection_name: str) -> BasePydanticVectorStore:
        """Get the Llama index class."""
        pass
