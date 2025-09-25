"""Abstract class for vector store implementations."""

import logging
import socket
from abc import ABC, abstractmethod
from typing import Any

from encourage.prompts.context import Document

logger = logging.getLogger(__name__)


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
    ) -> Any:
        """Create a collection."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    def count_documents(self, collection_name: str, embedding_function: Any = "") -> int:
        """Count documents in a collection."""
        pass

    @abstractmethod
    def insert_documents(
        self, collection_name: str, documents: list[Document], batch_size: int
    ) -> None:
        """Insert documents."""
        pass

    @abstractmethod
    def get_collection(self, collection_name: str) -> Any:
        """Get a collection by name.

        Args:
            collection_name: The name of the collection to retrieve

        Returns:
            Collection object (implementation-specific type)

        Raises:
            ValueError: If the collection does not exist
        """
        pass

    @abstractmethod
    def query(
        self, collection_name: str, query: list, top_k: int, batch_size: int, **kwargs: Any
    ) -> list[list[Document]]:
        """Query the collection."""
        pass
