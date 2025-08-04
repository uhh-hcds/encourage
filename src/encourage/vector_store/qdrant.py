"""Qdrant vector store implementation."""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.conversions import common_types

from encourage.prompts.context import Document
from encourage.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QdrantCustomClient(VectorStore):
    """Qdrant vector store implementation."""

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        model_name: str = "BAAI/bge-small-en",
    ):
        super().test_connection(url, port)
        self.client = QdrantClient(url=url, port=port)
        self.aclient = AsyncQdrantClient(url=url, port=port)
        self.client.set_model(model_name)
        self.aclient.set_model(model_name)
        logger.info(f"Qdrant client initialized at {url}:{port}")

    def create_collection(
        self,
        collection_name: str,
        distance: models.Distance = models.Distance.COSINE,
        size: int = 1024,
        overwrite: bool = False,
    ) -> None:
        """Create a Qdrant collection."""
        if overwrite and self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            logger.info(f"Collection {collection_name} deleted")
        self.client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(distance=distance, size=size),
        )
        logger.info(f"Collection {collection_name} successfully created")

    def count_documents(self, collection_name: str, embedding_function: Any = "") -> int:
        """Count documents in a Qdrant collection."""
        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        return int(self.client.count(collection_name).count) or 0

    def insert_documents(
        self,
        collection_name: str,
        documents: list[Document],
        batch_size: int = 2000,
        overwrite: bool = False,
    ) -> None:
        """Insert documents from a JSON file."""
        logger.warning("Batch size is not implemented. All documents will be inserted at once.")

        if not self.client.collection_exists(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")

        if self.client.collection_exists(collection_name) and overwrite:
            self.delete_collection(collection_name)
            logger.info(f"Collection {collection_name} deleted")
            self.create_collection(collection_name)

        ids = [str(doc.id) for doc in documents]
        content = [doc.content for doc in documents]
        meta_datas = [doc.meta_data for doc in documents]
        self.client.add(
            collection_name,
            documents=content,
            metadata=meta_datas,  # type: ignore
            ids=ids,
        )
        logger.info(f"{len(documents)} documents inserted into collection {collection_name}.")

    def get_or_create_collection(self, collection_name: str) -> common_types.CollectionInfo:
        """Get or create a Qdrant collection."""
        if not self.client.collection_exists(collection_name):
            self.create_collection(collection_name)
        return self.client.get_collection(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """Delete a Qdrant collection."""
        self.client.delete_collection(collection_name)

    def query(
        self,
        collection_name: str,
        query: list,
        top_k: int = 10,
        batch_size: int = 2000,
        **kwargs: Any,
    ) -> list:
        """Query a Qdrant collection."""
        logger.warning("Batch size is not implemented. All queries will be executed at once.")
        return self.client.search(collection_name, query, limit=top_k, **kwargs)
