"""Chroma vector store implementation."""

import logging
from contextlib import suppress
from typing import Any

import chromadb
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from encourage.vector_store.vector_store import VectorStore, VectorStoreBatch

logger = logging.getLogger(__name__)


class ChromaClient(VectorStore):
    """Chroma vector store implementation."""

    def __init__(self) -> None:
        self.client = chromadb.Client()

    def create_collection(
        self,
        collection_name: str,
        distance: str = "cosine",
        size: int = 1000,
        overwrite: bool = False,
        embedding_function: chromadb.EmbeddingFunction | None = None,
    ) -> None:
        """Create a collection."""
        if overwrite:
            with suppress(ValueError):
                self.client.delete_collection(collection_name)
        self.client.create_collection(
            name=collection_name,
            metadata={"distance": distance, "size": size},
            embedding_function=embedding_function,
        )

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        self.client.delete_collection(name=collection_name)

    def insert_documents(
        self, collection_name: str, vector_store_document: VectorStoreBatch
    ) -> None:
        """Insert documents."""
        collection = self.client.get_collection(name=collection_name)

        documents = [document.content for document in vector_store_document.documents]
        meta_data = [document.meta_data.to_dict() for document in vector_store_document.documents]
        ids = [str(document.id) for document in vector_store_document.documents]

        collection.add(documents=documents, metadatas=meta_data, ids=ids)  # type: ignore

    def query(self, collection_name: str, query: list, top_k: int, **kwargs: Any) -> dict:
        """Query the collection."""
        collection = self.client.get_collection(name=collection_name)
        result = collection.query(query_texts=query, n_results=top_k, **kwargs)
        return dict(result)

    def get_llama_index_class(self, collection_name: str) -> BasePydanticVectorStore:
        """Get the Llama index class."""
        collection = self.client.get_collection(name=collection_name)
        return ChromaVectorStore(chroma_collection=collection)
