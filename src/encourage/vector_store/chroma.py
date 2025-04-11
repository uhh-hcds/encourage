"""Chroma vector store implementation."""

import logging
import uuid
from contextlib import suppress
from typing import Any, Sequence, cast

import chromadb
from chromadb import EmbeddingFunction, Where
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChromaClient(VectorStore):
    """Chroma vector store implementation."""

    def __init__(self) -> None:
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))

    def create_collection(
        self,
        collection_name: str,
        distance: str = "cosine",
        size: int = 1000,
        overwrite: bool = False,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),  # type: ignore
    ) -> chromadb.Collection:
        """Create a collection."""
        if overwrite:
            with suppress(ValueError):
                self.client.delete_collection(collection_name)
        return self.client.create_collection(
            name=collection_name,
            metadata={"distance": distance, "size": size},
            embedding_function=embedding_function,
        )

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        self.client.delete_collection(name=collection_name)

    def insert_documents(
        self,
        collection_name: str,
        documents: list[Document],
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),  # type: ignore
    ) -> None:
        """Insert documents."""
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

        document_content = [document.content for document in documents]
        meta_data = [document.meta_data.to_dict() for document in documents]
        ids = [str(document.id) for document in documents]

        collection.add(documents=document_content, metadatas=meta_data, ids=ids)  # type: ignore

    def count_documents(
        self,
        collection_name: str,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),  # type: ignore
    ) -> int:
        """Count the number of documents in the collection."""
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        return collection.count()

    def list_collections(self) -> Sequence[str]:
        """Get the list of collections."""
        return self.client.list_collections()

    def query(
        self,
        collection_name: str,
        query: str | list[str],
        top_k: int,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),  # type: ignore
        where: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Query the collection with a list of queries.

        Args:
            collection_name: The name of the collection to query
            query: The query or list of queries to search for
            top_k: The number of results to return per query
            embedding_function: The embedding function to use
            where: Optional metadata filtering condition in the form {"key": "value"}
            **kwargs: Additional parameters to pass to the query

        Returns:
            A list of lists of Documents, one list per query

        """
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        if isinstance(query, str):
            query = [query]

        where_chromadb = cast(Where, where)
        result = collection.query(
            query_texts=query, n_results=top_k, where=where_chromadb, **kwargs
        )

        ids = cast(list[list[str]], result.get("ids"))
        docs = cast(list[list[str]], result.get("documents"))
        metadatas = cast(list[list[dict[str, str]]], result.get("metadatas"))
        distances = cast(list[list[float]], result.get("distances"))

        return [
            [
                Document(
                    id=uuid.UUID(ids[i][j]),
                    content=docs[i][j],
                    meta_data=MetaData(tags=metadatas[i][j]),
                    distance=distances[i][j],
                    score=1 - distances[i][j],
                )
                for j in range(len(ids[i]))
            ]
            for i in range(len(query))
        ]

    def get_llama_index_class(
        self,
        collection_name: str,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),  # type: ignore
    ) -> BasePydanticVectorStore:
        """Get the Llama index class."""
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        return ChromaVectorStore(chroma_collection=collection)
