"""Chroma vector store implementation."""

import logging
import uuid
from contextlib import suppress
from typing import Any, Sequence, cast

import chromadb
from chromadb import Collection, EmbeddingFunction, Where
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from tqdm import tqdm

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
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),
    ) -> chromadb.Collection:
        """Create a collection."""
        if overwrite:
            with suppress(NotFoundError):
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
        batch_size: int = 2000,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),
    ) -> None:
        """Insert documents."""
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

        document_content = [document.content for document in documents]
        meta_data = [
            doc_meta if doc_meta else {"__dummy__": "placeholder"}
            for doc_meta in (document.meta_data.to_dict() for document in documents)
        ]
        ids = [str(document.id) for document in documents]

        for i in tqdm(range(0, len(document_content), batch_size), desc="Inserting documents"):
            batch_documents = document_content[i : i + batch_size]
            batch_metadatas = meta_data[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            collection.add(documents=batch_documents, metadatas=batch_metadatas, ids=batch_ids)  # type: ignore

    def count_documents(
        self,
        collection_name: str,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),
    ) -> int:
        """Count the number of documents in the collection."""
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        return collection.count()

    def get_collection(
        self,
        collection_name: str,
    ) -> chromadb.Collection:
        """Get a collection by name.

        Args:
            collection_name: The name of the collection to retrieve

        Returns:
            ChromaDB Collection object

        Raises:
            NotFoundError: If the collection does not exist
        """
        try:
            return self.client.get_collection(
            name=collection_name
        )
        except NotFoundError as e:
            raise ValueError(f"Collection '{collection_name}' does not exist.") from e

    def list_collections(self) -> Sequence[Collection]:
        """Get the list of collections."""
        return self.client.list_collections()

    def query(
        self,
        collection_name: str,
        query: str | list[str],
        top_k: int,
        batch_size: int = 200,
        embedding_function: EmbeddingFunction = DefaultEmbeddingFunction(),
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
            batch_size: The number of queries to process in a single batch
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
        # Batch processing for queries larger than a certain size
        keys = ["ids", "documents", "metadatas", "distances"]
        all_results: dict[Any, Any] = {key: [] for key in keys}

        for i in tqdm(range(0, len(query), batch_size), desc="Querying documents"):
            batch_result = collection.query(
                query_texts=query[i : i + batch_size],
                n_results=top_k,
                where=where_chromadb,
                **kwargs,
            )
            for key, values in batch_result.items():
                if key in all_results:
                    all_results[key].extend(values)

        result = all_results

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
