"""Tests for the BM25RAG class."""

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import Document, MetaData
from encourage.rag import BM25RAG
from encourage.rag.base.config import BM25RAGConfig


@pytest.fixture
def doc_uuids() -> dict[str, uuid.UUID]:
    return {"doc1": uuid.uuid4(), "doc2": uuid.uuid4(), "doc3": uuid.uuid4()}


@pytest.fixture
def sample_documents(doc_uuids: dict[str, uuid.UUID]) -> list[Document]:
    return [
        Document(
            content="Python Programming: Python is a programming language",
            meta_data=MetaData({"source": "book1", "title": "Python Programming"}),
            id=doc_uuids["doc1"],
        ),
        Document(
            content="Data Science: Data science involves statistics and machine learning",
            meta_data=MetaData({"source": "book2", "title": "Data Science"}),
            id=doc_uuids["doc2"],
        ),
        Document(
            content="Machine Learning: Machine learning is a subset of artificial intelligence",
            meta_data=MetaData({"source": "article2", "title": "Machine Learning"}),
            id=doc_uuids["doc3"],
        ),
    ]


@pytest.fixture
def mock_embedding_function() -> str:
    return "all-MiniLM-L6-v2"


@pytest.fixture
def bm25_rag(sample_documents: list[Document], mock_embedding_function: str) -> BM25RAG:
    with (
        patch("encourage.rag.base_impl.embedding_functions.SentenceTransformerEmbeddingFunction"),
        patch("encourage.rag.base_impl.VectorStore") as mock_vector_store,
        patch("encourage.rag.base_impl.ChromaClient") as mock_chroma,
    ):
        vs_instance = mock_vector_store.return_value

        def mock_query(
            collection_name: str, query: list[str], top_k: int, **kwargs: Any
        ) -> list[list[Document]]:
            results = []
            for q in query:
                matched = [
                    d
                    for d in sample_documents
                    if any(w.lower() in d.content.lower() for w in q.split())
                ]
                results.append(matched[:top_k])
            return results

        vs_instance.query = MagicMock()
        vs_instance.query.side_effect = mock_query

        chroma_instance = mock_chroma.return_value
        chroma_instance.create_collection.return_value = None
        chroma_instance.insert_documents.return_value = None
        chroma_instance.query = MagicMock()
        chroma_instance.query.side_effect = mock_query

        config = BM25RAGConfig(
            context_collection=sample_documents,
            collection_name="test_collection",
            embedding_function=mock_embedding_function,
            top_k=2,
        )
        return BM25RAG(config=config)


class TestBM25RAG:
    def test_initialization(
        self, sample_documents: list[Document], mock_embedding_function: str
    ) -> None:
        with (
            patch(
                "encourage.rag.base_impl.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
            patch("encourage.rag.base_impl.ChromaClient") as mock_chroma,
        ):
            chroma_instance = mock_chroma.return_value
            chroma_instance.create_collection.return_value = None
            chroma_instance.insert_documents.return_value = None

            config = BM25RAGConfig(
                context_collection=sample_documents,
                collection_name="test_collection",
                embedding_function=mock_embedding_function,
                top_k=2,
            )
            rag = BM25RAG(config=config)
            assert hasattr(rag, "bm25_index")
            assert len(rag.document_texts) == len(sample_documents)

    def test_retrieve_contexts(self, bm25_rag: BM25RAG, doc_uuids: dict[str, uuid.UUID]) -> None:
        queries = ["Python programming"]
        results = bm25_rag.retrieve_contexts(query_list=queries)

        assert len(results) == 1
        assert len(results[0]) > 0
        ids = [str(d.id) for d in results[0]]
        assert str(doc_uuids["doc1"]) in ids

    def test_top_k_limit(self, bm25_rag: BM25RAG) -> None:
        # top_k is 2 in fixture
        results = bm25_rag.retrieve_contexts(query_list=["machine learning python"])
        assert len(results[0]) <= 2

    def test_run_method(self, bm25_rag: BM25RAG) -> None:
        mock_runner = MagicMock(spec=BatchInferenceRunner)
        mock_response = MagicMock(spec=ResponseWrapper)
        mock_runner.run.return_value = mock_response

        sys_prompt = "You are a helpful assistant."
        user_prompts = ["Tell me about Python"]

        with patch("encourage.rag.base_impl.PromptCollection"):
            response = bm25_rag.run(
                runner=mock_runner, sys_prompt=sys_prompt, user_prompts=user_prompts
            )
            mock_runner.run.assert_called_once()
            assert response == mock_response
