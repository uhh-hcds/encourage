"""Tests for the HybridBM25RAG class."""

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import Document, MetaData
from encourage.rag import HybridBM25RAG
from encourage.rag.base.config import HybridBM25RAGConfig


# Create fixture for document UUIDs to make them accessible in tests
@pytest.fixture
def doc_uuids() -> dict[str, uuid.UUID]:
    return {
        "doc1": uuid.uuid4(),
        "doc2": uuid.uuid4(),
        "doc3": uuid.uuid4(),
        "doc4": uuid.uuid4(),
        "doc5": uuid.uuid4(),
    }


# Sample documents for testing
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
            content="Natural Language Processing: NLP is used to process and analyze text data",
            meta_data=MetaData({"source": "article1", "title": "Natural Language Processing"}),
            id=doc_uuids["doc3"],
        ),
        Document(
            content="Machine Learning: Machine learning is a subset of artificial intelligence",
            meta_data=MetaData({"source": "article2", "title": "Machine Learning"}),
            id=doc_uuids["doc4"],
        ),
        Document(
            content="Deep Learning: Deep learning uses neural networks with multiple layers",
            meta_data=MetaData({"source": "paper1", "title": "Deep Learning"}),
            id=doc_uuids["doc5"],
        ),
    ]


# Mock embedding function
@pytest.fixture
def mock_embedding_function() -> str:
    return "all-MiniLM-L6-v2"  # Return a valid model name instead of a function


# Fixture for HybridBM25RAG instance
@pytest.fixture
def hybrid_rag(sample_documents: list[Document], mock_embedding_function: str) -> HybridBM25RAG:
    # Mock SentenceTransformerEmbeddingFunction
    with (
        patch("encourage.rag.base_impl.embedding_functions.SentenceTransformerEmbeddingFunction"),
        patch("encourage.rag.base_impl.VectorStore") as mock_vector_store,
        patch("encourage.rag.base_impl.ChromaClient") as mock_chroma,
    ):
        # Configure the mock vector store
        vs_instance = mock_vector_store.return_value

        # Mock the query method to return documents based on a simple pattern
        def mock_query(
            collection_name: str, query: list[str], top_k: int, **kwargs: Any
        ) -> list[list[Document]]:
            results = []
            for q in query:
                # Simple mock logic: return docs containing words from the query
                matched_docs = []
                for doc in sample_documents:
                    if any(word.lower() in doc.content.lower() for word in q.split()):
                        matched_docs.append(doc)
                results.append(matched_docs[:top_k])
            return results

        # Create a mock query method first, then apply side_effect to it
        vs_instance.query = MagicMock()
        vs_instance.query.side_effect = mock_query

        chroma_instance = mock_chroma.return_value
        chroma_instance.create_collection.return_value = None
        chroma_instance.insert_documents.return_value = None

        # Fix the ChromaClient query mock in the same way
        chroma_instance.query = MagicMock()
        chroma_instance.query.side_effect = mock_query

        # Create the HybridBM25RAG instance
        config = HybridBM25RAGConfig(
            context_collection=sample_documents,
            collection_name="test_collection",
            embedding_function=mock_embedding_function,
            top_k=3,
            alpha=0.6,
            beta=0.4,
        )

        return HybridBM25RAG(config=config)


class TestHybridBM25RAG:
    """Test class for HybridBM25RAG."""

    def test_initialization(
        self, sample_documents: list[Document], mock_embedding_function: str
    ) -> None:
        """Test that HybridBM25RAG initializes correctly."""
        # Mock SentenceTransformerEmbeddingFunction to avoid actual model loading
        with (
            patch(
                "encourage.rag.base_impl.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
            patch("encourage.rag.base_impl.ChromaClient") as mock_chroma,
        ):
            chroma_instance = mock_chroma.return_value
            chroma_instance.create_collection.return_value = None
            chroma_instance.insert_documents.return_value = None

            # Create a HybridBM25RAG instance
            config = HybridBM25RAGConfig(
                context_collection=sample_documents,
                collection_name="test_collection",
                embedding_function=mock_embedding_function,
                top_k=3,
                alpha=0.6,
                beta=0.4,
            )
            rag: HybridBM25RAG = HybridBM25RAG(config=config)

            # Verify that BM25 index was created
            assert hasattr(rag, "bm25_index")
            assert len(rag.document_texts) == len(sample_documents)
            assert rag.alpha == 0.6
            assert rag.beta == 0.4

    def test_invalid_weights(
        self, sample_documents: list[Document], mock_embedding_function: str
    ) -> None:
        """Test that initialization fails with invalid weights."""
        # Mock dependencies to avoid actual initialization
        with (
            patch(
                "encourage.rag.base_impl.embedding_functions.SentenceTransformerEmbeddingFunction"
            ),
            patch("encourage.rag.base_impl.ChromaClient"),
        ):
            # Alpha + Beta != 1
            with pytest.raises(AssertionError):
                config = HybridBM25RAGConfig(
                    context_collection=sample_documents,
                    collection_name="test_collection",
                    embedding_function=mock_embedding_function,
                    top_k=3,
                    alpha=0.3,
                    beta=0.3,
                )
                HybridBM25RAG(config=config)

            # Alpha outside [0,1]
            with pytest.raises(AssertionError):
                config = HybridBM25RAGConfig(
                    context_collection=sample_documents,
                    collection_name="test_collection",
                    embedding_function=mock_embedding_function,
                    top_k=3,
                    alpha=1.3,
                    beta=0.3,
                )
                HybridBM25RAG(config=config)

    def test_retrieve_contexts(
        self, hybrid_rag: HybridBM25RAG, doc_uuids: dict[str, uuid.UUID]
    ) -> None:
        """Test the retrieve_contexts method returns correctly ranked documents."""
        queries = ["programming in Python"]
        results = hybrid_rag.retrieve_contexts(query_list=queries)

        # Check that we got results
        assert len(results) == 1
        assert len(results[0]) > 0

        # The Python document should be ranked highly
        doc_ids = [str(doc.id) for doc in results[0]]
        assert str(doc_uuids["doc1"]) in doc_ids

    def test_hybrid_ranking(
        self,
        hybrid_rag: HybridBM25RAG,
        sample_documents: list[Document],
        doc_uuids: dict[str, uuid.UUID],
    ) -> None:
        """Test the hybrid ranking functionality."""
        # Mock vector and BM25 docs
        vector_docs = [sample_documents[0], sample_documents[2]]  # Python and NLP
        query = "programming language Python"

        # Get sparse retrieval results
        sparse_docs, sparse_scores = hybrid_rag._retrieve_sparse_results(query)

        # Calculate hybrid scores
        scored_docs = hybrid_rag._compute_hybrid_scores(vector_docs, sparse_docs, sparse_scores)

        # Sort by score and get top_k docs
        hybrid_results = [doc for _, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)][
            : hybrid_rag.top_k
        ]

        # Python document should be ranked first as it appears in both lists
        assert hybrid_results[0].id == doc_uuids["doc1"]

        # Should return at most top_k results
        assert len(hybrid_results) <= hybrid_rag.top_k

        # Alternative approach: use _rank_documents directly
        ranked_docs = hybrid_rag._rank_documents(vector_docs, query)
        assert ranked_docs[0].id == doc_uuids["doc1"]
        assert len(ranked_docs) <= hybrid_rag.top_k

    def test_run_method(self, hybrid_rag: HybridBM25RAG) -> None:
        """Test the run method with a mock runner."""
        # Create a mock BatchInferenceRunner
        mock_runner = MagicMock(spec=BatchInferenceRunner)
        mock_response = MagicMock(spec=ResponseWrapper)
        mock_runner.run.return_value = mock_response

        # Run with simple prompts
        sys_prompt = "You are a helpful assistant."
        user_prompts = ["Tell me about Python"]

        with patch("encourage.rag.base_impl.PromptCollection"):
            response = hybrid_rag.run(
                runner=mock_runner, sys_prompt=sys_prompt, user_prompts=user_prompts
            )

            # Verify runner was called
            mock_runner.run.assert_called_once()
            assert response == mock_response

    def test_empty_query(self, hybrid_rag: HybridBM25RAG) -> None:
        """Test behavior with empty query."""
        results = hybrid_rag.retrieve_contexts(query_list=[""])

        # Should still return results (though possibly not relevant ones)
        assert len(results) == 1

    def test_multiple_queries(self, hybrid_rag: HybridBM25RAG) -> None:
        """Test handling multiple queries."""
        queries = ["Python programming", "machine learning", "neural networks"]

        results = hybrid_rag.retrieve_contexts(query_list=queries)

        # Should return results for each query
        assert len(results) == len(queries)

        # First query should return Python-related docs
        assert any("python" in doc.content.lower() for doc in results[0])

        # Second query should return ML-related docs
        assert any("machine learning" in doc.content.lower() for doc in results[1])

        # Third query should return neural network-related docs
        assert any("neural" in doc.content.lower() for doc in results[2])
