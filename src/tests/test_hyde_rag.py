"""Tests for HYDE RAG method."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.rag.hyde import HydeRAG
from tests.conftest import create_mock_response


@pytest.fixture
def qa_dataset():
    """Create a test QA dataset."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "context": [
                "The capital of France is Paris.",
                "The capital of Germany is Berlin.",
                "The capital of Japan is Tokyo.",
            ],
            "question": [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of Japan?",
            ],
            "program_answer": ["Paris", "Berlin", "Tokyo"],
        }
    )


@pytest.fixture
def mock_batch_runner():
    """Create a mock batch inference runner."""
    mock_runner = MagicMock(spec=BatchInferenceRunner)
    mock_runner.run.return_value = create_mock_response(["Paris", "Berlin", "Tokyo"])
    return mock_runner


@pytest.fixture
def mock_hypothetical_runner():
    """Create a mock hypothetical answer generator."""
    mock_runner = MagicMock(spec=BatchInferenceRunner)

    # Mock hypothetical answers that contain relevant terms for better retrieval
    mock_answers = [
        "Paris is the capital and largest city of France. It is located on the Seine River.",
        "Berlin is the capital and largest city of Germany. It is known for its cultural scene.",
        "Tokyo is the capital of Japan and one of the most populous cities in the world.",
    ]

    mock_runner.run.return_value = create_mock_response(mock_answers)
    return mock_runner


@pytest.fixture
def hyde_rag(qa_dataset, mock_batch_runner, mock_hypothetical_runner, tmp_path):
    """Create a HydeRAG instance with a temporary database."""
    # Use a temporary path for the database
    os.environ["CHROMA_PATH"] = str(tmp_path)

    with patch("chromadb.PersistentClient"):
        # Create HydeRAG instance
        hyde_rag = HydeRAG(
            qa_dataset=qa_dataset,
            template_name="qa",
            collection_name=f"test_collection_{uuid.uuid4()}",
            top_k=1,
            embedding_function="all-MiniLM-L6-v2",
            meta_data_keys=["id"],
            runner=mock_hypothetical_runner,
            device="cpu",
            additional_prompt="Generate a detailed answer for the following question:",
            cache_hypothetical_answers=True,
        )

        # Mock the query method to return appropriate results
        hyde_rag.client.query = MagicMock()

        # For each query, return the corresponding context
        def mock_query_side_effect(collection_name, query, top_k, *args, **kwargs):
            results = []
            for q in query:
                if "France" in q:
                    doc_id = qa_dataset[qa_dataset["context"].str.contains("France")][
                        "context_id"
                    ].iloc[0]
                    results.append(
                        [hyde_rag.qa_dataset[hyde_rag.qa_dataset["context_id"] == doc_id].iloc[0]]
                    )
                elif "Germany" in q:
                    doc_id = qa_dataset[qa_dataset["context"].str.contains("Germany")][
                        "context_id"
                    ].iloc[0]
                    results.append(
                        [hyde_rag.qa_dataset[hyde_rag.qa_dataset["context_id"] == doc_id].iloc[0]]
                    )
                elif "Japan" in q:
                    doc_id = qa_dataset[qa_dataset["context"].str.contains("Japan")][
                        "context_id"
                    ].iloc[0]
                    results.append(
                        [hyde_rag.qa_dataset[hyde_rag.qa_dataset["context_id"] == doc_id].iloc[0]]
                    )
                else:
                    results.append([])
            return results

        hyde_rag.client.query.side_effect = mock_query_side_effect

        yield hyde_rag


def test_hyde_rag_initialization(hyde_rag):
    """Test that the HydeRAG class initializes correctly."""
    assert hyde_rag is not None
    assert hasattr(hyde_rag, "runner")
    assert hasattr(hyde_rag, "generate_hypothetical_answer")
    assert hasattr(hyde_rag, "hypothetical_answer_cache")


def test_generate_hypothetical_answer(hyde_rag, mock_hypothetical_runner):
    """Test the generation of hypothetical answers."""
    query = "What is the capital of France?"

    # First call should use the LLM
    answer = hyde_rag.generate_hypothetical_answer(query)
    assert answer is not None
    assert "Paris" in answer
    mock_hypothetical_runner.run.assert_called_once()

    # Second call should use the cache
    mock_hypothetical_runner.run.reset_mock()
    cached_answer = hyde_rag.generate_hypothetical_answer(query)
    assert cached_answer == answer
    mock_hypothetical_runner.run.assert_not_called()


def test_hyde_retrieval(hyde_rag, mock_hypothetical_runner):
    """Test the retrieval functionality of HYDE."""
    query = ["What is the capital of France?"]

    # Run HYDE retrieval
    hyde_rag._get_contexts_from_db(query)

    # Check that the hypothetical answer was generated
    assert hyde_rag.runner.run.called

    # Check that the query was performed with the hypothetical answer
    assert hyde_rag.client.query.called


def test_hyde_end_to_end(hyde_rag, mock_batch_runner):
    """Test the complete HYDE RAG pipeline."""
    # Set up the test
    sys_prompt = "Answer the question based on the provided context."
    user_prompts = ["What is the capital of France?"]

    # Run the HYDE pipeline
    response = hyde_rag.run(
        sys_prompt=sys_prompt,
        user_prompts=user_prompts,
    )

    # Verify that the hypothetical answer was generated
    assert hyde_rag.runner.run.called

    # Verify that the context retrieval was performed
    assert hyde_rag.client.query.called

    # Verify that the final LLM inference was performed
    # The response should have been returned already since we're using the runner directly

    # Check the response
    assert isinstance(response, ResponseWrapper)
