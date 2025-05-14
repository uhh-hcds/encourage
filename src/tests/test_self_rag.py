"""unittest‑based offline tests for `SelfRAG`.

Run with either `python -m unittest test_self_rag.py` **or** `pytest`.  When
invoked via `unittest`, discovery works because `TestSelfRAG` now inherits from
`unittest.TestCase`.
"""
from __future__ import annotations

import uuid
import unittest
from typing import Any, List
from unittest.mock import MagicMock, patch

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import Document, MetaData

# ---------------------------------------------------------------------------
# SUT import – adjust if your class/module lives elsewhere -------------------
from encourage.rag.self_rag import SelfRAG  # noqa: E402 – deferred import after patches

# ---------------------------------------------------------------------------
# Helper functions -----------------------------------------------------------


def _build_sample_documents() -> List[Document]:
    """Return a deterministic list of five toy `Document`s."""
    texts = [
        "Python Programming: Python is a programming language",
        "Data Science: Data science involves statistics and machine learning",
        "Natural Language Processing: NLP is used to process and analyse text data",
        "Machine Learning: Machine learning is a subset of artificial intelligence",
        "Deep Learning: Deep learning uses neural networks with multiple layers",
    ]
    return [
        Document(content=txt, meta_data=MetaData({"source": f"src{i}"}), id=uuid.UUID(int=i + 1))  # stable UUIDs
        for i, txt in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# TestCase -------------------------------------------------------------------


class TestSelfRAG(unittest.TestCase):
    """Offline smoke tests for the `SelfRAG` pipeline."""

    def setUp(self) -> None:  # noqa: D401 – unittest hook
        self.sample_docs = _build_sample_documents()
        self.mock_embed = "all-MiniLM-L6-v2"

        # Patch external components (SentenceTransformer, VectorStore, Chroma)
        patcher1 = patch("encourage.rag.base_impl.SentenceTransformerEmbeddingFunction")
        patcher2 = patch("encourage.rag.base_impl.VectorStore")
        patcher3 = patch("encourage.rag.base_impl.ChromaClient")

        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(patcher3.stop)

        self.mock_st_embed = patcher1.start()
        self.mock_vs_cls = patcher2.start()
        self.mock_chroma_cls = patcher3.start()

        # VectorStore query stub ------------------------------------------------
        vs_instance = self.mock_vs_cls.return_value

        def mock_query(collection_name: str, query: List[str], top_k: int, **kwargs: Any):  # type: ignore[unused-argument]
            results: List[List[Document]] = []
            for q in query:
                matches: List[Document] = []
                for d in self.sample_docs:
                    if any(word.lower() in d.content.lower() for word in q.split()):
                        matches.append(d)
                results.append(matches[:top_k])
            return results

        vs_instance.query = MagicMock(side_effect=mock_query)

        # Chroma – not used but patch for safety
        self.mock_chroma_cls.return_value.create_collection.return_value = None

        # Instantiate SUT
        self.rag = SelfRAG(
            context_collection=self.sample_docs,
            collection_name="test_collection",
            embedding_function=self.mock_embed,
            top_k=3,
            reflection_rounds=0,
        )

    # -------------------------------------------------------------------
    # Tests --------------------------------------------------------------

    def test_initialisation(self) -> None:
        self.assertEqual(self.rag.top_k, 3)
        self.assertEqual(self.rag.reflection_rounds, 0)

    def test_retrieve_contexts(self) -> None:
        out = self.rag.retrieve_contexts(["Python programming"])
        self.assertEqual(len(out), 1)
        self.assertGreater(len(out[0]), 0)
        # First document (UUID int=1) should be among results
        ids = [str(doc.id) for doc in out[0]]
        self.assertIn(str(uuid.UUID(int=1)), ids)

    def test_run_returns_wrapper(self) -> None:
        """`run` should produce a `ResponseWrapper` even when internals are stubbed."""
        mock_runner = MagicMock(spec=BatchInferenceRunner)

        # Short‑circuit SelfRAG internals so `mock_runner` isn't called
        with (
            patch.object(self.rag, "_decide_retrieval", return_value=[False]),
            patch.object(self.rag, "_retrieve_and_setup_contexts", return_value=[None]),
            patch.object(
                self.rag,
                "_batch_process_with_self_reflection",
                return_value=(["stubbed answer"], {}),
            ),
        ):
            wrapper = self.rag.run(
                runner=mock_runner,
                sys_prompt="You are a helpful assistant.",
                user_prompts=["Tell me about Python"],
            )

        mock_runner.run.assert_not_called()
        self.assertIsInstance(wrapper, ResponseWrapper)
        self.assertEqual(wrapper.get_responses(), ["stubbed answer"])


# ---------------------------------------------------------------------------
# Entry‑point for `python test_self_rag.py` ----------------------------------

if __name__ == "__main__":
    unittest.main()
