import unittest
import uuid
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.rag.reranker import RerankerRAG


class TestBaseRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_collection_{uuid.uuid4()}"
        # Create test documents directly instead of dataframe
        self.documents = [
            Document(
                id=uuid.uuid4(), content="AI is a field of study.", meta_data=MetaData({"id": "1"})
            ),
            Document(
                id=uuid.uuid4(), content="ML is a subset of AI.", meta_data=MetaData({"id": "2"})
            ),
        ]

        # Initialize RAG with the new interface
        self.rag = BaseRAG(
            context_collection=self.documents,
            template_name="llama3_conv.j2",
            collection_name=self.collection_name,
            top_k=1,
            embedding_function="all-MiniLM-L6-v2",
            device="cpu",
        )

        # Keep a reference to the original queries for testing
        self.queries = ["What is AI?", "Define ML"]

    def tearDown(self):
        self.rag.client.delete_collection(self.collection_name)

    def test_init_db_document_count(self):
        count = self.rag.client.count_documents(self.collection_name)
        self.assertEqual(count, 2)

    def test_retrieve_contexts(self):
        query = ["What is AI?"]
        documents = self.rag.retrieve_contexts(query)
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], list)
        self.assertIsInstance(documents[0][0], Document)
        self.assertGreaterEqual(len(documents[0]), 1)

    def test_run_with_mocked_runner(self):
        runner = create_autospec(BatchInferenceRunner)
        mock_responses = ResponseWrapper(["Mock response 1", "Mock response 2"])
        runner.run.return_value = mock_responses

        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_instruction=["Define ML", "What is AI?"],
        )

        runner.run.assert_called_once()
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(result.response_data, ["Mock response 1", "Mock response 2"])


class TestRerankerRAG(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_reranker_collection_{uuid.uuid4()}"
        # Create test documents
        self.documents = [
            Document(
                id=uuid.uuid4(), content="AI is a field of study.", meta_data=MetaData({"id": "1"})
            ),
            Document(
                id=uuid.uuid4(), content="ML is a subset of AI.", meta_data=MetaData({"id": "2"})
            ),
            Document(
                id=uuid.uuid4(),
                content="Deep learning is a type of ML.",
                meta_data=MetaData({"id": "3"}),
            ),
            Document(
                id=uuid.uuid4(),
                content="Neural networks are used in deep learning.",
                meta_data=MetaData({"id": "4"}),
            ),
        ]

        # Keep a reference to the original queries for testing
        self.queries = ["What is AI?", "Define ML"]

    def tearDown(self):
        if hasattr(self, "rag") and hasattr(self.rag, "client"):
            self.rag.client.delete_collection(self.collection_name)

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_reranker_initialization(self, mock_cross_encoder):
        # Setup mock for CrossEncoder
        mock_cross_encoder.return_value = MagicMock()

        # Initialize RerankerRAG
        reranker_rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_ratio=2.0,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Verify CrossEncoder was initialized with correct arguments
        mock_cross_encoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        # Verify initial_top_k calculation
        self.assertEqual(reranker_rag.initial_top_k, 4)  # top_k * rerank_ratio = 2 * 2.0
        self.assertEqual(reranker_rag.top_k, 2)  # Original top_k preserved

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_retrieve_contexts_with_reranking(self, mock_cross_encoder):
        # Setup the mock for CrossEncoder
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance

        # Set up the mock to return predictable scores when predict is called
        # Higher scores should be better for reranking
        mock_instance.predict.return_value = [0.9, 0.3, 0.8, 0.2]

        # Initialize RerankerRAG with rerank_ratio=2.0, so it retrieves twice top_k documents
        self.rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            rerank_ratio=2.0,
            device="cpu",
        )

        # Test retrieval with reranking
        query = ["What is AI?"]
        documents = self.rag.retrieve_contexts(query)

        # Verify predict was called
        mock_instance.predict.assert_called()

        # Verify correct number of documents returned after reranking
        self.assertEqual(len(documents), 1)  # One query
        self.assertEqual(len(documents[0]), 2)  # top_k = 2

        # Note: since the reranking scores are mocked, we're not testing the actual order,
        # just that the right number of documents is returned

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_run_with_mocked_runner(self, mock_cross_encoder):
        # Setup CrossEncoder mock
        mock_cross_encoder.return_value = MagicMock()
        mock_cross_encoder.return_value.predict.return_value = [0.9, 0.8, 0.7, 0.6]

        # Setup runner mock
        runner = create_autospec(BatchInferenceRunner)
        mock_responses = ResponseWrapper(["Mock reranker response 1", "Mock reranker response 2"])
        runner.run.return_value = mock_responses

        # Initialize RerankerRAG
        self.rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Test run method
        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_instruction=["What is AI?", "Define ML"],
        )

        # Verify runner was called
        runner.run.assert_called_once()

        # Verify response
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(
            result.response_data, ["Mock reranker response 1", "Mock reranker response 2"]
        )

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_reranker_edge_case_empty_documents(self, mock_cross_encoder):
        # Setup CrossEncoder mock
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance
        mock_instance.predict.return_value = []

        # Initialize RerankerRAG
        self.rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Patch the super().retrieve_contexts to return empty results
        with patch.object(BaseRAG, "retrieve_contexts", return_value=[[]]):
            # Test retrieval with reranking for an empty result
            query = ["Invalid query with no results"]
            documents = self.rag.retrieve_contexts(query)

            # Verify predict was not called (no documents to rerank)
            mock_instance.predict.assert_not_called()

            # Verify empty list returned
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(documents[0]), 0)


if __name__ == "__main__":
    unittest.main()
