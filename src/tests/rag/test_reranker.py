import unittest
import uuid
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import Document, MetaData
from encourage.rag import BaseRAG, RerankerRAG
from encourage.rag.base.config import RerankerRAGConfig
from tests.fake_responses import create_responses


class TestRerankerRAG(unittest.TestCase):
    def setUp(self):
        responses_content_list = [
            "This is a generated answer.",
            "Another generated answer.",
        ]
        document_list = [
            [
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"), content="Here is an example content"
                ),
                Document(id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content="Here is example content"),
            ],
            [
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"), content="Here is an example content"
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "4"),
                    content="Here is an example content with extra",
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "3"),
                    content="Here is an example content with extra",
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"),
                    content="Here is an example content with extra",
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "5"),
                    content="Here is an example content with extra",
                ),
            ],
        ]
        meta_data_list = [
            MetaData(
                tags={
                    "reference_answer": "This is a generated answer.",
                    "reference_document": document_list[0][1],
                }
            ),
            MetaData(
                tags={
                    "reference_answer": "Another reference answer.",
                    "reference_document": document_list[0][0],
                }
            ),
        ]
        self.responses = ResponseWrapper(
            create_responses(2, responses_content_list, document_list, meta_data_list)
        )
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

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_initialization(self, mock_cross_encoder):
        # Setup mock for CrossEncoder
        mock_cross_encoder.return_value = MagicMock()

        # Initialize RerankerRAG
        config = RerankerRAGConfig(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            rerank_ratio=2.0,
            device="cpu",
        )
        reranker_rag = RerankerRAG(config=config)

        # Verify CrossEncoder was initialized with correct arguments
        mock_cross_encoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        # Verify initial_top_k calculation
        self.assertEqual(reranker_rag.initial_top_k, 4)  # top_k * rerank_ratio = 2 * 2.0
        self.assertEqual(reranker_rag.top_k, 2)  # Original top_k preserved

    @patch("sentence_transformers.CrossEncoder")
    def test_retrieve_contexts_with_reranking(self, mock_cross_encoder):
        # Setup the mock for CrossEncoder
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance

        # Set up the mock to return predictable scores when predict is called
        # Higher scores should be better for reranking
        mock_instance.predict.return_value = [0.9, 0.3, 0.8, 0.2]

        # Initialize RerankerRAG with rerank_ratio=2.0, so it retrieves twice top_k documents
        config = RerankerRAGConfig(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            rerank_ratio=2.0,
            device="cpu",
        )
        self.rag = RerankerRAG(config=config)

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

    @patch("sentence_transformers.CrossEncoder")
    def test_run_with_mocked_runner(self, mock_cross_encoder):
        # Setup CrossEncoder mock
        mock_cross_encoder.return_value = MagicMock()
        mock_cross_encoder.return_value.predict.return_value = [0.9, 0.8, 0.7, 0.6]

        # Setup runner mock
        runner = create_autospec(BatchInferenceRunner)
        runner.run.return_value = self.responses

        # Initialize RerankerRAG
        config = RerankerRAGConfig(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            device="cpu",
        )
        self.rag = RerankerRAG(config=config)

        # Test run method
        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_queries=["What is AI?", "Define ML"],
        )

        # Verify runner was called
        runner.run.assert_called_once()

        # Verify response
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(result.response_data, self.responses.response_data)

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_edge_case_empty_documents(self, mock_cross_encoder):
        # Setup CrossEncoder mock
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance
        mock_instance.predict.return_value = []

        # Initialize RerankerRAG
        config = RerankerRAGConfig(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            device="cpu",
        )
        self.rag = RerankerRAG(config=config)

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
