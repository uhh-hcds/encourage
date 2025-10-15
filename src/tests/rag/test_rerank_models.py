import unittest
import uuid
from unittest.mock import MagicMock, patch

from encourage.prompts.context import Document
from encourage.rag.rerank.models import JinaV3, MSMarco


class TestRerankModels(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(id=uuid.uuid4(), content="Doc A"),
            Document(id=uuid.uuid4(), content="Doc B"),
            Document(id=uuid.uuid4(), content="Doc C"),
            Document(id=uuid.uuid4(), content="Doc D"),
        ]
        self.query = "test query"

    @patch("encourage.rag.rerank.models.CrossEncoder")
    def test_ms_marco_reranks_and_limits(self, mock_cross_encoder):
        # Setup CrossEncoder mock instance and predictable scores
        instance = MagicMock()
        instance.predict.return_value = [0.1, 0.9, 0.5, 0.2]
        mock_cross_encoder.return_value = instance

        reranker = MSMarco(rerank_ratio=2.0, device="cpu")
        # Request top_k=2
        top = reranker.rerank_documents(self.query, self.documents, top_k=2)

        # Ensure predict was called with pairs
        instance.predict.assert_called_once()
        # Ensure two documents returned
        self.assertEqual(len(top), 2)
        # Highest score corresponds to document with index 1
        self.assertIn(self.documents[1], top)

    @patch("encourage.rag.rerank.models.AutoModel")
    def test_jina_v3_reranks_and_limits(self, mock_auto_model):
        # Create a mock model that has rerank method
        model_instance = MagicMock()
        # Suppose rerank returns scores aligned with documents
        model_instance.rerank.return_value = [0.7, 0.2]
        mock_auto_model.from_pretrained.return_value = model_instance

        reranker = JinaV3(rerank_ratio=2.0, device="cpu")
        # Use only first two documents to match mock returns
        docs = self.documents[:2]
        top = reranker.rerank_documents(self.query, docs, top_k=2)

        # Ensure rerank was called with expected args
        model_instance.rerank.assert_called_once()
        self.assertEqual(len(top), 2)


if __name__ == "__main__":
    unittest.main()
