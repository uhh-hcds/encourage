import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from encourage.llm import ResponseWrapper
from encourage.metrics import AnswerSimilarity
from tests.fake_responses import create_responses


class TestAnswerSimilarity(unittest.TestCase):
    def setUp(self) -> None:
        self.responses: ResponseWrapper = ResponseWrapper(create_responses(5))
        self.model_name: str = "mock-model"

    @patch("sentence_transformers.SentenceTransformer", autospec=True)
    def test_similarity_computation(self, mock_sentence_transformer: MagicMock) -> None:
        mock_model: MagicMock = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)
        mock_model.similarity.return_value = np.array([[0.8, 0.3], [0.3, 0.9]])

        metric: AnswerSimilarity = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model
        result = metric(self.responses)

        self.assertAlmostEqual(result.score, 0.85, places=2)
        self.assertEqual(result.raw, [0.8, 0.9])

    @patch("sentence_transformers.SentenceTransformer", autospec=True)
    def test_empty_responses(self, mock_sentence_transformer: MagicMock) -> None:
        empty_responses: ResponseWrapper = ResponseWrapper([])

        mock_model: MagicMock = mock_sentence_transformer.return_value
        metric: AnswerSimilarity = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model
        result = metric(empty_responses)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])

    @patch("sentence_transformers.SentenceTransformer", autospec=True)
    def test_single_response(self, mock_sentence_transformer: MagicMock) -> None:
        single_response: ResponseWrapper = ResponseWrapper([self.responses[0]])

        mock_model: MagicMock = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)
        mock_model.similarity.return_value = np.array([[0.7]])

        metric: AnswerSimilarity = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model
        result = metric(single_response)

        self.assertAlmostEqual(result.score, 0.7, places=2)
        self.assertEqual(result.raw, [0.7])

    @patch("sentence_transformers.SentenceTransformer", autospec=True)
    def test_embedding_computation(self, mock_sentence_transformer: MagicMock) -> None:
        mock_model: MagicMock = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.ones((len(texts), 768))

        metric: AnswerSimilarity = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model
        embedding = metric._get_embedding("This is a test response")

        self.assertEqual(embedding.shape, (1, 768))
        self.assertTrue((embedding == 1).all())
