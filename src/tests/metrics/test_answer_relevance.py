import unittest
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np

from encourage.llm import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics import AnswerRelevance, MetricOutput
from tests.fake_responses import create_responses


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.responses = ResponseWrapper(create_responses(2))

        # Create a mock runner
        self.runner = create_autospec(BatchInferenceRunner)

    @patch("sentence_transformers.SentenceTransformer", autospec=True)
    def test_answer_relevance_empty_committal_responses(self, mock_sentence_transformer):
        # Setup mock for SentenceTransformer
        mock_model = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)

        # Instantiate AnswerRelevance with mocks
        metric = AnswerRelevance(runner=self.runner, model_name="mock-model")
        metric.embeddings_model = mock_model

        # Mock the non_answer_critic output
        metric.non_answer_critic = MagicMock(
            return_value=MetricOutput(
                score=0.0,
                raw=[],
                misc={
                    "noncommittal": [1, 1, 1],
                    "rationales": None,
                    "generated_questions": None,
                },
            )
        )

        result = metric(self.responses)

        # Assertions for the empty case
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["noncommittal"], [1, 1, 1])
        self.assertIsNone(result.misc["rationales"])
        self.assertEqual(result.misc["generated_questions"], None)

    # @patch("encourage.metrics.answer_relevance.SentenceTransformer", autospec=True)
    # def test_answer_relevance_non_empty_committal_responses(self, mock_sentence_transformer):
    #     # Setup mock for SentenceTransformer
    #     mock_model = mock_sentence_transformer.return_value
    #     mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)

    #     # Instantiate AnswerRelevance with mocks
    #     metric = AnswerRelevance(runner=self.runner, model_name="mock-model")
    #     metric.embeddings_model = mock_model

    #     # Mock the non_answer_critic output to have at least one committal response
    #     metric.non_answer_critic = MagicMock()
    #     metric.non_answer_critic.return_value.raw = [
    #         ClassifiedAnswer(rationale="First example rationale.", non_answer=0),
    #         ClassifiedAnswer(rationale="Second example rationale.", non_answer=1),
    #         ClassifiedAnswer(rationale="Third example rationale.", non_answer=0),
    #     ]  # Mix of answers and non-answers

    #     # Create non-empty committal responses scenario
    #     responses = ResponseWrapper(self.responses)
    #     result = metric(responses)

    #     # Assertions for the non-empty case
    #     self.assertIn("noncommittal", result.misc)
    #     self.assertIn("rationales", result.misc)
    #     self.assertIn("generated_questions", result.misc)
    #     self.assertNotEqual(result.score, 0.0)
