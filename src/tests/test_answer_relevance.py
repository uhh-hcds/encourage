import unittest
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.answer_relevance import AnswerRelevance
from encourage.metrics.metric import MetricOutput
from encourage.metrics.non_answer_critic import ClassifiedAnswer  # ## Do we need this?
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample responses for testing
        self.responses = [
            Response(
                request_id="1",
                prompt_id="p1",
                sys_prompt="System prompt example.",
                user_prompt="User prompt example.",
                response="This is a generated answer.",
                conversation_id=1,
                meta_data=MetaData(
                    tags={
                        "reference_answer": "This is a generated answer.",
                        "reference_document": Document(id="1", content=""),
                    }
                ),
                context=Context.from_documents(
                    [
                        {"id": 1, "content": "Here is an example content", "score": 1.0},
                        {"id": 0, "content": "Here is example content", "score": 0.5},
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
            Response(
                request_id="2",
                prompt_id="p2",
                sys_prompt="Another system prompt.",
                user_prompt="Another user prompt.",
                response="Another generated answer.",
                conversation_id=2,
                meta_data=MetaData(
                    tags={
                        "reference_answer": "Another reference answer.",
                        "reference_document": Document(id="0", content=""),
                    }
                ),
                context=Context.from_documents(
                    [
                        {"id": 1, "content": "Here is example content", "score": 1.0},
                        {"id": 2, "content": "Here is an example content with extra", "score": 0.0},
                        {"id": 3, "content": "Here is an example content with extra", "score": 0.0},
                        {"id": 4, "content": "Here is an example content with extra", "score": 0.0},
                        {"id": 0, "content": "Here is an example content with extra", "score": 0.0},
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]
        # Create a mock runner
        self.runner = create_autospec(BatchInferenceRunner)

    @patch("encourage.metrics.answer_relevance.SentenceTransformer", autospec=True)
    def test_answer_relevance_empty_committal_responses(self, mock_sentence_transformer):
        # Setup mock for SentenceTransformer
        mock_model = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)

        # Instantiate AnswerRelevance with mocks
        metric = AnswerRelevance(runner=self.runner, model_name="mock-model")
        metric.embeddings_model = mock_model

        # Mock the non_answer_critic output
        metric.non_answer_critic = MagicMock()
        metric.non_answer_critic.return_value = MetricOutput(
            score=0.0,  # since all responses are non-answers
            raw=[],  # assuming an empty list for raw non-answers
            misc={
                "noncommittal": [1, 1, 1],  # mock noncommittal responses
                "rationales": None,
                "generated_questions": None,
            },
        )

        # Create empty committal responses scenario
        responses = ResponseWrapper(self.responses)
        result = metric(responses)

        # Assertions for the empty case
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["noncommittal"], [1, 1, 1])
        self.assertIsNone(result.misc["rationales"])
        self.assertEqual(result.misc["generated_questions"], None)

    @patch("encourage.metrics.answer_relevance.SentenceTransformer", autospec=True)
    def test_answer_relevance_non_empty_committal_responses(self, mock_sentence_transformer):
        # Setup mock for SentenceTransformer
        mock_model = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)

        # Instantiate AnswerRelevance with mocks
        metric = AnswerRelevance(runner=self.runner, model_name="mock-model")
        metric.embeddings_model = mock_model

        # Mock the non_answer_critic output to have at least one committal response
        metric.non_answer_critic = MagicMock()
        metric.non_answer_critic.return_value.raw = [
            ClassifiedAnswer(rationale="First example rationale.", non_answer=0),
            ClassifiedAnswer(rationale="Second example rationale.", non_answer=1),
            ClassifiedAnswer(rationale="Third example rationale.", non_answer=0),
        ]  # Mix of answers and non-answers

        # Create non-empty committal responses scenario
        responses = ResponseWrapper(self.responses)
        result = metric(responses)

        # Assertions for the non-empty case
        self.assertIn("noncommittal", result.misc)
        self.assertIn("rationales", result.misc)
        self.assertIn("generated_questions", result.misc)
        self.assertNotEqual(result.score, 0.0)
