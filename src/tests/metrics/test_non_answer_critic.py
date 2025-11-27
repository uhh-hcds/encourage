import unittest
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.metrics import NonAnswerCritic
from tests.fake_responses import create_responses


class TestNonAnswerCritic(unittest.TestCase):
    def setUp(self):
        self.responses = ResponseWrapper(create_responses(n=2))
        self.runner = create_autospec(BatchInferenceRunner)
        self.non_answer_critic = NonAnswerCritic(self.runner)
        self.non_answer_critic._runner.run = MagicMock(return_value=[])  # type: ignore[assignment]

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection):
        mock_prompt_collection.return_value.create_prompts.return_value = ["mock_prompt"] * len(
            self.responses
        )

        metric = NonAnswerCritic(self.runner)
        result = metric(self.responses)

        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.raw, list)
        self.assertIsInstance(result.misc["raw_output"], list)

    def test_empty_responses(self):
        self.runner.run.return_value = ResponseWrapper([])
        result = NonAnswerCritic(self.runner)(ResponseWrapper([]))

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["raw_output"], [])

    def test_calculate_metric_equal(self):
        responses = create_responses(
            n=2,
            response_content_list=[
                '{"rationale": "...", "non_answer": 1}',
                '{"rationale": "...", "non_answer": 0}',
            ],
        )
        self.runner.run.return_value = ResponseWrapper(responses)
        result = NonAnswerCritic(self.runner)(self.responses)

        self.assertEqual(result.score, 0.5)

    def test_calculate_metric_all_non_answers(self):
        responses = create_responses(
            n=2,
            response_content_list=[
                '{"rationale": "...", "non_answer": 1}',
                '{"rationale": "...", "non_answer": 1}',
            ],
        )
        self.runner.run.return_value = ResponseWrapper(responses)
        result = NonAnswerCritic(self.runner)(self.responses)

        self.assertEqual(result.score, 0.0)

    def test_calculate_metric_no_non_answers(self):
        responses = create_responses(
            n=2,
            response_content_list=[
                '{"rationale": "...", "non_answer": 0}',
                '{"rationale": "...", "non_answer": 0}',
            ],
        )
        self.runner.run.return_value = ResponseWrapper(responses)
        result = NonAnswerCritic(self.runner)(self.responses)

        self.assertEqual(result.score, 1.0)
