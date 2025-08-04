import unittest
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm import BatchInferenceRunner, Response, ResponseWrapper
from encourage.metrics import ContextPrecision
from tests.fake_responses import create_responses


class TestContextPrecision(unittest.TestCase):
    def setUp(self) -> None:
        self.responses_mock: list[Response] = create_responses(5)
        self.responses: ResponseWrapper = ResponseWrapper(self.responses_mock)
        self.runner: BatchInferenceRunner = create_autospec(BatchInferenceRunner)

    def test_average_precision(self) -> None:
        metric: ContextPrecision = ContextPrecision(runner=self.runner)
        test_cases: list[tuple[list[int], float]] = [
            ([1, 1, 0, 0], 1.0),
            ([1, 0, 1, 0], 0.83),
            ([0, 0, 0, 0], 0.0),
            ([1, 1, 1, 1], 1.0),
        ]
        for labels, expected in test_cases:
            result = metric._average_precision(labels)
            self.assertAlmostEqual(result, expected, places=2)

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection: MagicMock) -> None:
        mock_prompt_collection = mock_prompt_collection.return_value
        mock_prompt_collection.create_prompts.return_value = ["mock_prompt"] * len(self.responses)

        self.runner.run.return_value = [  # pyright: ignore[reportAttributeAccessIssue]
            MagicMock(verdict=1),
            MagicMock(verdict=0),
            MagicMock(verdict=1),
        ]

        metric: ContextPrecision = ContextPrecision(runner=self.runner)
        result = metric(self.responses)

        self.assertIn("labeled_contexts", result.misc)
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.raw, list)
        self.assertIsInstance(result.misc["labeled_contexts"], list)

    def test_empty_responses(self) -> None:
        metric: ContextPrecision = ContextPrecision(runner=self.runner)
        empty_responses: ResponseWrapper = ResponseWrapper([])
        result = metric(empty_responses)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["labeled_contexts"], [])

    def test_calculate_metric(self):
        self.runner.run.return_value = [  # pyright: ignore[reportAttributeAccessIssue]
            MagicMock(verdict=1),
            MagicMock(verdict=1),
            MagicMock(verdict=0),
            MagicMock(verdict=1),
        ]

        metric = ContextPrecision(runner=self.runner)
        metric.responses = self.runner.run.return_value  # type: ignore
        metric.context_mapping = [0, 0, 1, 1]
        metric.original_responses_count = 2
        result = metric._calculate_metric()

        self.assertIn("labeled_contexts", result.misc)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
