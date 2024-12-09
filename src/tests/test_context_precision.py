import unittest
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.context_precision import ContextPrecision
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData


class TestContextPrecision(unittest.TestCase):
    def setUp(self):
        # Sample responses as setup
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
                        "reference_answer": "This is the reference answer.",
                        "reference_document": ["doc1"],
                    }
                ),
                context=Context.from_documents(
                    [
                        {"content": "Here is an example content", "document": "doc1", "score": 1.0},
                        {"content": "Here is example content", "document": "doc2", "score": 0.5},
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
                        "reference_document": ["doc2"],  # Required field for MRR
                    }
                ),
                context=Context.from_documents(
                    [
                        {"content": "Here is an example content", "document": "doc2", "score": 1.0},
                        {"content": "Here is an example content", "document": "doc1", "score": 0.0},
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]
        self.responses = ResponseWrapper(self.responses)  # Wrap the provided responses
        self.runner = create_autospec(BatchInferenceRunner)  # Mock runner

    def test_average_precision(self):
        # Instantiate ContextPrecision without needing the runner's behavior
        metric = ContextPrecision(runner=self.runner)

        # Test cases for _average_precision
        test_cases = [
            ([1, 1, 0, 0], 1.0),  # All relevant first, expected precision is 1.0
            ([1, 0, 1, 0], 0.83),  # Mixed case
            ([0, 0, 0, 0], 0.0),  # No relevant items, expected precision is 0.0
            ([1, 1, 1, 1], 1.0),  # All relevant, precision should be 1.0
        ]

        for labels, expected in test_cases:
            result = metric._average_precision(labels)
            self.assertAlmostEqual(result, expected, places=2)

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection):
        # Mock prompt collection and runner
        mock_prompt_collection = mock_prompt_collection.return_value
        mock_prompt_collection.create_prompts.return_value = ["mock_prompt"] * len(self.responses)
        self.runner.run.return_value = [
            MagicMock(verdict=1),
            MagicMock(verdict=0),
            MagicMock(verdict=1),
        ]

        # Instantiate metric with mocked runner
        metric = ContextPrecision(runner=self.runner)

        # Execute the metric's __call__ method
        result = metric(self.responses)

        # Assertions
        self.assertIn("labeled_contexts", result.misc)
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.raw, list)
        self.assertIsInstance(result.misc["labeled_contexts"], list)

    def test_empty_responses(self):
        # Instantiate metric and use an empty ResponseWrapper
        metric = ContextPrecision(runner=self.runner)
        empty_responses = ResponseWrapper([])

        # Call with empty responses
        result = metric(empty_responses)

        # Assertions for empty responses
        self.assertEqual(result.score, 0.0)  # Should be 0 score
        self.assertEqual(result.raw, [])  # Should have an empty raw list
        self.assertEqual(result.misc["labeled_contexts"], [])  # No labeled contexts

    def test_calculate_metric(self):
        # Setup verdicts for multiple contexts and calculate metric
        self.runner.run.return_value = [
            MagicMock(verdict=1),
            MagicMock(verdict=1),
            MagicMock(verdict=0),
            MagicMock(verdict=1),
        ]

        # Instantiate metric
        metric = ContextPrecision(runner=self.runner)
        metric.responses = self.runner.run.return_value  # Set mock responses

        # Run _calculate_metric directly
        result = metric._calculate_metric(self.responses)

        # Assertions for calculated metric
        self.assertIn("labeled_contexts", result.misc)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
