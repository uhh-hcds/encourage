import unittest
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.context_recall import ContextRecall


class TestContextRecall(unittest.TestCase):
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
                meta_data={
                    "reference_answer": "This is the reference answer.",
                    "reference_document": ["doc1"],  # Required field for MRR
                },
                context={
                    "contexts": [  # Required field for MRR
                        {"content": "Here is an example content", "document": "doc1", "score": 1.0},
                        {"content": "Here is example content", "document": "doc2", "score": 0.5},
                    ]
                },
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
                meta_data={
                    "reference_answer": "Another reference answer.",
                    "reference_document": ["doc2"],  # Required field for MRR
                },
                context={
                    "contexts": [  # Required field for MRR
                        {"content": "Here is an example content", "document": "doc2", "score": 1.0},
                        {"content": "Here is an example content", "document": "doc1", "score": 0.0},
                    ]
                },
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]
        self.responses = ResponseWrapper(self.responses)  # Wrap the provided responses
        self.runner = create_autospec(BatchInferenceRunner)  # Mock runner

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection):
        # Setup mocks for prompt collection and runner
        mock_prompt_collection = mock_prompt_collection.return_value
        mock_prompt_collection.create_prompts.return_value = ["mock_prompt"] * len(self.responses)
        self.runner.run.return_value = [
            MagicMock(sentences=[MagicMock(label=1), MagicMock(label=0)]),
            MagicMock(sentences=[MagicMock(label=1), MagicMock(label=1)]),
        ]

        # Initialize ContextRecall with mocked runner
        metric = ContextRecall(runner=self.runner)

        # Execute __call__ with sample responses
        result = metric(self.responses)

        # Assertions
        self.assertIn("score", result)
        self.assertIn("raw", result)
        self.assertIn("total", result)
        self.assertIn("attributed", result)
        self.assertIn("sentences", result)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["raw"], list)
        self.assertIsInstance(result["total"], list)
        self.assertIsInstance(result["attributed"], list)
        self.assertIsInstance(result["sentences"], list)

    def test_calculate_metric(self):
        # Manually set up responses with varying sentence counts and labels
        metric = ContextRecall(runner=self.runner)
        metric.responses = [
            MagicMock(response=MagicMock(sentences=[MagicMock(label=1), MagicMock(label=0)])),
            MagicMock(response=MagicMock(sentences=[MagicMock(label=1), MagicMock(label=1)])),
        ]

        # Execute _calculate_metric directly
        result = metric._calculate_metric()

        # Assertions for calculated metric
        self.assertIn("score", result)
        self.assertIn("raw", result)
        self.assertIn("total", result)
        self.assertIn("attributed", result)
        self.assertIn("sentences", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertIsInstance(result["raw"], list)
        self.assertEqual(len(result["raw"]), len(metric.responses))

    def test_empty_responses(self):
        # Instantiate with an empty ResponseWrapper
        metric = ContextRecall(runner=self.runner)
        empty_responses = ResponseWrapper([])

        # Call with empty responses
        result = metric(empty_responses)

        # Assertions for empty responses
        self.assertEqual(result["score"], 0.0)  # Should be 0 score if no sentences
        self.assertEqual(result["raw"], [])  # Should have an empty raw list
        self.assertEqual(result["total"], [])  # Should have an empty total list
        self.assertEqual(result["attributed"], [])  # Should have an empty attributed list
        self.assertEqual(result["sentences"], [])  # No sentences

    def test_no_attributed_sentences(self):
        # Setup mock responses with only non-attributed sentences
        self.runner.run.return_value = [
            MagicMock(sentences=[MagicMock(label=0), MagicMock(label=0)]),
            MagicMock(sentences=[MagicMock(label=0), MagicMock(label=0)]),
        ]

        metric = ContextRecall(runner=self.runner)
        metric.responses = self.runner.run.return_value

        # Calculate metric with all non-attributed sentences
        result = metric._calculate_metric()

        # Check the score is zero as there are no attributed sentences
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(sum(result["attributed"]), 0)
        self.assertTrue(all(x == 0 for x in result["attributed"]))
