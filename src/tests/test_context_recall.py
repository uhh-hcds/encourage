import unittest
import uuid
from unittest.mock import create_autospec, patch

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.context_recall import ContextRecall
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData


class TestContextRecall(unittest.TestCase):
    def setUp(self):
        # Sample responses as setup
        self.responses_mock = [
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
                        "reference_document": Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content=""
                        ),
                    }
                ),
                context=Context.from_documents(
                    [
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"),
                            content="Here is an example content",
                            score=1.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"),
                            content="Here is example content",
                            score=0.5,
                        ),
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
                        "reference_document": Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content=""
                        ),
                    }
                ),
                context=Context.from_documents(
                    [
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"),
                            content="Here is example content",
                            score=1.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "3"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "4"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "5"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]
        self.responses = ResponseWrapper(self.responses_mock)  # Wrap the provided responses
        self.runner = create_autospec(BatchInferenceRunner)  # Mock runner
        self.runner.run.return_value = ResponseWrapper(
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"sentences": [{"sentence": "Based on the story, Sharkie was not necessarily a friend, but rather a friend of Asta", "reason": "The context explicitly states that Sharkie is Asta's friend", "label": 1}, {"sentence": "but rather a friend of Asta's, as the story describes Sharkie as ", "reason": "The context explicitly states that Sharkie is Asta's friend", "label": 1}, {"sentence": "they work together to open the bottle and read the note", "reason": "The context supports the idea that Sharkie and Asta have a collaborative relationship", "label": 1}]}""",  # noqa: E501
                )
            ]
        )

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection):
        # Setup mocks for prompt collection and runner
        mock_prompt_collection = mock_prompt_collection.return_value
        mock_prompt_collection.create_prompts.return_value = ["mock_prompt"] * len(self.responses)
        # Initialize ContextRecall with mocked runner
        metric = ContextRecall(runner=self.runner)

        # Execute __call__ with sample responses
        result = metric(self.responses)

        # Assertions
        self.assertIn("total", result.misc)
        self.assertIn("attributed", result.misc)
        self.assertIn("sentences", result.misc)
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.raw, list)
        self.assertIsInstance(result.misc["total"], list)
        self.assertIsInstance(result.misc["attributed"], list)
        self.assertIsInstance(result.misc["sentences"], list)

    def test_calculate_metric(self):
        # Manually set up responses with varying sentence counts and labels
        metric = ContextRecall(runner=self.runner)
        result = metric(responses=self.responses)

        # Assertions for calculated metric
        self.assertIn("total", result.misc)
        self.assertIn("attributed", result.misc)
        self.assertIn("sentences", result.misc)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIsInstance(result.raw, list)
        self.assertEqual(len(result.raw), len(metric.responses))

    def test_empty_responses(self):
        # Instantiate with an empty ResponseWrapper
        metric = ContextRecall(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper([])
        empty_responses = ResponseWrapper([])

        # Call with empty responses
        result = metric(empty_responses)

        # Assertions for empty responses
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["total"], [])
        self.assertEqual(result.misc["attributed"], [])
        self.assertEqual(result.misc["sentences"], [])

    def test_no_attributed_sentences(self):
        # Setup mock responses with only non-attributed sentences

        metric = ContextRecall(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper(
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"sentences": [{"sentence": "Based on the story, Sharkie was not necessarily a friend, but rather a friend of Asta", "reason": "The context explicitly states that Sharkie is Asta's friend", "label": 0}, {"sentence": "but rather a friend of Asta's, as the story describes Sharkie as ", "reason": "The context explicitly states that Sharkie is Asta's friend", "label": 0}, {"sentence": "they work together to open the bottle and read the note", "reason": "The context supports the idea that Sharkie and Asta have a collaborative relationship", "label": 0}]}""",  # noqa: E501
                )
            ]
        )

        # Calculate metric with all non-attributed sentences
        result = metric(self.responses)

        # Check the score is zero as there are no attributed sentences
        self.assertEqual(result.score, 0.0)
        self.assertEqual(sum(result.misc["attributed"]), 0)
        self.assertTrue(all(x == 0 for x in result.misc["attributed"]))
