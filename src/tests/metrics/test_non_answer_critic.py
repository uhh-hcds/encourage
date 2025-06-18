import unittest
import uuid
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.non_answer_critic import NonAnswerCritic
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData


class TestNonAnswerCritic(unittest.TestCase):
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
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 1}""",  # noqa: E501
                )
            ]
        )
        # Create the NonAnswerCritic instance with the mocked runner
        self.non_answer_critic = NonAnswerCritic(self.runner)

        # Mock the _runner.run and create_prompts method in the NonAnswerCritic
        self.non_answer_critic._runner.run = MagicMock(
            return_value=[]
        )  # Mock the run method to return empty list

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection):
        # Mock prompt collection and runner
        mock_prompt_collection = mock_prompt_collection.return_value
        mock_prompt_collection.create_prompts.return_value = ["mock_prompt"] * len(self.responses)

        # Instantiate metric with mocked runner
        metric = NonAnswerCritic(runner=self.runner)

        # Execute the metric's __call__ method
        result = metric(self.responses)

        print(result)

        # Assertions
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.raw, list)
        self.assertIsInstance(result.misc["raw_output"], list)

    def test_empty_responses(self):
        # Instantiate metric and use an empty ResponseWrapper
        metric = NonAnswerCritic(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper([])
        empty_responses = ResponseWrapper([])

        # Call with empty responses
        result = metric(empty_responses)

        # Assertions for empty responses
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["raw_output"], [])

    def test_calculate_metric_equal(self):
        # Setup mock for verdicts and non-answer flags

        # Instantiate metric
        metric = NonAnswerCritic(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper(
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 1}""",  # noqa: E501
                ),
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 0}""",  # noqa: E501
                ),
            ]
        )
        result = metric(self.responses)

        # Assertions for calculated metric
        self.assertGreaterEqual(result.score, 0.0)
        self.assertEqual(result.score, 0.5)
        self.assertLessEqual(result.score, 1.0)

    def test_calculate_metric_all(self):
        # Setup mock for verdicts and non-answer flags

        # Instantiate metric
        metric = NonAnswerCritic(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper(
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 1}""",  # noqa: E501
                ),
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 1}""",  # noqa: E501
                ),
            ]
        )
        result = metric(self.responses)

        # Assertions for calculated metric
        self.assertGreaterEqual(result.score, 0.0)
        self.assertEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_calculate_metric_no_answers(self):
        # Instantiate metric
        metric = NonAnswerCritic(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper(
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 0}""",  # noqa: E501
                ),
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response="""{"rationale": "The response indicates that there is no information about a fish with a name in the second part of the story, which is characteristic of a non-answer.", "non_answer": 0}""",  # noqa: E501
                ),
            ]
        )
        result = metric(self.responses)

        # Assertions for calculated metric
        self.assertGreaterEqual(result.score, 0.0)
        self.assertEqual(result.score, 1)
        self.assertLessEqual(result.score, 1.0)
