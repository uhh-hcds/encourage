import unittest
import uuid
from unittest.mock import create_autospec

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response import Response
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
        # Create a mock runner
        self.runner = create_autospec(BatchInferenceRunner)

    ## TODO: fix the test
    # def test_answer_faithfulness(self):
    #     mock_response = OutputNLI(
    #         verdicts=[
    #             Verdict(statement="", reason="", verdict=1),
    #             Verdict(statement="", reason="", verdict=0),
    #         ]
    #     )

    #     self.runner.run.return_value = ResponseWrapper([mock_response])

    #     # Instantiate AnswerFaithfulness
    #     metric = AnswerFaithfulness(runner=self.runner)

    #     # Wrap the responses as a ResponseWrapper
    #     wrapped_responses = ResponseWrapper(self.responses)

    #     # Invoke the metric
    #     result = metric(wrapped_responses)

    # # Check the result
    # self.assertIn("supported", result.misc)
    # self.assertIn("total", result.misc)

    # # Specific test checks, these may vary depending on your mocked output logic
    # self.assertAlmostEqual(result.score, 0.5, places=1)
