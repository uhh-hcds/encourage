import unittest
from unittest.mock import create_autospec

from encourage.llm.inference_runner import BatchInferenceRunner
from tests.fake_responses import create_responses


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample responses for testing
        self.responses = create_responses(5)
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
