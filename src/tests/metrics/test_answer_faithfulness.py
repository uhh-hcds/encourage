import unittest
from unittest.mock import create_autospec, patch

from encourage.llm import ResponseWrapper
from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.metrics.answer_faithfulness import (
    AnswerFaithfulness,
    ExampleNLI,
    ExampleSplit,
    OutputNLI,
    OutputSplit,
    Verdict,
)
from encourage.prompts import PromptCollection
from tests.fake_responses import create_responses


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample responses for testing
        self.responses = create_responses(5)
        # Create a mock runner
        self.runner = create_autospec(BatchInferenceRunner)

    @patch("nltk.sent_tokenize", return_value=["A single sentence."])
    def test_answer_faithfulness_allows_custom_sys_prompts_and_examples(self, _mock_sent_tokenize):
        split_example = ExampleSplit(
            question="Q?",
            answer="A.",
            sentence="A.",
            output=OutputSplit(simpler_statements=["A"]),
        )
        nli_example = ExampleNLI(
            context="Context",
            statements=["A"],
            output=OutputNLI(verdicts=[Verdict(statement="A", reason="Supported", verdict=1)]),
        )

        metric = AnswerFaithfulness(
            runner=self.runner,
            split_examples=[split_example],
            nli_examples=[nli_example],
            split_sys_prompt="split-system",
            nli_sys_prompt="nli-system",
        )

        input_responses = ResponseWrapper(
            create_responses(
                n=1,
                response_content_list=["A single sentence."],
            )
        )

        split_output = ResponseWrapper(
            create_responses(
                n=1,
                response_content_list=['{"simpler_statements": ["Claim 1"]}'],
            )
        )
        nli_output = ResponseWrapper(
            create_responses(
                n=1,
                response_content_list=[
                    '{"verdicts": [{"statement": "Claim 1", "reason": "Supported", "verdict": 1}]}'
                ],
            )
        )
        self.runner.run.side_effect = [split_output, nli_output]

        with patch.object(
            PromptCollection,
            "create_prompts",
            side_effect=["split", "nli"],
        ) as mocked:
            result = metric(input_responses)

        split_call = mocked.call_args_list[0].kwargs
        nli_call = mocked.call_args_list[1].kwargs

        self.assertEqual(split_call["sys_prompts"], "split-system")
        self.assertEqual(nli_call["sys_prompts"], "nli-system")
        self.assertEqual(split_call["contexts"][0].prompt_vars["examples"], [split_example])
        self.assertEqual(nli_call["contexts"][0].prompt_vars["examples"], [nli_example])
        self.assertEqual(result.score, 1.0)
