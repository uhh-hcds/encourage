import unittest
from unittest.mock import create_autospec, patch

from encourage.llm import BatchInferenceRunner, Response, ResponseWrapper
from encourage.metrics import ContextRecall
from tests.fake_responses import create_responses


class TestContextRecall(unittest.TestCase):
    def setUp(self) -> None:
        self.responses: ResponseWrapper = ResponseWrapper(create_responses(2))
        self.runner: BatchInferenceRunner = create_autospec(BatchInferenceRunner)
        self.runner.run.return_value = ResponseWrapper(  # pyright: ignore[reportAttributeAccessIssue]
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response=(
                        '{"sentences": ['
                        '{"sentence": "Based on the story, Sharkie was not necessarily a friend, '
                        'but rather a friend of Asta", '
                        '"reason": "The context explicitly states that Sharkie is Asta\'s friend", '
                        '"label": 1}, '
                        '{"sentence": "but rather a friend of Asta\'s, as the story describes Sharkie as ", '  # noqa: E501
                        '"reason": "The context explicitly states that Sharkie is Asta\'s friend", '
                        '"label": 1}, '
                        '{"sentence": "they work together to open the bottle and read the note", '
                        '"reason": "The context supports the idea that Sharkie and Asta have a collaborative relationship", '  # noqa: E501
                        '"label": 1}]}'
                    ),
                )
            ]
        )

    @patch("encourage.prompts.prompt_collection.PromptCollection", autospec=True)
    def test_call_with_responses(self, mock_prompt_collection) -> None:
        mock_prompt_collection = mock_prompt_collection.return_value
        mock_prompt_collection.create_prompts.return_value = ["mock_prompt"] * len(self.responses)

        metric = ContextRecall(runner=self.runner)
        result = metric(self.responses)

        self.assertIn("total", result.misc)
        self.assertIn("attributed", result.misc)
        self.assertIn("sentences", result.misc)
        self.assertIsInstance(result.score, float)
        self.assertIsInstance(result.raw, list)
        self.assertIsInstance(result.misc["total"], list)
        self.assertIsInstance(result.misc["attributed"], list)
        self.assertIsInstance(result.misc["sentences"], list)

    def test_calculate_metric(self) -> None:
        metric = ContextRecall(runner=self.runner)
        result = metric(responses=self.responses)

        self.assertIn("total", result.misc)
        self.assertIn("attributed", result.misc)
        self.assertIn("sentences", result.misc)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertIsInstance(result.raw, list)
        self.assertEqual(len(result.raw), len(metric.responses))

    def test_empty_responses(self) -> None:
        metric = ContextRecall(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper([])  # pyright: ignore[reportAttributeAccessIssue]
        empty_responses = ResponseWrapper([])

        result = metric(empty_responses)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.raw, [])
        self.assertEqual(result.misc["total"], [])
        self.assertEqual(result.misc["attributed"], [])
        self.assertEqual(result.misc["sentences"], [])

    def test_no_attributed_sentences(self) -> None:
        metric = ContextRecall(runner=self.runner)
        self.runner.run.return_value = ResponseWrapper(  # pyright: ignore[reportAttributeAccessIssue]
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="System prompt example.",
                    user_prompt="User prompt example.",
                    response=(
                        '{"sentences": ['
                        '{"sentence": "Based on the story, Sharkie was not necessarily a friend, '
                        'but rather a friend of Asta", '
                        '"reason": "The context explicitly states that Sharkie is Asta\'s friend", '
                        '"label": 0}, '
                        '{"sentence": "but rather a friend of Asta\'s, as the story describes Sharkie as ", '  # noqa: E501
                        '"reason": "The context explicitly states that Sharkie is Asta\'s friend", '
                        '"label": 0}, '
                        '{"sentence": "they work together to open the bottle and read the note", '
                        '"reason": "The context supports the idea that Sharkie and Asta have a collaborative relationship", '  # noqa: E501
                        '"label": 0}]}'
                    ),
                )
            ]
        )

        result = metric(self.responses)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(sum(result.misc["attributed"]), 0)
        self.assertTrue(all(x == 0 for x in result.misc["attributed"]))
