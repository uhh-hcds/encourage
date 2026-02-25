import unittest

from pydantic import BaseModel, Field

from encourage.llm import ResponseWrapper
from encourage.metrics import map_pydantic_field_to_response
from tests.fake_responses import create_responses


class StructuredAnswer(BaseModel):
    final_answer: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class TestMetricHelpers(unittest.TestCase):
    def test_map_pydantic_field_to_response_with_field_name(self) -> None:
        responses = ResponseWrapper(
            create_responses(
                n=2,
                response_content_list=[
                    {
                        "final_answer": "42",
                        "reasoning": "calculated",
                        "confidence": 0.9,
                    },
                    {
                        "final_answer": "13",
                        "reasoning": "estimated",
                        "confidence": 0.7,
                    },
                ],
            )
        )

        updated = map_pydantic_field_to_response(responses, StructuredAnswer, "final_answer")

        self.assertIsNot(updated, responses)
        self.assertEqual(updated[0].response, "42")
        self.assertEqual(updated[1].response, "13")
        self.assertEqual(updated[0].meta_data["reasoning"], "calculated")
        self.assertEqual(updated[1].meta_data["reasoning"], "estimated")
        self.assertEqual(updated[0].meta_data["confidence"], 0.9)
        self.assertEqual(updated[1].meta_data["confidence"], 0.7)
        self.assertEqual(responses[0].response["final_answer"], "42")
        self.assertNotIn("reasoning", responses[0].meta_data.tags)

    def test_map_pydantic_field_to_response_with_field_object(self) -> None:
        responses = ResponseWrapper(
            create_responses(
                n=1,
                response_content_list=[
                    {
                        "final_answer": "yes",
                        "reasoning": "from context",
                        "confidence": 1.0,
                    }
                ],
            )
        )

        updated = map_pydantic_field_to_response(
            responses,
            StructuredAnswer,
            StructuredAnswer.model_fields["final_answer"],
        )

        self.assertEqual(updated[0].response, "yes")
        self.assertEqual(updated[0].meta_data["reasoning"], "from context")
        self.assertEqual(updated[0].meta_data["confidence"], 1.0)

    def test_map_pydantic_field_to_response_with_json_string(self) -> None:
        responses = ResponseWrapper(
            create_responses(
                n=1,
                response_content_list=[
                    '{"final_answer":"42","reasoning":"calculated","confidence":0.8}'
                ],
            )
        )

        updated = map_pydantic_field_to_response(responses, StructuredAnswer, "final_answer")

        self.assertEqual(updated[0].response, "42")
        self.assertEqual(updated[0].meta_data["reasoning"], "calculated")
        self.assertEqual(updated[0].meta_data["confidence"], 0.8)
        self.assertEqual(
            responses[0].response,
            '{"final_answer":"42","reasoning":"calculated","confidence":0.8}',
        )

    def test_sets_empty_response_on_non_dict_response(self) -> None:
        responses = ResponseWrapper(create_responses(n=1, response_content_list=["plain text"]))

        updated = map_pydantic_field_to_response(responses, StructuredAnswer, "final_answer")

        self.assertEqual(updated[0].response, "")

    def test_raises_on_invalid_selected_field(self) -> None:
        responses = ResponseWrapper(
            create_responses(
                n=1,
                response_content_list=[
                    {
                        "final_answer": "42",
                        "reasoning": "calculated",
                        "confidence": 0.9,
                    }
                ],
            )
        )

        with self.assertRaises(ValueError):
            map_pydantic_field_to_response(responses, StructuredAnswer, "unknown_field")
