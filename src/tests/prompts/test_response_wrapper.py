import unittest
from typing import Optional

from encourage.llm import ResponseWrapper
from encourage.llm.response import Response
from encourage.prompts.prompt_collection import PromptCollection
from tests.fake_responses import (
    create_mock_chatcompletions,
    create_mock_request_outputs,
    create_prompt_collection,
)


class TestResponseWrapper(unittest.TestCase):
    def setUp(self):
        """
        ResponseWrapper generated from fake ChatCompletions and PromptCollection.
        """
        self.prompt_collection: PromptCollection = create_prompt_collection(5)
        content = [
            "What would a pirate say?",
            "What would a pirate do?",
            "What would a pirate eat?",
            "What would a pirate drink?",
            "What would a pirate wear?",
        ]
        self.chat_completions = create_mock_chatcompletions(5, content)
        self.request_outputs = create_mock_request_outputs(5, content)

        self.response_wrapper: ResponseWrapper = ResponseWrapper.from_prompt_collection(
            self.chat_completions,
            self.prompt_collection,
        )

    def test_initialization(self):
        """Test initialization of ResponseWrapper."""
        self.assertIsInstance(self.response_wrapper, ResponseWrapper)
        self.assertEqual(len(self.response_wrapper.response_data), 5)

    def test_from_prompt_collection(self):
        """Test creation of ResponseWrapper from RequestOutput and PromptCollection."""
        response = self.response_wrapper.response_data[1]
        self.assertEqual(response.prompt_id, str(self.prompt_collection.prompts[1].id))
        self.assertEqual(response.response, "What would a pirate do?")

    def test_from_request_outputs(self):
        """Test creation of ResponseWrapper from RequestOutput."""
        response_wrapper: ResponseWrapper = ResponseWrapper.from_request_output(
            self.request_outputs,
            self.prompt_collection,
        )
        self.assertIsInstance(response_wrapper, ResponseWrapper)
        self.assertEqual(len(response_wrapper.response_data), len(self.request_outputs))

    def test_get_response_by_prompt_id(self):
        """Test get_response_by_prompt_id method."""
        response: Optional[Response] = self.response_wrapper.get_response_by_prompt_id(
            str(self.prompt_collection.prompts[0].id)
        )
        self.assertIsNotNone(response)
        if response is not None:
            self.assertEqual(response.prompt_id, self.prompt_collection.prompts[0].id)

    def test_no_response_found_by_prompt_id(self):
        """Test behavior when no response is found by prompt ID."""
        response: Optional[Response] = self.response_wrapper.print_response_by_prompt_id(
            "invalid_id"
        )
        self.assertIsNone(response)

    def test_no_response_found_by_request_id(self):
        """Test behavior when no response is found by request ID."""
        response: Optional[Response] = self.response_wrapper.print_response_by_request_id(
            "invalid_id"
        )
        self.assertIsNone(response)


if __name__ == "__main__":
    unittest.main()
