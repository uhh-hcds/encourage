import unittest
from unittest.mock import MagicMock

from encourage.llm.response_wrapper import (
    ResponseWrapper,  # Replace with the actual import path
)
from encourage.prompts.prompt_collection import PromptCollection


class TestResponseWrapper(unittest.TestCase):
    def setUp(self):
        # Sample data setup
        self.mock_prompt_collection = MagicMock(spec=PromptCollection)

        # Create two mock request outputs to match the number of prompts
        self.mock_request_output_1 = MagicMock()
        self.mock_request_output_2 = MagicMock()

        # Mock responses for the first request output
        self.mock_request_output_1.request_id = "1"
        self.mock_request_output_1.outputs = [
            MagicMock(text="Response text 1"),
        ]
        self.mock_request_output_1.metrics = MagicMock()
        self.mock_request_output_1.metrics.arrival_time = 0
        self.mock_request_output_1.metrics.finished_time = 1

        # Mock responses for the second request output
        self.mock_request_output_2.request_id = "2"
        self.mock_request_output_2.outputs = [
            MagicMock(text="Response text 2"),
        ]
        self.mock_request_output_2.metrics = MagicMock()
        self.mock_request_output_2.metrics.arrival_time = 0
        self.mock_request_output_2.metrics.finished_time = 1

        # Mock prompts in the collection
        self.mock_prompt_collection.prompts = [
            MagicMock(
                id="prompt1",
                sys_prompt="System Prompt 1",
                user_prompt="User Prompt 1",
                meta_data={"key": "value"},
                context={"context_key": "context_value"},
            ),
            MagicMock(
                id="prompt2",
                sys_prompt="System Prompt 2",
                user_prompt="User Prompt 2",
                meta_data={"key": "value"},
                context={"context_key": "context_value"},
            ),
        ]

        # Create a ResponseWrapper instance, passing a list of two mock request outputs
        self.response_wrapper = ResponseWrapper.from_prompt_collection(
            [self.mock_request_output_1, self.mock_request_output_2],  # Two request outputs
            self.mock_prompt_collection,
        )

    def test_initialization(self):
        """Test initialization of ResponseWrapper."""
        self.assertIsInstance(self.response_wrapper, ResponseWrapper)
        self.assertEqual(len(self.response_wrapper.response_data), 2)

    def test_from_prompt_collection(self):
        """Test creation of ResponseWrapper from RequestOutput and PromptCollection."""
        response = self.response_wrapper.response_data[0]
        self.assertEqual(response.request_id, "1")
        self.assertEqual(response.prompt_id, str(self.mock_prompt_collection.prompts[0].id))
        self.assertEqual(response.response, "Response text 1")

    def test_get_response_by_prompt_id(self):
        """Test get_response_by_prompt_id method."""
        response = self.response_wrapper.get_response_by_prompt_id(
            str(self.mock_prompt_collection.prompts[0].id)
        )
        self.assertIsNotNone(response)
        self.assertEqual(response.request_id, "1")

    def test_get_response_by_request_id(self):
        """Test get_response_by_request_id method."""
        response = self.response_wrapper.get_response_by_request_id("1")
        self.assertIsNotNone(response)
        self.assertEqual(response.request_id, "1")

    def test_no_response_found_by_prompt_id(self):
        """Test behavior when no response is found by prompt ID."""
        response = self.response_wrapper.print_response_by_prompt_id("invalid_id")
        self.assertIsNone(response)  # Adjust as needed based on method behavior

    def test_no_response_found_by_request_id(self):
        """Test behavior when no response is found by request ID."""
        response = self.response_wrapper.print_response_by_request_id("invalid_id")
        self.assertIsNone(response)  # Adjust as needed based on method behavior


if __name__ == "__main__":
    unittest.main()
