import unittest
from unittest.mock import MagicMock

from vllm import RequestOutput

from encourage.llm.inference_runner import BatchInferenceRunner, ChatInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.conversation import Conversation
from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_collection import PromptCollection


class TestChatInferenceRunner(unittest.TestCase):
    def setUp(self):
        """Setup common resources used in tests."""
        self.llm = MagicMock()  # Mock LLM
        self.sampling_parameters = MagicMock()  # Mock SamplingParams
        # Create a mock prompt and conversation
        self.prompt = MagicMock()  # Mock Prompt with sys_prompt and user_prompt
        self.conversation = Conversation(self.prompt)
        self.runner = ChatInferenceRunner(
            self.llm,
            self.sampling_parameters,
        )

    def test_run_success(self):
        """Test successful run of ChatInferenceRunner."""
        # Mock LLM's chat method response
        mock_request_output = MagicMock(spec=RequestOutput)
        mock_request_output.outputs = [MagicMock(text="Hello, how can I help you?")]
        self.llm.chat.return_value = [mock_request_output]

        # Run the inference
        response = self.runner.run(self.conversation)

        # Verify the chat method was called with correct parameters
        self.llm.chat.assert_called_once_with(
            self.conversation.dialog, self.sampling_parameters, use_tqdm=True
        )

        # Check the response is as expected
        self.assertEqual(response.outputs[0].text, "Hello, how can I help you?")

    def test_run_empty_response(self):
        """Test run with an empty response."""
        # Mock LLM's chat method to return an empty text response
        mock_request_output = MagicMock(spec=RequestOutput)
        mock_request_output.outputs = [MagicMock(text="")]
        self.llm.chat.return_value = [mock_request_output]

        # Run the inference
        response = self.runner.run(self.conversation)

        # Verify the chat method was called
        self.llm.chat.assert_called_once_with(
            self.conversation.dialog, self.sampling_parameters, use_tqdm=True
        )

        # Check the response contains an empty message
        self.assertEqual(response.outputs[0].text, "")


class TestBatchInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock()  # Mock LLM
        self.sampling_parameters = MagicMock()  # Mock SamplingParams
        self.prompt1 = Prompt(sys_prompt="Prompt 1", user_prompt="User 1")
        self.prompt2 = Prompt(sys_prompt="Prompt 2", user_prompt="User 2")
        self.prompt_collection = PromptCollection(prompts=[self.prompt1, self.prompt2])
        self.batch_runner = BatchInferenceRunner(self.llm, self.sampling_parameters)

    def test_run_success(self):
        """Test successful run of BatchInferenceRunner."""
        # Mock RequestOutput and its attributes
        mock_request_output1 = MagicMock()
        mock_request_output1.request_id = "req_1"
        mock_request_output1.outputs = [MagicMock(text="Response 1")]
        mock_request_output1.metrics.arrival_time = 1000.0
        mock_request_output1.metrics.finished_time = 1010.0

        mock_request_output2 = MagicMock()
        mock_request_output2.request_id = "req_2"
        mock_request_output2.outputs = [MagicMock(text="Response 2")]
        mock_request_output2.metrics.arrival_time = 1000.0
        mock_request_output2.metrics.finished_time = 1010.0

        self.llm.generate.return_value = [mock_request_output1, mock_request_output2]

        response_wrapper = self.batch_runner.run(self.prompt_collection)

        # Assert that a ResponseWrapper object is returned
        self.assertIsInstance(response_wrapper, ResponseWrapper)
        self.assertEqual(len(response_wrapper.response_data), 2)

        # Assert that the first response matches
        response1 = response_wrapper.get_response_by_prompt_id(str(self.prompt1.id))
        self.assertEqual(response1.response, "Response 1")

        # Assert that the second response matches
        response2 = response_wrapper.get_response_by_prompt_id(str(self.prompt2.id))
        self.assertEqual(response2.response, "Response 2")

        self.llm.generate.assert_called_once()

    def test_run_empty_responses(self):
        """Test run with empty responses."""
        # Mock RequestOutput with empty responses
        mock_request_output1 = MagicMock()
        mock_request_output1.request_id = "req_1"
        mock_request_output1.outputs = [MagicMock(text="")]
        mock_request_output1.metrics.arrival_time = 1000.0
        mock_request_output1.metrics.finished_time = 1010.0

        mock_request_output2 = MagicMock()
        mock_request_output2.request_id = "req_2"
        mock_request_output2.outputs = [MagicMock(text="")]
        mock_request_output2.metrics.arrival_time = 1000.0
        mock_request_output2.metrics.finished_time = 1010.0

        self.llm.generate.return_value = [mock_request_output1, mock_request_output2]

        response_wrapper = self.batch_runner.run(self.prompt_collection)

        # Assert that a ResponseWrapper object is returned
        self.assertIsInstance(response_wrapper, ResponseWrapper)
        self.assertEqual(len(response_wrapper.response_data), 2)

        # Assert that the first response is empty
        response1 = response_wrapper.get_response_by_prompt_id(str(self.prompt1.id))
        self.assertEqual(response1.response, "")

        # Assert that the second response is empty
        response2 = response_wrapper.get_response_by_prompt_id(str(self.prompt2.id))
        self.assertEqual(response2.response, "")

        self.llm.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
