"""Module that defines the ResponseWrapper class."""

import logging
from typing import Iterator

from vllm import RequestOutput

from encourage.llm.response import Response
from encourage.prompts.conversation import Conversation, Role
from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_collection import PromptCollection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseWrapper:
    """Class that aggregates RequestOutput with corresponding PromptCollection details."""

    def __repr__(self) -> str:
        return f"ResponseWrapper({self.response_data})"

    def __init__(self, responses: list[Response]):
        self.response_data = responses

    def __iter__(self) -> Iterator[Response]:
        """Allows iteration over the responses."""
        return iter(self.response_data)

    def __getitem__(self, key: int) -> Response:
        """Returns the response at the given index."""
        return self.response_data[key]

    @classmethod
    def from_prompt_collection(
        cls,
        request_outputs: list[RequestOutput],
        collection: PromptCollection,
    ) -> "ResponseWrapper":
        """Create ResponseWrapper from RequestOutput and PromptCollection or Conversation."""
        if len(request_outputs) != len(collection.prompts):
            raise ValueError("The number of request outputs does not match the number of prompts.")

        # Create responses for each request_output and its corresponding prompt
        responses = [
            cls.handle_prompt_response(request_output, prompt)
            for request_output, prompt in zip(request_outputs, collection.prompts)
        ]
        return cls(responses)

    @classmethod
    def from_conversation(
        cls,
        request_outputs: list[RequestOutput],
        conversation: Conversation,
        meta_datas: list[dict] = [],
    ) -> "ResponseWrapper":
        """Create ResponseWrapper from RequestOutput and Conversation."""
        # Check for mismatched lengths
        if len(request_outputs) != len(conversation.get_messages_by_role(Role.USER)):
            raise ValueError("The number of request outputs does not match the number of messages.")

        # Safely access the system prompt, handle missing system messages
        sys_messages = conversation.get_messages_by_role(Role.SYSTEM)
        if not sys_messages:
            raise ValueError("No system messages found in the conversation.")
        sys_prompt = sys_messages[0]["content"]

        # Extract user messages
        user_messages = [msg["content"] for msg in conversation.get_messages_by_role(Role.USER)]

        responses = []
        for conversation_id, (request_output, user_message, meta_data) in enumerate(
            zip(request_outputs, user_messages, meta_datas or [{}] * len(request_outputs))
        ):
            response = cls.handle_conversation_response(
                sys_prompt, conversation_id, request_output, user_message, meta_data
            )
            responses.append(response)

        return cls(responses)

    @staticmethod
    def handle_conversation_response(
        sys_prompt: str,
        conversation_id: int,
        request_output: RequestOutput,
        message: str,
        meta_data: dict,
    ) -> Response:
        """Create a Response object from a RequestOutput and Conversation."""
        return Response(
            request_id=request_output.request_id,
            prompt_id="",
            conversation_id=conversation_id,
            sys_prompt=sys_prompt,
            user_prompt=message,
            response=request_output.outputs[0].text if request_output.outputs else "No response",
            meta_data=[meta_data],
            context=[],
            arrival_time=(
                request_output.metrics.arrival_time
                if request_output.metrics and request_output.metrics.arrival_time is not None
                else 0.0
            ),
            finished_time=(
                request_output.metrics.finished_time
                if request_output.metrics and request_output.metrics.finished_time is not None
                else 0.0
            ),
        )

    @staticmethod
    def handle_prompt_response(request_output: RequestOutput, prompt: Prompt) -> Response:
        """Create a Response object from a RequestOutput and Prompt."""
        return Response(
            request_id=request_output.request_id,
            prompt_id=str(prompt.id),
            conversation_id=prompt.conversation_id,
            sys_prompt=prompt.sys_prompt,
            user_prompt=prompt.user_prompt,
            response=request_output.outputs[0].text if request_output.outputs else "No response",
            meta_data=prompt.meta_data,
            context=prompt.context,
            arrival_time=(
                request_output.metrics.arrival_time
                if request_output.metrics and request_output.metrics.arrival_time is not None
                else 0.0
            ),
            finished_time=(
                request_output.metrics.finished_time
                if request_output.metrics and request_output.metrics.finished_time is not None
                else 0.0
            ),
        )

    def print_response_summary(self) -> None:
        """Prints a summary of all responses."""
        for response in self.response_data:
            response.print_response()

    def print_response_by_prompt_id(self, prompt_id: str) -> None:
        """Print the response details for a specific prompt ID."""
        response = self._find_response_by("prompt_id", prompt_id)
        if response:
            response.print_response()
        else:
            print(f"No response found for Prompt ID: {prompt_id}")

    def print_response_by_request_id(self, request_id: str) -> None:
        """Print the response details for a specific request ID."""
        response = self._find_response_by("request_id", request_id)
        if response:
            response.print_response()
        else:
            logger.error(f"No response found for Request ID: {request_id}")

    def get_response_by_prompt_id(self, prompt_id: str) -> Response | None:
        """Return the response details for a specific prompt ID."""
        return self._find_response_by("prompt_id", prompt_id)

    def get_response_by_request_id(self, request_id: str | int) -> Response | None:
        """Return the response details for a specific request ID."""
        return self._find_response_by("request_id", str(request_id))

    def _find_response_by(self, key: str, value: str) -> Response | None:
        """Helper method to find a response by a given key and value."""
        for response in self.response_data:
            if getattr(response, key) == value:
                return response
        return None
