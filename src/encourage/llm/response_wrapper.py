"""Module that defines the ResponseWrapper class."""

import logging
from typing import Iterator

from openai.types.chat import ChatCompletion, ChatCompletionMessage

from encourage.llm.response import Response
from encourage.prompts.context import Context
from encourage.prompts.conversation import Conversation, Role
from encourage.prompts.meta_data import MetaData
from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_collection import PromptCollection
from encourage.utils.tracing import enable_tracing

# Configure logging
logger = logging.getLogger(__name__)


class ResponseWrapper:
    """Class that aggregates ChatCompletion with corresponding PromptCollection details."""

    @classmethod
    def from_prompt_collection(
        cls,
        chat_completions: list[ChatCompletion],
        collection: PromptCollection,
    ) -> "ResponseWrapper":
        """Create ResponseWrapper from ChatCompletion and PromptCollection or Conversation."""
        if len(chat_completions) != len(collection.prompts):
            raise ValueError("The number of request outputs does not match the number of prompts.")

        # Create responses for each chat_completion and its corresponding prompt
        responses = [
            cls.handle_prompt_response(chat_completion, prompt)
            for chat_completion, prompt in zip(chat_completions, collection.prompts)
        ]
        return cls(responses)

    @classmethod
    @enable_tracing(span_name="handle_prompt_response")
    def handle_prompt_response(cls, chat_completion: ChatCompletion, prompt: Prompt) -> Response:
        """Create a Response object from a ChatCompletion and Prompt."""
        prompt.conversation = transform_chat_completion_to_conversation(prompt.conversation)

        return Response(
            request_id=chat_completion.id,
            prompt_id=str(prompt.id),
            conversation_id=0,
            sys_prompt=prompt.conversation.sys_prompt,
            user_prompt=prompt.conversation.get_last_message_by_user(),
            response=chat_completion.choices[0].message.content
            if chat_completion.choices
            else "No response",
            meta_data=prompt.meta_data,
            context=prompt.context,
            arrival_time=(chat_completion.created if chat_completion.created is not None else 0.0),
            finished_time=0.0,
        )

    @classmethod
    def from_conversation(
        cls,
        chat_completions: list[ChatCompletion],
        conversation: Conversation,
        contexts: list[Context] = [],
        meta_datas: list[MetaData] = [],
    ) -> "ResponseWrapper":
        """Create ResponseWrapper from ChatCompletion and Conversation."""
        # Check for mismatched lengths
        if len(chat_completions) != len(conversation.get_messages_by_role(Role.USER)):
            raise ValueError("The number of request outputs does not match the number of messages.")

        # Safely access the system prompt, handle missing system messages
        sys_messages = conversation.get_messages_by_role(Role.SYSTEM)
        if not sys_messages:
            raise ValueError("No system messages found in the conversation.")
        sys_prompt = sys_messages[0]["content"]

        # Extract user messages
        user_messages = [msg["content"] for msg in conversation.get_messages_by_role(Role.USER)]

        responses = []
        meta_datas = meta_datas or [MetaData()] * len(chat_completions)
        contexts = contexts or [Context()] * len(chat_completions)
        for conversation_id, (chat_completion, user_message, meta_data, context) in enumerate(
            zip(chat_completions, user_messages, meta_datas, contexts)
        ):
            response = cls.handle_conversation_response(
                sys_prompt, conversation_id, chat_completion, user_message, context, meta_data
            )
            responses.append(response)

        return cls(responses)

    @staticmethod
    @enable_tracing(span_name="handle_conversation_response")
    def handle_conversation_response(
        sys_prompt: str,
        conversation_id: int,
        chat_completion: ChatCompletion,
        message: str,
        context: Context,
        meta_data: MetaData,
    ) -> Response:
        """Create a Response object from a ChatCompletion and Conversation."""
        return Response(
            request_id=chat_completion.id,
            prompt_id="",
            conversation_id=conversation_id,
            sys_prompt=sys_prompt,
            user_prompt=message,
            response=chat_completion.choices[0].message.content
            if chat_completion.choices
            else "No response",
            context=context,
            meta_data=meta_data,
            arrival_time=(chat_completion.created if chat_completion.created is not None else 0.0),
            finished_time=0.0,
        )

    def get_responses(self) -> list[str]:
        """Return the list of actually text responses from the model."""
        return [response.response for response in self.response_data]

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

    def __repr__(self) -> str:
        return f"ResponseWrapper({self.response_data})"

    def __init__(self, responses: list[Response]):
        self.response_data = responses

    def __iter__(self) -> Iterator[Response]:
        """Allows iteration over the responses."""
        return iter(self.response_data)

    def __getitem__(self, key: int) -> Response:
        """Returns the response at the given index."""
        if not self.response_data:
            raise IndexError("Response data is empty.")
        if key < 0 or key >= len(self.response_data):
            raise IndexError("Index out of range.")
        return self.response_data[key]

    def __len__(self) -> int:
        """Returns the number of responses."""
        return len(self.response_data)


def transform_chat_completion_to_conversation(conversation: Conversation) -> Conversation:
    """Transforms a Conversation object to a ChatCompletion object."""
    # Create a new conversation object
    new_dialog = []
    for message in conversation.dialog:
        if isinstance(message, ChatCompletionMessage):  # type: ignore
            message = {"role": Role.TOOL, "content": message.tool_calls[0].function.arguments}  # type: ignore
        new_dialog.append(message)
    conversation.dialog = new_dialog
    return conversation
