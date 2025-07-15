"""InferenceRunner class for running models."""

import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Type

from openai import OpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
    Choice,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel
from tqdm import tqdm

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_collection import PromptCollection

if TYPE_CHECKING:
    from vllm import SamplingParams


class InferenceRunner(ABC):
    """Abstract base class for model inference."""

    def __init__(
        self,
        sampling_parameters: "SamplingParams",
        model_name: str,
        base_url: str = "http://localhost:18123/v1/",
        env_var_name: str = "VLLM_API_KEY",
    ):
        self.sampling_parameters = sampling_parameters
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = os.getenv(env_var_name) or ""
        if not self.api_key:
            raise ValueError("API key cannot be empty")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> ChatCompletion | ResponseWrapper:
        """Abstract method to run the model with the given inputs."""
        pass


class ChatInferenceRunner(InferenceRunner):
    """Class for single chat inference."""

    def __init__(
        self,
        sampling_parameters: "SamplingParams",
        model_name: str,
        base_url: str = "http://localhost:18123/v1/",
        env_var_name: str = "VLLM_API_KEY",
    ):
        super().__init__(sampling_parameters, model_name, base_url, env_var_name)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def run(
        self,
        prompt: Prompt,
        response_format: Type[BaseModel] | None = None,
        raw_output: bool = False,
    ) -> ChatCompletion | ResponseWrapper:
        """Run the model with the given query."""
        if isinstance(prompt, PromptCollection):  # type: ignore
            raise ValueError("PromptCollection is not supported for single chat inference")

        extra_body = {}
        if response_format:
            extra_body = {"guided_json": response_format.model_json_schema()}

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt.conversation.dialog,  # type: ignore
            max_tokens=self.sampling_parameters.max_tokens,
            temperature=self.sampling_parameters.temperature,
            top_p=self.sampling_parameters.top_p,
            extra_body=extra_body,
        )
        if raw_output:
            return response
        return ResponseWrapper.from_prompt_collection(
            [response], PromptCollection.from_prompts([prompt])
        )


class ToolInferenceRunner(InferenceRunner):
    """Class for running tool calls."""

    def run(
        self, prompt: Prompt, tool_json: list[dict], tool_functions: list[Callable]
    ) -> ResponseWrapper:
        """Run the model with the given query and tool calls."""
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        messages = prompt.conversation.dialog
        import mlflow
        from mlflow.entities import SpanType

        with mlflow.start_span("ToolChain", span_type=SpanType.TOOL):
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                max_tokens=self.sampling_parameters.max_tokens,
                temperature=self.sampling_parameters.temperature,
                top_p=self.sampling_parameters.top_p,
                tools=tool_json,  # type: ignore
            )
            tool_call = completion.choices[0].message.tool_calls[0]  # type: ignore
            args = json.loads(tool_call.function.arguments)
            matching_function = next(
                (
                    func
                    for func in tool_functions
                    if getattr(func, "__name__", None) == tool_call.function.name
                ),
                None,
            )
            result = None
            if matching_function:
                result = matching_function(**args)
            else:
                print(f"No matching function found for {tool_call.function.name}")

            messages.append(completion.choices[0].message)  # type: ignore
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )
            completion_2 = client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                tools=tool_json,  # type: ignore
                max_tokens=self.sampling_parameters.max_tokens,
                temperature=self.sampling_parameters.temperature,
                top_p=self.sampling_parameters.top_p,
            )
        return ResponseWrapper.from_prompt_collection(
            [completion, completion_2], PromptCollection.from_prompts([prompt, prompt])
        )


class BatchInferenceRunner(InferenceRunner):
    """Class for batch chat inference."""

    def run(
        self,
        prompt_collection: PromptCollection,
        response_format: Type[BaseModel] | str | None = None,
        batch_size: int = 50,  # Default batch size
    ) -> ResponseWrapper:
        """Run the model with the given queries."""
        from litellm import batch_completion

        extra_body = {}
        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                extra_body = {"guided_json": response_format.model_json_schema()}
            if isinstance(response_format, str):
                extra_body = {"guided_json": response_format}  # type: ignore

        all_messages = [prompt.conversation.dialog for prompt in prompt_collection.prompts]
        total_samples = len(all_messages)

        # Calculate the number of batches
        num_batches = (total_samples + batch_size - 1) // batch_size
        all_responses = []

        # Process in batches with progress tracking
        with tqdm(total=total_samples, desc="Processing prompts") as bar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_samples)
                current_batch_size = end_idx - start_idx

                # Get current batch of messages
                batch_messages = all_messages[start_idx:end_idx]
                bar.set_description(
                    f"Batch {batch_idx + 1}/{num_batches} ({current_batch_size} prompts)"
                )

                # Process the batch
                batch_responses = batch_completion(
                    model=f"hosted_vllm/{self.model_name}",
                    messages=batch_messages,
                    max_tokens=self.sampling_parameters.max_tokens,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    temperature=self.sampling_parameters.temperature,
                    top_p=self.sampling_parameters.top_p,
                    extra_body=extra_body,
                )

                all_responses.extend(batch_responses)
                bar.update(current_batch_size)

        return ResponseWrapper.from_prompt_collection(
            list(map(model_response_to_chat_completion, all_responses)), prompt_collection
        )


class OpenAIChatInferenceRunner(InferenceRunner):
    """Class for model inference."""

    def __init__(
        self,
        sampling_parameters: "SamplingParams",
        model_name: str,
    ):
        env_var_name = "OPENAI_API_KEY"
        super().__init__(sampling_parameters, model_name, env_var_name=env_var_name)
        self.client = OpenAI()

    def run(
        self,
        prompt: Prompt,
        response_format: Type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Run the model with the given query."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt.conversation.dialog,  # type: ignore
            max_tokens=self.sampling_parameters.max_tokens,
            temperature=self.sampling_parameters.temperature,
            response_format=response_format,  # type: ignore
        )
        return ResponseWrapper.from_prompt_collection(
            [response], PromptCollection.from_prompts([prompt])
        )


def model_response_to_chat_completion(model_response: Any) -> ChatCompletion:
    """Convert a ModelResponse object to a ChatCompletion object."""
    from litellm.types.utils import ModelResponse

    if not isinstance(model_response, ModelResponse):
        # Handle case where response is not a ModelResponse
        return ChatCompletion(
            id="",
            choices=[],
            created=0,
            model="",
            object="chat.completion",
            system_fingerprint="",
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                completion_tokens_details=None,
                prompt_tokens_details=None,
            ),
        )
    else:
        # For valid ModelResponse objects
        return ChatCompletion(
            id=model_response.id,
            choices=[
                Choice(
                    finish_reason="length",
                    index=choice.index,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=choice.message.content,  # type: ignore
                        refusal=None,
                        role=choice.message.role,  # type: ignore
                        audio=None,
                        function_call=choice.message.function_call,  # type: ignore
                        tool_calls=choice.message.tool_calls or [],  # type: ignore
                    ),
                )
                for choice in model_response.choices
            ],
            created=model_response.created,
            model=model_response.model,  # type: ignore
            object=model_response.object,  # type: ignore
            service_tier=model_response.service_tier,  # type: ignore
            system_fingerprint=model_response.system_fingerprint,
            usage=CompletionUsage(
                completion_tokens=model_response.usage.completion_tokens,  # type: ignore
                prompt_tokens=model_response.usage.prompt_tokens,  # type: ignore
                total_tokens=model_response.usage.total_tokens,  # type: ignore
                completion_tokens_details=model_response.usage.completion_tokens_details,  # type: ignore
                prompt_tokens_details=model_response.usage.prompt_tokens_details,  # type: ignore
            ),
        )
