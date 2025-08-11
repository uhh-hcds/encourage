"""InferenceRunner class for running models."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Type

from openai import OpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
)
from pydantic import BaseModel

from encourage.llm.batch_handler import process_batches
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.llm.sampling_params import SamplingParams
from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_collection import PromptCollection


class InferenceRunner(ABC):
    """Abstract base class for model inference."""

    def __init__(
        self,
        sampling_parameters: SamplingParams,
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
        sampling_parameters: SamplingParams,
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
            seed=self.sampling_parameters.seed,
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

    def __init__(
        self,
        sampling_parameters: SamplingParams,
        model_name: str,
        base_url: str = "http://localhost:18123/v1/",
        env_var_name: str = "VLLM_API_KEY",
    ):
        super().__init__(sampling_parameters, model_name, base_url, env_var_name)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def run(
        self,
        prompt_collection: PromptCollection,
        response_format: Type[BaseModel] | str | None = None,
        max_workers: int = 100,
    ) -> ResponseWrapper:
        """Run the model with the given queries."""
        extra_body: dict[str, Any] = {}
        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                extra_body = {"guided_json": response_format.model_json_schema()}
            elif isinstance(response_format, str):
                extra_body = {"guided_json": response_format}  # type: ignore

        all_messages = [prompt.conversation.dialog for prompt in prompt_collection.prompts]

        # Prepare args for process_batches
        args: dict[str, Any] = {
            "model": f"{self.model_name}",
            "max_tokens": self.sampling_parameters.max_tokens,
            "temperature": self.sampling_parameters.temperature,
            "top_p": self.sampling_parameters.top_p,
            "seed": self.sampling_parameters.seed,
            "extra_body": extra_body,
        }

        all_responses = process_batches(
            client=self.client,
            batch_messages=all_messages,
            args=args,
            max_workers=max_workers,
        )
        return ResponseWrapper.from_prompt_collection(all_responses, prompt_collection)


class OpenAIChatInferenceRunner(InferenceRunner):
    """Class for model inference."""

    def __init__(
        self,
        sampling_parameters: SamplingParams,
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
