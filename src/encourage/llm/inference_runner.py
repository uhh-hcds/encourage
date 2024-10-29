"""InferenceRunner class for running models."""

import os
import uuid
from typing import Callable, Union

from openai import OpenAI
from outlines import generate, models
from pydantic import BaseModel
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.conversation import Conversation
from encourage.prompts.prompt_collection import PromptCollection


class ChatInferenceRunner:
    """Class for model inference."""

    def __init__(self, llm: LLM, sampling_parameters: SamplingParams):
        self.llm = llm
        self.sampling_parameters = sampling_parameters

    def run(
        self,
        conversation: Conversation,
    ) -> RequestOutput:
        """Run the model with the given query."""
        chat_response = self.llm.chat(
            conversation.dialog,  # type: ignore
            self.sampling_parameters,
            use_tqdm=True,
        )
        return chat_response[0]


class OpenAIChatInferenceRunner:
    """Class for model inference."""

    def __init__(
        self, sampling_parameters: SamplingParams, model_name: str, api_key: str = ""
    ) -> None:
        # Allow for overriding API key and fallback to env variable
        api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        self.sampling_parameters = sampling_parameters
        self.model_name = model_name

    def run(self, conversation: Conversation) -> RequestOutput:
        """Run the model with the given query."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation.dialog,  # type: ignore
            max_tokens=self.sampling_parameters.max_tokens,
            temperature=self.sampling_parameters.temperature,
        )
        return get_new_request_output(completion.choices[0].message.content or "")


class BatchInferenceRunner:
    """Handles batch inference for a model with support for structured output."""

    def __init__(self, llm: LLM, sampling_parameters: SamplingParams):
        self.llm = llm
        self.sampling_parameters = sampling_parameters

    def run(
        self,
        prompt_collection: PromptCollection,
        schema: Union[str, BaseModel, Callable] = "",
    ) -> "ResponseWrapper":
        """Performs batch inference on a collection of prompts.

        Args:
            prompt_collection (PromptCollection): The prompts for inference.
            schema (str | BaseModel | callable): Schema required if structured_output is True.

        Returns:
            ResponseWrapper: Object containing formatted responses.

        """
        reformatted_prompts = [prompt.reformatted for prompt in prompt_collection.prompts]

        # Choose generation method based on output structure requirement
        if schema != "":
            model = models.VLLM(self.llm)
            generator = generate.json(model, schema_object=schema)
            result = generator(reformatted_prompts, sampling_params=self.sampling_parameters)  # type: ignore
            responses: list[RequestOutput] = [
                get_new_request_output(response)  # type: ignore
                for response in result  # type: ignore
            ]
        else:
            responses = self.llm.generate(reformatted_prompts, self.sampling_parameters)

        return ResponseWrapper.from_prompt_collection(responses, prompt_collection)


def get_new_request_output(generation_output: str) -> RequestOutput:
    """Get a new RequestOutput object."""
    return RequestOutput(
        request_id=str(uuid.uuid4()),
        prompt="",
        prompt_token_ids=[],
        prompt_logprobs=[],
        finished=True,
        outputs=[
            CompletionOutput(
                index=0,
                text=generation_output or "",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=[],
            )
        ],
    )
