"""InferenceRunner class for running models."""

import os
import uuid

from openai import OpenAI
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.conversation import Conversation
from encourage.prompts.prompt_collection import PromptCollection


class ChatInferenceRunner:
    """Class for model inference."""

    def __init__(self, llm: LLM, sampling_parameters: SamplingParams):
        self.llm = llm
        self.sampling_parameters = sampling_parameters

    def run(self, conversation: Conversation) -> RequestOutput:
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
        self, sampling_parameters: SamplingParams, model_name: str, api_key: str = None
    ) -> None:
        # Allow for overriding API key and fallback to env variable
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        self.sampling_parameters = sampling_parameters
        self.model_name = model_name

    def run(self, conversation: Conversation) -> str:
        """Run the model with the given query."""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation.dialog,  # type: ignore
            max_tokens=self.sampling_parameters.max_tokens,
            temperature=self.sampling_parameters.temperature,
        )
        return RequestOutput(
            request_id=str(uuid.uuid4()),
            prompt="",
            prompt_token_ids=[],
            prompt_logprobs=[],
            finished=True,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=completion.choices[0].message.content,
                    token_ids=[],
                    cumulative_logprob=0.0,
                    logprobs=[],
                )
            ],
        )


class BatchInferenceRunner:
    """Class for model batch inference."""

    def __init__(self, llm: LLM, sampling_parameters: SamplingParams):
        self.llm = llm
        self.sampling_parameters = sampling_parameters

    def run(self, prompt_collection: PromptCollection) -> ResponseWrapper:
        """Performs batch inference with the provided prompts.

        Returns:
            ResponseWrapper: A wrapper object containing the responses.

        """
        reformated_prompts = [prompt.reformated for prompt in prompt_collection.prompts]
        responses = self.llm.generate(reformated_prompts, self.sampling_parameters)
        return ResponseWrapper.from_prompt_collection(responses, prompt_collection)
