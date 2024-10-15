"""InferenceRunner class for running models."""

from vllm import LLM, RequestOutput, SamplingParams

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
