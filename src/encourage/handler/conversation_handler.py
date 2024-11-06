"""Prompt and Conversation collection module."""

from dataclasses import dataclass, field

from encourage.llm.inference_runner import ChatInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.conversation import Conversation
from encourage.prompts.prompt_reformatter import PromptReformatter


@dataclass
class ConversationHandler:
    """Collection of conversations."""

    chat_inference_runner: ChatInferenceRunner
    system_prompt: str
    user_prompts: list[str]
    contexts: list[dict] = field(default_factory=list)
    meta_data: list[dict] = field(default_factory=list)
    template_name: str = "generate_python_code_conv.j2"
    conversation: Conversation = field(init=False)

    def run(
        self,
    ) -> ResponseWrapper:
        """Run the conversation handler."""
        if (
            self.contexts
            and self.meta_data
            and (
                len(self.contexts) != len(self.user_prompts)
                or len(self.meta_data) != len(self.user_prompts)
            )
        ):
            raise ValueError("contexts and meta_data must have the same length as user_prompts")

        conversation = Conversation(sys_prompt=self.system_prompt)

        request_outputs = []
        for idx, user_prompt in enumerate(self.user_prompts):
            context = self.contexts[idx] if self.contexts else {}
            user_prompt_reformated = PromptReformatter.reformat_conversation(
                user_prompt, context, template_name=self.template_name
            )
            conversation.add_message("user", user_prompt_reformated)

            request_output = self.chat_inference_runner.run(conversation)
            request_outputs.append(request_output)
            conversation.add_message("assistant", request_output.outputs[0].text or "")

        self.conversation = conversation
        return ResponseWrapper.from_conversation(request_outputs, conversation, self.meta_data)

    def __len__(self) -> int:
        return len(self.conversation)
