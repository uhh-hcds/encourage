"""Prompt and Conversation collection module."""

from dataclasses import dataclass, field

from encourage.llm.inference_runner import ChatInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.context import Context
from encourage.prompts.conversation import Conversation
from encourage.prompts.meta_data import MetaData
from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_reformatter import PromptReformatter


@dataclass
class ConversationHandler:
    """Collection of conversations."""

    chat_inference_runner: ChatInferenceRunner
    system_prompt: str
    user_prompts: list[str]
    contexts: list[Context] = field(default_factory=list)
    meta_data: list[MetaData] = field(default_factory=list)
    template_name: str = "llama3_conv.j2"
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
            context = self.contexts[idx] if self.contexts else Context()
            conversation.add_message("user", user_prompt)
            prompt = Prompt(
                conversation=conversation,
                context=context,
                meta_data=self.meta_data[idx] if self.meta_data else MetaData(),
            )
            prompt = PromptReformatter.reformat_user_prompt(
                prompt, template_name=self.template_name
            )

            request_output = self.chat_inference_runner.run(prompt, raw_output=True)
            request_outputs.append(request_output)
            conversation.add_message("assistant", request_output.choices[0].message.content or "")  # type: ignore

        self.conversation = conversation
        return ResponseWrapper.from_conversation(
            request_outputs,  # type: ignore
            conversation,
            self.contexts,
            self.meta_data,
        )

    def __len__(self) -> int:
        return len(self.conversation)
