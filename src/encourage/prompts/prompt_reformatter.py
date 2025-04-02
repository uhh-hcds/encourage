"""Module for reformatting prompts for batch inference."""

import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template, exceptions

from encourage.prompts.conversation import Role
from encourage.prompts.prompt import Prompt


class PromptReformatter:
    """Class for reformatting prompts for batch inference."""

    TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompts/templates"
    env = Environment(
        loader=FileSystemLoader([str(p) for p in TEMPLATE_DIR.glob("**/") if p.is_dir()])
    )

    @classmethod
    def reformat_user_prompt(
        cls,
        prompt: Prompt,
        template_name: str = "",
    ) -> Prompt:
        """Reformats the prompts and adds custom key-value pairs to the template."""
        if not template_name:
            raise ValueError("template_name must be provided.")

        try:
            template = cls.env.get_template(template_name)
        except exceptions.TemplateNotFound:
            template = cls.get_custom_template(template_name)

        rendered_prompt = template.render(
            {
                "user_prompt": prompt.conversation.get_last_message_by_user(),
                "documents": getattr(prompt.context, "documents", []),
                **(getattr(prompt.context, "prompt_vars", {}) or {}),
            }
        )
        if prompt.conversation.get_messages_by_role(Role.USER):
            user_messages = prompt.conversation.get_messages_by_role(Role.USER)
            last_message = user_messages[-1]
            if isinstance(last_message["content"], list):
                for item in last_message["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        item["text"] = rendered_prompt
                        break
            else:
                last_message["content"] = rendered_prompt
        return prompt

    @classmethod
    def get_custom_template(cls, template_name: str) -> Template:
        """Returns a custom template."""
        class_dir = os.path.dirname(os.path.abspath(cls.__module__))
        try:
            env = Environment(loader=FileSystemLoader(class_dir))

            # Try to load and return the template
            return env.get_template(template_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Template file '{template_name}' not found in '{class_dir}'."
            ) from e
