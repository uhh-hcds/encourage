"""Module for reformatting prompts for batch inference."""

import logging
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template, exceptions

from encourage.prompts.prompt import Prompt

logger = logging.getLogger(__name__)


class PromptReformatter:
    """Class for reformatting prompts for batch inference."""

    LLAMA3_MODELS = {
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "llama3_default.j2",
        "meta-llama/Meta-Llama-3.1-8B": "llama3_default.j2",
        "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct": "llama3_default.j2",
    }
    PHI3_MODELS = {
        "microsoft/Phi-3-mini-128k-instruct": "phi3_default.j2",
        "microsoft/Phi-3-medium-128k-instruct": "phi3_default.j2",
    }
    PHI3_5_MODELS = {
        "microsoft/Phi-3.5-mini-instruct": "phi3-5_default.j2",
        "microsoft/Phi-3.5-MoE-instruct": "phi3-5_default.j2",
    }
    GEMMA_MODELS = {
        "google/gemma-2-2b-it": "gemma_default.j2",
        "google/gemma-2-9b-it": "gemma_default.j2",
    }
    AYA_MODELS = {
        "CohereForAI/aya-23-8B": "aya_default.j2",
    }

    MODEL_TEMPLATES = {
        **LLAMA3_MODELS,
        **PHI3_MODELS,
        **PHI3_5_MODELS,
        **GEMMA_MODELS,
        **AYA_MODELS,
    }

    TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompts/templates"
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    @classmethod
    def reformat_prompt(
        cls,
        prompt: Prompt,
        model_name: str = "",
        template_name: str = "",
    ) -> str:
        """Reformats the prompts and adds custom key-value pairs to the template."""
        if not model_name and not template_name:
            raise ValueError("Either model_name or template_name must be provided.")

        if model_name:
            template = cls.get_template(model_name)
        elif template_name:
            try:
                template = cls.env.get_template(template_name)
            except exceptions.TemplateNotFound:
                template = cls.get_custom_template(template_name)
        else:
            raise ValueError("Either model_name or template_name must be provided.")

        return template.render(
            {
                "system_prompt": prompt.sys_prompt,
                "user_prompt": prompt.user_prompt,
                **prompt.context,
            }
        )

    @classmethod
    def reformat_conversation(
        cls,
        user_prompt: str,
        context: dict,
        model_name: str = "",
        template_name: str = "",
    ) -> str:
        """Reformats the conversation and adds custom key-value pairs to the template."""
        if model_name:
            template = cls.get_template(model_name)
        elif template_name:
            try:
                template = cls.env.get_template(template_name)
            except exceptions.TemplateNotFound:
                template = cls.get_custom_template(template_name)
        else:
            raise ValueError("Either model_name or template_name must be provided.")

        return template.render({"user_prompt": user_prompt, **context})

    @classmethod
    def get_template(cls, model_name: str) -> Template:
        """Maps the model name to the corresponding template."""
        template_name = cls.MODEL_TEMPLATES.get(model_name)
        if not template_name:
            raise ValueError(f"Model {model_name} not supported.")
        return cls.env.get_template(template_name)

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
