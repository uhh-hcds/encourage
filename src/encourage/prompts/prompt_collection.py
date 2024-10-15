"""Prompt and Conversation collection module."""

import json
from dataclasses import dataclass
from typing import Iterator, Union

from encourage.prompts.prompt import Prompt
from encourage.prompts.prompt_reformatter import PromptReformatter


@dataclass
class PromptCollection:
    """Collection of prompts."""

    prompts: list[Prompt]

    @classmethod
    def create_prompts(
        cls,
        sys_prompts: list[str] | str,
        user_prompts: list[str],
        contexts: list[dict] = [],
        meta_datas: list[dict] = [],
        model_name: str = "",
        template_name: str = "",
    ) -> "PromptCollection":
        """Create a PromptCollection from the given data.

        Args:
            sys_prompts (List[str]): List of system prompts.
            user_prompts (List[str]): List of user prompts.
            contexts (Optional[List[List[Dict]]]): List of context dictionaries.
            meta_datas (Optional[List[List[Dict]]]): List of meta data dictionaries.
            model_name (str, optional): Name of the model. Defaults to "".
            template_name (str, optional): Name of the template. Defaults to "".

        Returns:
            PromptCollection: A new instance of PromptCollection.

        Raises:
            ValueError: If the lengths of sys_prompts and user_prompts do not match.

        """
        if isinstance(sys_prompts, str):
            sys_prompts = [sys_prompts] * len(user_prompts)
        else:
            if len(sys_prompts) != len(user_prompts):
                raise ValueError(
                    "The number of system prompts must match the number of user prompts."
                )

        prompts = []
        for idx, (sys_prompt, user_prompt) in enumerate(zip(sys_prompts, user_prompts)):
            context = contexts[idx] if contexts and idx < len(contexts) else []
            meta_data = meta_datas[idx] if meta_datas and idx < len(meta_datas) else []

            prompt = Prompt(
                sys_prompt=sys_prompt,
                user_prompt=user_prompt,
                context=context,  # type: ignore
                meta_data=meta_data,  # type: ignore
            )
            prompt.reformated = PromptReformatter.reformat_prompt(prompt, model_name, template_name)
            prompts.append(prompt)

        return cls(prompts=prompts)

    @classmethod
    def from_json(cls, json_data: str) -> "PromptCollection":
        """Deserialize a JSON string to a PromptCollection object.

        Args:
            json_data (str): JSON string representing the PromptCollection.

        Returns:
            PromptCollection: The deserialized PromptCollection object.

        """
        data = json.loads(json_data)
        prompts = [Prompt.from_json(json.dumps(p)) for p in data.get("prompts", [])]
        return cls(prompts=prompts)

    def to_json(self) -> str:
        """Serialize the PromptCollection to a JSON string.

        Returns:
            str: JSON string representation of the PromptCollection.

        """
        return json.dumps({"prompts": [json.loads(p.to_json()) for p in self.prompts]})

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self.prompts)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Prompt, "PromptCollection"]:
        if isinstance(idx, slice):
            return PromptCollection(prompts=self.prompts[idx])
        elif isinstance(idx, int):
            return self.prompts[idx]
        else:
            raise TypeError("Index must be an integer or slice.")
