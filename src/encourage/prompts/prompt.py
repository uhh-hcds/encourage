"""Prompt module to store prompt information."""

import json
import uuid
from dataclasses import dataclass, field


@dataclass
class Prompt:
    """Prompt dataclass to store prompt information."""

    sys_prompt: str
    user_prompt: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: int = 0
    context: list[dict] = field(default_factory=list)
    meta_data: list[dict] = field(default_factory=list)
    reformated: str = ""

    def __len__(self) -> int:
        return len(self.sys_prompt) + len(self.user_prompt)

    def __str__(self) -> str:
        return (
            f"id: {self.id}, "
            f"conversation_id: {self.conversation_id}, "
            f"sys_prompt: {self.sys_prompt},"
            f"user_prompt: {self.user_prompt}, "
            f"context: {self.context}, "
            f"meta_data: {self.meta_data}, reformated: {self.reformated}"
        )

    def to_json(self) -> str:
        """Serialize the Prompt object to a JSON string."""
        return json.dumps(
            {
                "id": str(self.id),
                "conversation_id": self.conversation_id,
                "sys_prompt": self.sys_prompt,
                "user_prompt": self.user_prompt,
                "context": self.context,
                "meta_data": self.meta_data,
                "reformated": self.reformated,
            }
        )

    @staticmethod
    def from_json(data: str) -> "Prompt":
        """Deserialize a JSON string to a Prompt object."""
        json_data = json.loads(data)
        return Prompt(
            id=str(json_data.get("id", "")),
            conversation_id=json_data.get("conversation_id", 0),
            sys_prompt=json_data.get("sys_prompt", ""),
            user_prompt=json_data.get("user_prompt", ""),
            context=json_data.get("context", []),
            meta_data=json_data.get("meta_data", []),
            reformated=json_data.get("reformated", ""),
        )
