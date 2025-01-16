"""Prompt module to store prompt information."""

import json
import uuid
from dataclasses import dataclass, field

from encourage.prompts.context import Context
from encourage.prompts.conversation import Conversation
from encourage.prompts.meta_data import MetaData


@dataclass
class Prompt:
    """Prompt dataclass to store prompt information."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation: Conversation = field(default_factory=Conversation)
    context: Context = field(default_factory=Context)
    meta_data: MetaData = field(default_factory=MetaData)

    def __len__(self) -> int:
        return len(self.conversation)

    def __str__(self) -> str:
        return (
            f"id: {self.id}, "
            f"conversation: {self.conversation}, "
            f"context: {self.context}, "
            f"meta_data: {self.meta_data},"
        )

    def to_json(self) -> str:
        """Serialize the Prompt object to a JSON string."""
        return json.dumps(
            {
                "id": str(self.id),
                "conversation": self.conversation.to_json(),
                "context": self.context.to_dict(),
                "meta_data": self.meta_data.to_dict(),
            }
        )

    @staticmethod
    def from_json(data: str) -> "Prompt":
        """Deserialize a JSON string to a Prompt object."""
        json_data = json.loads(data)
        return Prompt(
            id=json_data.get("id", str(uuid.uuid4())),
            conversation=Conversation.from_json(json_data.get("conversation", Conversation())),
            context=json_data.get("context", Context()),
            meta_data=json_data.get("meta_data", MetaData()),
        )
