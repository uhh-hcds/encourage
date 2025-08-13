"""Conversation dataclass to store conversation information."""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Collection, Iterator, Sequence


class Role(Enum):
    """Enum class to represent the role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Conversation:
    """Conversation dataclass to store conversation information."""

    dialog: list[dict[str, Any]] = field(default_factory=list)

    def __init__(
        self, sys_prompt: str = "", user_prompt: str | Sequence[Collection[Any]] = ""
    ) -> None:
        self.sys_prompt = sys_prompt
        if sys_prompt != "":
            self.dialog = [
                {"role": Role.SYSTEM.value, "content": self.sys_prompt},
            ]
        self.add_message(Role.USER.value, user_prompt)

    def add_message(self, role: str, content: str | Sequence[Collection[Any]]) -> None:
        """Add a new message to the conversation."""
        if role not in {role.value for role in Role}:
            raise ValueError(f"Role must be one of {', '.join([role.value for role in Role])}.")
        self.dialog.append({"role": role, "content": content})

    def get_messages_by_role(self, role: Role) -> list[dict[str, Any]]:
        """Retrieve all messages with a specific role."""
        if role not in Role:
            raise ValueError(f"Role must be one of {', '.join([role.value for role in Role])}.")
        return [msg for msg in self.dialog if msg["role"] == role.value]

    def get_last_message_by_user(self) -> str:
        """Retrieve the last message with a specific role."""
        user_messages = self.get_messages_by_role(Role.USER)
        return user_messages[-1]["content"] if user_messages else ""

    def clear_conversation(self) -> None:
        """Clear all messages in the conversation."""
        self.dialog = []
        self.add_message(Role.SYSTEM.value, self.sys_prompt)

    def to_json(self) -> str:
        """Serialize the Conversation object to a JSON string."""
        return json.dumps({"dialog": self.dialog})

    def print_chat_log(self) -> None:
        """Prints the chat log to the console."""
        for entry in self.dialog:
            role = entry["role"].capitalize()
            content = entry["content"]
            print(f"{role}: {content}\n")

    @staticmethod
    def from_json(data: str) -> "Conversation":
        """Deserialize a JSON string to a Conversation object."""
        json_data = json.loads(data)
        conv = Conversation(sys_prompt="")
        conv.dialog = json_data.get("dialog", [])
        return conv

    def __str__(self) -> str:
        return str(self.dialog)

    def __len__(self) -> int:
        return len(self.dialog)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.dialog[index]

    def __setitem__(self, index: int, value: dict[str, str]) -> None:
        self.dialog[index] = value

    def __delitem__(self, index: int) -> None:
        del self.dialog[index]

    def __iter__(self) -> Iterator:
        return iter(self.dialog)
