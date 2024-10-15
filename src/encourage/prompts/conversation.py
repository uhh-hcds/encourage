"""Conversation dataclass to store conversation information."""

import json
from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Conversation dataclass to store conversation information."""

    sys_prompt: str
    dialog: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dialog = [
            {"role": "system", "content": self.sys_prompt},
        ]

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the conversation."""
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")
        self.dialog.append({"role": role, "content": content})

    def get_messages_by_role(self, role: str) -> list[dict[str, str]]:
        """Retrieve all messages with a specific role."""
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Role must be 'system', 'user', or 'assistant'.")
        return [msg for msg in self.dialog if msg["role"] == role]

    def clear_conversation(self) -> None:
        """Clear all messages in the conversation."""
        self.dialog = []
        self.add_message("system", self.sys_prompt)

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
