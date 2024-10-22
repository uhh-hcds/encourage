"""Module to handle individual response data."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Union


@dataclass
class Response:
    """Class to handle individual response data."""

    request_id: str
    prompt_id: str
    sys_prompt: str
    user_prompt: str
    response: str
    conversation_id: int = 0
    meta_data: list[dict[str, Any]] = field(default_factory=list)
    context: Union[list[dict[str, Any]], dict[str, Any]] = field(default_factory=list)
    arrival_time: float = 0.0
    finished_time: float = 0.0
    processing_time: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate processing time."""
        self.processing_time = self.finished_time - self.arrival_time

    def format_timestamp(self, timestamp: float) -> str:
        """Format timestamp as a string."""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def formatted_arrival_time(self) -> str:
        """Get formatted arrival time."""
        return self.format_timestamp(self.arrival_time)

    @property
    def formatted_finished_time(self) -> str:
        """Get formatted finished time."""
        return self.format_timestamp(self.finished_time)

    def to_string(self) -> str:
        """Get the response details as a formatted string."""
        response_details = [
            "-" * 50,
            f"🧑‍💻 User Prompt:\n{self.user_prompt}",
        ]

        if isinstance(self.context, dict):
            keys = ", ".join(self.context.keys())
            response_details.append(f"📚 Added Context keys: {keys} (See Template for details.)")
        elif isinstance(self.context, list) and self.context:
            all_keys: set[str] = set()
            for item in self.context:
                all_keys.update(item.keys())
                keys = ", ".join(all_keys)
                response_details.append(
                    f"📚 Added Context keys: {keys} (See Template for details.)"
                )

        if self.sys_prompt and len(self.sys_prompt) > 200:
            system_prompt = f"{self.sys_prompt[:200]}[...]\n"
        else:
            system_prompt = f"{self.sys_prompt}\n"

        response_details.extend(
            [
                "",
                f"💬 Response:\n{self.response.strip()}",
                "",
                f"🤖 System Prompt:\n{system_prompt}",
                f"🗂️ Metadata: {self.meta_data if self.meta_data else 'None'}",
                f"🆔 Request ID: {self.request_id}",
                f"🆔 Prompt ID: {self.prompt_id}",
                f"🆔 Conversation ID: {self.conversation_id}",
                f"⏳ Processing Time: {round(self.processing_time, 4)} seconds",
                "",
            ]
        )

        return "\n".join(response_details)

    def to_dict(self) -> dict[str, Any]:
        """Get the response details as a dictionary."""
        return {
            "request_id": self.request_id,
            "prompt_id": self.prompt_id,
            "sys_prompt": self.sys_prompt,
            "user_prompt": self.user_prompt,
            "response": self.response,
            "conversation_id": self.conversation_id,
            "meta_data": self.meta_data,
            "context": self.context,
            "arrival_time": self.arrival_time,
            "finished_time": self.finished_time,
            "processing_time": self.processing_time,
        }

    def print_response(self) -> None:
        """Print or log the response details for a specific response."""
        print(self.to_string())
