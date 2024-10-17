"""Module to handle individual response data."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    context: list[dict[str, Any]] = field(default_factory=list)
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

    def get_response_details(self) -> str:
        """Get the response details as a formatted string."""
        response_details = [
            "-" * 50,
            f"🧑‍💻 User Prompt:\n{self.user_prompt}",
        ]

        if self.context:
            keys = ", ".join(self.context[0].keys())
            response_details.append(f"📚 Added Context keys: {keys} (See Template for details.)")

        response_details.extend(
            [
                "",
                f"💬 Response:\n{self.response.strip()}",
                "",
                f"🤖 System Prompt:\n{self.sys_prompt}",
                f"🗂️ Metadata: {self.meta_data if self.meta_data else 'None'}",
                f"🆔 Request ID: {self.request_id}",
                f"🆔 Prompt ID: {self.prompt_id}",
                f"🆔 Conversation ID: {self.conversation_id}",
                f"⏳ Processing Time: {round(self.processing_time, 4)} seconds",
                "",
            ]
        )

        return "\n".join(response_details)

    def print_response(self) -> None:
        """Print or log the response details for a specific response."""
        print(self.get_response_detail())
