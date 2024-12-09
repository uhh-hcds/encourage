"""Module to handle individual response data."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData


@dataclass
class Response:
    """Class to handle individual response data."""

    request_id: str
    prompt_id: str
    sys_prompt: str
    user_prompt: str
    response: Any | str
    conversation_id: int = 0
    context: Context = field(default_factory=Context)
    meta_data: MetaData = field(default_factory=MetaData)
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

        if self.context.documents:
            response_details.append("📄 Context Documents:")
            for idx, document in enumerate(self.context.documents):
                response_details.append(
                    f"  {idx + 1}. {document.content} (Score: {document.score})"
                )
            response_details.append(" Prompt Variables:")
            for key, value in self.context.prompt_vars.items():
                response_details.append(f"  {key}: {value}")

        if self.sys_prompt and len(self.sys_prompt) > 50:
            system_prompt = f"{self.sys_prompt[:50]}[...]\n"
        else:
            system_prompt = f"{self.sys_prompt}\n"

        response_details.extend(
            [
                "",
                f"💬 Response:\n{self.response.strip() if isinstance(self.response, str) else str(self.response)}",  # noqa: E501
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

    def to_dict(self, truncated: bool = True) -> dict[str, Any]:
        """Get the response details as a dictionary."""
        return {
            "request_id": self.request_id,
            "prompt_id": self.prompt_id,
            "sys_prompt": self.sys_prompt if not truncated else self.sys_prompt[:50],
            "user_prompt": self.user_prompt,
            "response": self.response,
            "conversation_id": self.conversation_id,
            "meta_data": self.meta_data.to_dict(),
            "context": self.context.to_dict(),
            "arrival_time": self.arrival_time,
            "finished_time": self.finished_time,
            "processing_time": self.processing_time,
        }

    def print_response(self) -> None:
        """Print or log the response details for a specific response."""
        print(self.to_string())
