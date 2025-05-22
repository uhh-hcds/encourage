"""Module contains the MetaData class."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass
class MetaData:
    """Represents additional metadata associated with the context or documents."""

    tags: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Optional[str]:
        """Retrieve the value of a tag safely, returning None if the key does not exist."""
        return self.tags.get(key)

    def __setitem__(self, key: str, value: str) -> None:
        """Set the value of a tag."""
        self.tags[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a tag."""
        del self.tags[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self.tags

    def __iter__(self) -> Iterable[str]:
        """Iterate over the keys in the metadata."""
        return iter(self.tags)

    def to_dict(self, truncated: bool = False) -> dict[str, str]:
        """Convert the metadata to a JSON-safe dictionary."""
        if not self.tags:
            return {}

        def convert_value(value: str, truncated: bool = False) -> str | Any:
            """Convert the value to a JSON-safe format."""
            if not isinstance(value, str) and hasattr(value, "to_dict"):
                return value.to_dict(truncated=truncated)
            return value

        return {key: convert_value(value, truncated=truncated) for key, value in self.tags.items()}

    @classmethod
    def from_dict(cls, meta_dict: dict[str, Any]) -> "MetaData":
        """Update the metadata from a dictionary."""

        def convert_value(value: Any) -> Any:
            """Convert the value to the appropriate type."""
            from encourage.prompts.context import Document

            if isinstance(value, dict) and "content" in value:
                return Document(**value)
            return value

        converted_dict = {key: convert_value(value) for key, value in meta_dict.items()}
        return cls(tags=converted_dict)
