"""Module contains the MetaData class."""

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class MetaData:
    """Represents additional metadata associated with the context or documents."""

    tags: dict[str, str] = field(default_factory=dict)

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

    def to_dict(self) -> dict[str, str]:
        """Convert the metadata to a JSON-safe dictionary."""
        return {key: str(value) for key, value in self.tags.items()}
