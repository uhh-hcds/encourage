"""Module contains the MetaData class."""

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class MetaData:
    """Represents additional metadata associated with the context or documents.

    Attributes:
        tags (List[str]): A list of tags associated with the data.

    """

    tags: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> str:
        return self.tags[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.tags[key] = value

    def __delitem__(self, key: str) -> None:
        del self.tags[key]

    def __contains__(self, key: str) -> bool:
        return key in self.tags

    def __iter__(self) -> Iterable[str]:
        return iter(self.tags)

    def get_tag(self, key: str) -> str:
        """Get the value of a tag.

        Args:
            key (str): The key of the tag.

        Returns:
            str: The value of the tag.

        """
        return self.tags[key]

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the metadata.

        Args:
            key (str): The key of the tag.
            value (str): The value of the tag.

        """
        self.tags[key] = value
