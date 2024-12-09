"""Module contains the MetaData class."""

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class MetaData:
    """Represents additional metadata associated with the context or documents.

    Attributes:
        tags (dict[str, str]):
        A dictionary of key-value pairs representing tags
        associated with the metadata.

    """

    tags: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> str:
        """Retrieve the value of a tag using dictionary-style access.

        Args:
            key (str): The key of the tag to retrieve.

        Returns:
            str: The value associated with the given key.

        """
        return self.tags[key]

    def __setitem__(self, key: str, value: str) -> None:
        """Set the value of a tag using dictionary-style access.

        Args:
            key (str): The key of the tag to set.
            value (str): The value to associate with the key.

        """
        self.tags[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a tag from the metadata using dictionary-style access.

        Args:
            key (str): The key of the tag to delete.

        """
        del self.tags[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the metadata.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        """
        return key in self.tags

    def __iter__(self) -> Iterable[str]:
        """Iterate over the keys in the metadata.

        Returns:
            Iterable[str]: An iterator over the keys in the metadata.

        """
        return iter(self.tags)

    def get_tag(self, key: str) -> str:
        """Get the value of a tag.

        Args:
            key (str): The key of the tag.

        Returns:
            str: The value of the tag associated with the key.

        """
        return self.tags[key]

    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the metadata.

        Args:
            key (str): The key of the tag.
            value (str): The value of the tag.

        """
        self.tags[key] = value
