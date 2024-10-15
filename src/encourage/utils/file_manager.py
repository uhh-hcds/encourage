"""A module to manage file operations."""

import json
from pathlib import Path
from typing import Any, Union

import yaml


class FileManager:
    """A class to manage file operations."""

    def __init__(self, filepath: Union[str, Path]) -> None:
        """Initialize FileManager with a given file path.

        Args:
            filepath (Union[str, Path]): The path to the file to manage.

        """
        self.filepath = Path(filepath)
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Ensure that the directory for the file exists."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def read(self, mode: str = "rt") -> Any:
        """Read the entire content of the file.

        Args:
            mode (str): The mode in which to open the file (default is 'rt').

        Returns:
            str: The content of the file.

        """
        with open(self.filepath, mode, encoding="utf-8") as file:
            return file.read()

    def write(self, data: str, mode: str = "wt") -> None:
        """Write the provided content to the file, overwriting any existing content.

        Args:
            data (str): The content to write to the file.
            mode (str): The mode in which to open the file (default is 'wt').

        """
        with open(self.filepath, mode, encoding="utf-8") as file:
            file.write(data)

    def append(self, data: str) -> None:
        """Append data to the end of the file.

        Args:
            data (str): The data to append to the file.

        """
        with open(self.filepath, "at", encoding="utf-8") as file:
            file.write(data)

    def delete(self) -> None:
        """Delete the file if it exists."""
        if self.file_exists():
            self.filepath.unlink()

    def file_exists(self) -> bool:
        """Check if the file exists.

        Returns:
            bool: True if the file exists, False otherwise.

        """
        return self.filepath.exists()

    def load_jsonlines(self) -> list[dict[str, Any]]:
        """Load data from a JSON lines file.

        Returns:
            list[dict[str, Any]]: A list of dictionaries with the data from the JSON lines file.

        """
        lines = self.read().strip().splitlines()
        return [json.loads(line) for line in lines]

    def dump_jsonlines(self, data: list[dict[str, Any]], ensure_ascii: bool = False) -> None:
        """Dump data to a JSON lines file.

        Args:
            data (list[dict[str, Any]]): The data to write to the JSON lines file.
            ensure_ascii (bool): Whether to escape non-ASCII characters in the output.
            kwargs: Additional keyword arguments to pass to json.dumps.

        """
        lines = "\n".join(json.dumps(item, ensure_ascii=ensure_ascii) for item in data)
        self.write(f"{lines}\n")

    def load_json(self) -> Any:
        """Load data from a JSON file.

        Returns:
            Any: The data loaded from the JSON file.

        """
        return json.loads(self.read())

    def dump_json(self, data: Any, ensure_ascii: bool = False) -> None:
        """Dump data to a JSON file.

        Args:
            data (Any): The data to write to the JSON file.
            ensure_ascii (bool): Whether to escape non-ASCII characters in the output.
            kwargs: Additional keyword arguments to pass to json.dumps.

        """
        self.write(json.dumps(data, ensure_ascii=ensure_ascii))

    def load_yaml(self) -> Any:
        """Load data from a YAML file.

        Returns:
            Any: The data loaded from the YAML file.

        """
        return yaml.safe_load(self.read())

    def dump_yaml(self, data: Any) -> None:
        """Dump data to a YAML file.

        Args:
            data (Any): The data to write to the YAML file.
            kwargs: Additional keyword arguments to pass to yaml.safe_dump.

        """
        self.write(yaml.safe_dump(data))
