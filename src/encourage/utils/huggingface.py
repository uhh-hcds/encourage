"""Huggingface Dataset Handler."""

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from datasets import Dataset


class HuggingfaceDatasetHandler:
    """Handler for Huggingface Datasets."""

    def __init__(
        self, dataset_name: str, dataset: dict | list | pd.DataFrame, dataset_split: str = "train"
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset = self.transform_to_dataset(dataset)
        self.dataset_split = dataset_split

    def transform_to_dataset(self, dataset: dict | list | pd.DataFrame | Any) -> "Dataset":
        """Transform the data to a Huggingface Dataset."""
        from datasets import Dataset

        if isinstance(dataset, list):
            return Dataset.from_list(dataset)
        elif isinstance(dataset, dict):
            return Dataset.from_dict(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return Dataset.from_pandas(dataset)
        else:
            raise ValueError("Dataset must be a list, dictionary, or pandas DataFrame.")

    def push_to_hf(self, private: bool = True, **kwargs: Any) -> None:
        """Push the dataset to the Huggingface Hub."""
        self.dataset.push_to_hub(
            repo_id=self.dataset_name, split=self.dataset_split, private=private, **kwargs
        )
