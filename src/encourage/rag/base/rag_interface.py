"""DatasetCollectionInterface class definition."""

from abc import ABC, abstractmethod

from encourage.llm import BatchInferenceRunner, ResponseWrapper


class RAGMethodInterface(ABC):
    """Interface for dataset collections."""

    @abstractmethod
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        retrieval_instruction: list[str] = [],
    ) -> ResponseWrapper:
        """Run the dataset."""
        pass
