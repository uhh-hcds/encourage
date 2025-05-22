"""Base class for metric calculations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper


class MetricTemplates(Enum):
    """Enum class for metric templates."""

    ANSWER_FAITHFULNESS_NLI = "answer_faithfulness_nli.j2"
    ANSWER_FAITHFULNESS_SPLIT = "answer_faithfulness_split.j2"
    ANSWER_RELEVANCE = "answer_relevance.j2"
    CONTEXT_PRECISION = "context_precision.j2"
    CONTEXT_RECALL = "context_recall.j2"
    NON_ANSWER_CRITIQUE = "non_answer_critic.j2"
    ANSWER_EXTRACTION = "answer_extraction.j2"


@dataclass
class MetricOutput:
    """Dataclass to store metric output."""

    score: float
    raw: list[float] | list[int] | list[float | None] | list[Any]
    misc: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the MetricOutput instance to a dictionary."""
        return {
            "score": self.score,
            "raw": self.raw,
            "misc": self.misc,
        }

    @classmethod
    def from_dict(cls, metric_dict: dict[str, Any]) -> "MetricOutput":
        """Create a MetricOutput object from a dictionary."""
        return cls(
            score=metric_dict["score"],
            raw=metric_dict["raw"],
            misc=metric_dict["misc"],
        )


class Metric(ABC):
    """Base class for metric calculations with optional LLM support."""

    def __init__(
        self,
        name: str,
        description: str,
        runner: BatchInferenceRunner = None,  #  type: ignore
        required_meta_data: list[str] = [],
        required_prompt_vars: list[str] = [],
        required_documents: bool = False,
    ):
        self._name = name
        self._description = description
        self._runner = runner  # Only used for LLM metrics
        self.required_meta_data = required_meta_data or []
        self.required_prompt_vars = required_prompt_vars or []
        self.required_documents = required_documents

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    @property
    def description(self) -> str:
        """Returns a brief description of the metric."""
        return self._description

    @classmethod
    def requires_runner(cls) -> bool:
        """Override in subclasses if runner is required."""
        return False

    def validate_nested_keys(self, responses: ResponseWrapper) -> None:
        """Validates that each response contains required fields in meta_data and context."""
        for response in responses:
            for key in self.required_meta_data:
                if key not in response.meta_data:
                    raise ValueError(f"meta_data must contain '{key}' for {self._name} metric.")
            for key in self.required_prompt_vars:
                if key not in response.context.prompt_vars:
                    raise ValueError(f"context must contain '{key}' for {self._name} metric.")

            if self.required_documents and len(response.context.documents) == 0:
                raise ValueError(f"context must contain documents for {self._name} metric.")
            if not isinstance(response.context.documents, list):
                raise ValueError(
                    f"""
                    {self._name.upper()}: response.context.documents must be a list
                    of documents.
                    """
                )

            for doc in response.context.documents:
                if not doc.content:
                    raise ValueError(
                        f"Each document must contain 'content' for {self._name} metric."
                    )
                if doc.score is not None and not isinstance(doc.score, (int, float)):
                    raise ValueError(
                        f"Document score must be a number or None for {self._name} metric."
                    )

    @abstractmethod
    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Abstract method to be implemented by subclasses."""
