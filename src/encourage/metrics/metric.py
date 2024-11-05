"""Base class for metric calculations."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

from encourage.llm.inference_runner import InferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper


class MetricTemplates(Enum):
    """Enum class for metric templates."""

    LLAMA3_ANSWER_FAITHFULNESS_NLI = "llama3_answer_faithfulness_nli.j2"
    LLAMA3_ANSWER_FAITHFULNESS_SPLIT = "llama3_answer_faithfulness_split.j2"
    LLAMA3_ANSWER_RELEVANCE = "llama3_answer_relevance.j2"
    LLAMA3_CONTEXT_PRECISION = "llama3_context_precision.j2"
    LLAMA3_CONTEXT_RECALL = "llama3_context_recall.j2"
    LLAMA3_NON_ANSWER_CRITIQUE = "llama3_non_answer_critic.j2"


@dataclass
class MetricOutput:
    """Dataclass to store metric output."""

    score: float
    raw: list[float]
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    raw_output: list[str] | None = None


class Metric:
    """Base class for metric calculations."""

    def __init__(
        self,
        name: str,
        description: str,
        required_meta_data: list[str] = [],
        required_context: list[str] = [],
    ):
        self._name = name
        self._description = description
        self.required_meta_data = required_meta_data or []
        self.required_context = required_context or []

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    @property
    def description(self) -> str:
        """Returns a brief description of the metric."""
        return self._description

    def validate_nested_fields(self, responses: ResponseWrapper) -> None:
        """Validates that each response contains required fields in meta_data and context."""
        for response in responses:
            for field in self.required_meta_data:
                if field not in response.meta_data:
                    raise ValueError(f"meta_data must contain '{field}' for {self._name} metric.")
            for field in self.required_context:
                if field not in response.context:
                    raise ValueError(f"context must contain '{field}' for {self._name} metric.")

    @abstractmethod
    def __call__(self, responses: ResponseWrapper) -> MetricOutput: ...  # noqa: D102


class LLMMetric:
    """Base class for LLM metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        runner: InferenceRunner | None = None,
        required_meta_data: list[str] = [],
        required_context: list[str] = [],
    ):
        self._name = name
        self._description = description
        self._runner = runner
        self.required_meta_data = required_meta_data or []
        self.required_context = required_context or []

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self._name

    @property
    def description(self) -> str:
        """Returns a brief description of the metric."""
        return self._description

    def validate_nested_fields(self, responses: ResponseWrapper) -> None:
        """Validates that each response contains required fields in meta_data and context."""
        for response in responses:
            for field in self.required_meta_data:
                if field not in response.meta_data:
                    raise ValueError(f"meta_data must contain '{field}' for {self._name} metric.")
            for field in self.required_context:
                if field not in response.context:
                    raise ValueError(f"context must contain '{field}' for {self._name} metric.")

    @abstractmethod
    def __call__(self, responses: ResponseWrapper) -> MetricOutput: ...  # noqa: D102
