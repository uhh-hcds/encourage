"""Module contains the MetricType enum class."""

from enum import Enum

from encourage.metrics import (
    BLEU,
    F1,
    GLEU,
    ROUGE,
    AnswerFaithfulness,
    ContextLength,
    ContextPrecision,
    ContextRecall,
    ExactMatch,
    GeneratedAnswerLength,
    MeanReciprocalRank,
    NonAnswerCritic,
    NumberMatch,
    RecallAtK,
    ReferenceAnswerLength,
)


class MetricType(Enum):
    """Enum class for metric types."""

    F1 = "F1"
    BLEU = "BLEU"
    NumberMatch = "NumberMatch"
    MeanReciprocalRank = "MeanReciprocalRank"
    RecallAtK = "RecallAtK"
    ROUGE = "ROUGE"
    GeneratedAnswerLength = "GeneratedAnswerLength"
    ReferenceAnswerLength = "ReferenceAnswerLength"
    ContextLength = "ContextLength"
    GLEU = "GLEU"
    ExactMatch = "ExactMatch"
    AnswerFaithfulness = "AnswerFaithfulness"
    ContextPrecision = "ContextPrecision"
    ContextRecall = "ContextRecall"
    NonAnswerCritic = "NonAnswerCritic"

    def get_class(self) -> type:
        """Get the class associated with the metric type."""
        metrics = {
            MetricType.F1: F1,
            MetricType.BLEU: BLEU,
            MetricType.NumberMatch: NumberMatch,
            MetricType.MeanReciprocalRank: MeanReciprocalRank,
            MetricType.RecallAtK: RecallAtK,
            MetricType.ROUGE: ROUGE,
            MetricType.GeneratedAnswerLength: GeneratedAnswerLength,
            MetricType.ReferenceAnswerLength: ReferenceAnswerLength,
            MetricType.ContextLength: ContextLength,
            MetricType.GLEU: GLEU,
            MetricType.ExactMatch: ExactMatch,
            MetricType.AnswerFaithfulness: AnswerFaithfulness,
            MetricType.ContextPrecision: ContextPrecision,
            MetricType.ContextRecall: ContextRecall,
            MetricType.NonAnswerCritic: NonAnswerCritic,
        }
        return metrics[self]
