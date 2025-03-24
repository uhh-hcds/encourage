"""Module contains the MetricType enum class."""

from enum import Enum

from encourage.metrics.answer_faithfulness import AnswerFaithfulness
from encourage.metrics.answer_relevance import AnswerRelevance
from encourage.metrics.answer_similarity import AnswerSimilarity
from encourage.metrics.classic import (
    BLEU,
    F1,
    GLEU,
    ROUGE,
    ContextLength,
    ExactMatch,
    GeneratedAnswerLength,
    MeanReciprocalRank,
    RecallAtK,
    ReferenceAnswerLength,
)
from encourage.metrics.context_precision import ContextPrecision
from encourage.metrics.context_recall import ContextRecall
from encourage.metrics.non_answer_critic import NonAnswerCritic
from encourage.metrics.number_match import NumberMatch


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
    AnswerRelevance = "AnswerRelevance"
    AnswerSimilarity = "AnswerSimilarity"
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
            MetricType.AnswerRelevance: AnswerRelevance,
            MetricType.AnswerSimilarity: AnswerSimilarity,
            MetricType.ContextPrecision: ContextPrecision,
            MetricType.ContextRecall: ContextRecall,
            MetricType.NonAnswerCritic: NonAnswerCritic,
        }
        return metrics[self]
