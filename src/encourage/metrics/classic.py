"""Classic metrics for evaluating RAG."""

from typing import Any

import evaluate
import ir_measures
import numpy as np
from nltk import word_tokenize

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput


class GeneratedAnswerLength(Metric):
    """Computes the average length of the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="generated_answer_length",
            description="Average length of the generated answers.",
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        lengths = [len(word_tokenize(r.response)) for r in responses]
        score = float(np.mean(lengths))
        return MetricOutput(score=score, raw=lengths)


class ReferenceAnswerLength(Metric):
    """Computes the average length of the reference answers."""

    def __init__(self) -> None:
        super().__init__(
            name="reference_answer_length",
            description="Average length of the reference answers",
            required_meta_data=["reference_answer"],
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        lengths = [len(word_tokenize(r.meta_data["reference_answer"])) for r in responses]
        score = float(np.mean(lengths))
        return MetricOutput(score=score, raw=lengths)


class ContextLength(Metric):
    """Computes the average length of the context."""

    def __init__(self) -> None:
        super().__init__(
            name="context_length",
            description="Average length of the context",
            required_documents=True,
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        lengths = [
            sum(len(word_tokenize(document.content)) for document in r.context.documents)
            for r in responses
        ]
        score = float(np.mean(lengths))
        return MetricOutput(score=score, raw=lengths)


class BLEU(Metric):
    """Computes the BLEU score for the generated answers."""

    def __init__(self, n_grams: int = 4) -> None:
        super().__init__(name="bleu", description="BLEU score for the generated answers")
        self.metric = evaluate.load("bleu")
        self.n_grams = n_grams

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        output = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[[r.meta_data["reference_answer"]] for r in responses],
            max_order=self.n_grams,
        )
        return MetricOutput(score=output["bleu"], raw=output)


class GLEU(Metric):
    """Computes the GLEU score for the generated answers."""

    def __init__(self) -> None:
        super().__init__(name="gleu", description="GLEU score for the generated answers")
        self.metric = evaluate.load("google_bleu")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        output = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data["reference_answer"] for r in responses],
        )
        return MetricOutput(score=output["google_bleu"], raw=[])


class ROUGE(Metric):
    """Computes the ROUGE score for the generated answers."""

    def __init__(self, rouge_type: str) -> None:
        assert rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        super().__init__(
            name=rouge_type,
            description="ROUGE score for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.metric = evaluate.load("rouge")
        self.rouge_type = rouge_type

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        output = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data["reference_answer"] for r in responses],
            rouge_types=[self.rouge_type],
            use_aggregator=False,
        )[self.rouge_type]
        scores = np.mean(output)
        return MetricOutput(score=scores, raw=output)


class BERTScore(Metric):
    """Computes the BERTScore for the generated answers."""

    def __init__(self, **metric_args: Any) -> None:
        super().__init__(
            name="bertscore",
            description="BERTScore for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.metric = evaluate.load("bertscore")
        self.metric_args = metric_args

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        result = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data["reference_answer"] for r in responses],
            **self.metric_args,
        )
        return MetricOutput(
            score=np.mean(result["f1"]),
            raw=result["f1"],
            misc={"precision": result["precision"], "recall": result["recall"]},
        )


class F1(Metric):
    """Computes the F1 score for the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="f1",
            description="F1 score for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.metric = evaluate.load("squad_v2")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Use zip to iterate over predictions and references
        for i, r in enumerate(responses):
            formatted_predictions.append(
                {
                    "id": str(i),
                    "prediction_text": r.response,
                    "no_answer_probability": 0.0,
                }
            )
            formatted_references.append(
                {
                    "id": str(i),
                    "answers": [{"text": r.meta_data["reference_answer"], "answer_start": 0}],
                }
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
        )
        scores = np.mean(output["f1"]) / 100
        return MetricOutput(score=scores, raw=output)


class ExactMatch(Metric):
    """Computes the exact match for the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="exact_match",
            description="Exact match for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.metric = evaluate.load("squad_v2")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Use zip to iterate over predictions and references
        for i, r in enumerate(responses):
            formatted_predictions.append(
                {
                    "id": str(i),
                    "prediction_text": r.response,
                    "no_answer_probability": 0.0,
                }
            )
            formatted_references.append(
                {
                    "id": str(i),
                    "answers": [{"text": r.meta_data["reference_answer"], "answer_start": 0}],
                }
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
        )
        scores = np.mean(output["exact"]) / 100
        return MetricOutput(score=scores, raw=output["exact"], misc={"output": output})


class RetrievalMetric(Metric):
    """Base class for retrieval specific metrics."""

    def responses_to_trec(self, responses: ResponseWrapper) -> tuple:
        """Converts responses to TREC format."""
        qrels, run = {}, {}
        for response in responses:
            query_id = response.request_id
            relevant = {str(response.meta_data["reference_document"].id): 1}  # type: ignore
            retrieved = {
                str(document.id): document.score for document in response.context.documents
            }
            qrels[query_id] = relevant
            run[query_id] = retrieved
        return qrels, run


class MeanReciprocalRank(RetrievalMetric):
    """Computes the Mean Reciprocal Rank (MRR) for the responses."""

    def __init__(self) -> None:
        super().__init__(
            name="mrr",
            description="Mean Reciprocal Rank (MRR) for the responses",
            required_meta_data=["reference_document"],
            required_documents=True,
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        qrels, run = self.responses_to_trec(responses)
        mrr = ir_measures.MRR()
        scores = [score.value for score in mrr.iter_calc(qrels, run)]
        return MetricOutput(score=np.mean(scores), raw=scores)


class RecallAtK(RetrievalMetric):
    """Computes Recall@k for the responses."""

    def __init__(self, k: int) -> None:
        super().__init__(
            name=f"recall{k}",
            description=(
                "Measures the proportion of relevant documents found "
                "within the top-k retrieved documents."
            ),
            required_meta_data=["reference_document"],
            required_documents=True,
        )
        self.k = k

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        qrels, run = self.responses_to_trec(responses)
        recall = ir_measures.R(cutoff=self.k)
        scores = [score.value for score in recall.iter_calc(qrels, run)]
        return MetricOutput(score=np.mean(scores), raw=scores)


class HitRateAtK(RetrievalMetric):
    """Computes HitRate@k for the responses."""

    def __init__(self, k: int) -> None:
        super().__init__(
            name=f"hit{k}",
            description=(
                "Checks if at least one relevant document is " "in the top-k retrieved documents."
            ),
            required_meta_data=["reference_document"],
            required_documents=True,
        )
        self.k = k

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        qrels, run = self.responses_to_trec(responses)
        hit_rate = ir_measures.Success(cutoff=self.k)
        scores = [score.value for score in hit_rate.iter_calc(qrels, run)]
        return MetricOutput(score=np.mean(scores), raw=scores)
