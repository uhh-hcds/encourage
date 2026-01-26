"""Classic metrics for evaluating RAG."""

import re
import string
import uuid
from functools import cached_property
from typing import Any, Union

import ir_measures
import numpy as np

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput
from encourage.metrics.registry import register_metric
from encourage.prompts.context import Document


@register_metric("GeneratedAnswerLength")
class GeneratedAnswerLength(Metric):
    """Computes the average length of the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="generated_answer_length",
            description="Average length of the generated answers.",
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        from nltk.tokenize import word_tokenize

        lengths = [len(word_tokenize(r.response)) for r in responses]
        score = float(np.mean(lengths))
        return MetricOutput(score=score, raw=lengths)


@register_metric("ReferenceAnswerLength")
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
        from nltk.tokenize import word_tokenize

        self.validate_nested_keys(responses)
        lengths = [len(word_tokenize(r.meta_data["reference_answer"])) for r in responses]
        score = float(np.mean(lengths))
        return MetricOutput(score=score, raw=lengths)


@register_metric("ContextLength")
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
        from nltk.tokenize import word_tokenize

        self.validate_nested_keys(responses)
        lengths = [
            sum(len(word_tokenize(document.content)) for document in r.context.documents)
            for r in responses
        ]
        score = float(np.mean(lengths))
        return MetricOutput(score=score, raw=lengths)


@register_metric("BLEU")
class BLEU(Metric):
    """Computes the BLEU score for the generated answers."""

    def __init__(self, n_grams: int = 4) -> None:
        super().__init__(name="bleu", description="BLEU score for the generated answers")

        self.metric = self.load_metric
        self.n_grams = n_grams

    @cached_property
    def load_metric(self) -> Any:
        """Loads the BLEU metric."""
        from evaluate import load

        return load("bleu")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        output = self.metric.compute(
            predictions=[str(r.response) for r in responses],
            references=[[str(r.meta_data["reference_answer"])] for r in responses],
            max_order=self.n_grams,
        )

        if output is None or "bleu" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(score=output["bleu"], raw=[output["bleu"]])


@register_metric("gleu")
class GLEU(Metric):
    """Computes the GLEU score for the generated answers."""

    def __init__(self) -> None:
        super().__init__(name="gleu", description="GLEU score for the generated answers")
        self.metric = self.load_metric

    @cached_property
    def load_metric(self) -> Any:
        """Loads the GLEU metric."""
        from evaluate import load

        return load("google_bleu")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        output = self.metric.compute(
            predictions=[str(r.response) for r in responses],
            references=[str(r.meta_data["reference_answer"]) for r in responses],
        )
        if output is None or "google_bleu" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(score=output["google_bleu"], raw=[])


@register_metric("ROUGE")
class ROUGE(Metric):
    """Computes the ROUGE score for the generated answers."""

    def __init__(self, rouge_type: str) -> None:
        assert rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        super().__init__(
            name=rouge_type,
            description="ROUGE score for the generated answers",
            required_meta_data=["reference_answer"],
        )

        self.metric = self.load_metric
        self.rouge_type = rouge_type

    @cached_property
    def load_metric(self) -> Any:
        """Loads the ROUGE metric."""
        from evaluate import load

        return load("rouge")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)

        output = self.metric.compute(
            predictions=[str(r.response) for r in responses if "reference_answer" in r.meta_data],
            references=[
                str(r.meta_data["reference_answer"])
                for r in responses
                if "reference_answer" in r.meta_data
            ],
            rouge_types=[self.rouge_type],
            use_aggregator=False,
        )
        if output is None or self.rouge_type not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=float(np.mean(output[self.rouge_type])), raw=output[self.rouge_type]
        )


@register_metric("ROUGEDetailed")
class ROUGEDetailed(Metric):
    """Computes the ROUGE score for the generated answers.

    And also returns the precision and the recall in raw format.
    """

    def __init__(self, rouge_type: str) -> None:
        from rouge_score import rouge_scorer

        assert rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        super().__init__(
            name=rouge_type,
            description="ROUGE score for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        precision = []
        recall = []
        f1 = []
        for ref, pred in zip(
            [str(r.meta_data["reference_answer"]) for r in responses],
            [str(r.response) for r in responses],
        ):
            score = self.scorer.score(ref or "", pred)
            precision.append(score[self.rouge_type].precision)
            recall.append(score[self.rouge_type].recall)
            f1.append(score[self.rouge_type].fmeasure)

        f1_mean = sum(f1) / len(f1)
        return MetricOutput(score=f1_mean, raw=f1, misc={"precision": precision, "recall": recall})


@register_metric("BERTScore")
class BERTScore(Metric):
    """Computes the BERTScore for the generated answers."""

    def __init__(self, **metric_args: Any) -> None:
        super().__init__(
            name="bertscore",
            description="BERTScore for the generated answers",
            required_meta_data=["reference_answer"],
        )

        self.metric = self.load_metric
        self.metric_args = metric_args

    @cached_property
    def load_metric(self) -> Any:
        """Loads the BERTScore metric."""
        from evaluate import load

        return load("bertscore")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        result = self.metric.compute(
            predictions=[str(r.response) for r in responses],
            references=[str(r.meta_data["reference_answer"]) for r in responses],
            **self.metric_args,
        )
        if result is None or "f1" not in result:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=float(np.mean(result["f1"])),
            raw=result["f1"],
            misc={"precision": result["precision"], "recall": result["recall"]},
        )


@register_metric("F1")
class F1(Metric):
    """Computes the F1 score for the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="f1squad",
            description="Squad score for the generated answers",
            required_meta_data=["reference_answer"],
        )

        self.metric = self.load_metric

    @cached_property
    def load_metric(self) -> Any:
        """Loads the F1 SQuAD v2 metric."""
        from evaluate import load

        return load("squad_v2")

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
                    "answers": [{"text": str(r.meta_data["reference_answer"]), "answer_start": 0}],
                }
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
        )

        if output is None or "f1" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=float(np.mean(output["f1"]) / 100), raw=output["f1"], misc={"output": output}
        )


@register_metric("F1Classification")
class F1Classification(Metric):
    """Computes the F1 score for the generated answers."""

    def __init__(self, average="binary", pos_label="True") -> None:
        super().__init__(
            name="f1",
            description="F1 score for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.average = average
        self.metric = self.load_metric
        self.pos_label = pos_label

    @cached_property
    def load_metric(self) -> Any:
        """Loads the F1 metric."""
        from evaluate import load

        return load("f1")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Iterate over predictions and references
        for r in responses:
            formatted_predictions.append(1 if str(r.response) == self.pos_label else 0)
            formatted_references.append(
                1 if str(r.meta_data["reference_answer"]) == self.pos_label else 0
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
            average=self.average,
            pos_label=1,
        )
        if output is None or "f1" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=output["f1"],
            raw=output["f1"],
            misc={
                "average": self.average,
                "pos_label": self.pos_label,
                "predictions": str(formatted_predictions),
                "references": str(formatted_references),
            },
        )


@register_metric("Precision")
class Precision(Metric):
    """Computes the Precision score for the generated answers."""

    def __init__(self, average="binary") -> None:
        super().__init__(
            name="precision",
            description="Precision score for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.average = average
        self.metric = self.load_metric

    @cached_property
    def load_metric(self) -> Any:
        """Loads the Precision metric."""
        from evaluate import load

        return load("precision")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Use zip to iterate over predictions and references
        for r in responses:
            formatted_predictions.append(1 if str(r.response).lower() == "true" else 0)
            formatted_references.append(
                1 if str(r.meta_data["reference_answer"]).lower() == "true" else 0
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions, references=formatted_references, average=self.average
        )
        if output is None or "precision" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=output["precision"],
            raw=output["precision"],
            misc={
                "average": self.average,
                "predictions": str(formatted_predictions),
                "references": str(formatted_references),
            },
        )


@register_metric("Recall")
class Recall(Metric):
    """Computes the Recall score for the generated answers."""

    def __init__(self, average="binary") -> None:
        super().__init__(
            name="recall",
            description="Recall score for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.average = average
        self.metric = self.load_metric

    @cached_property
    def load_metric(self) -> Any:
        """Loads the Recall metric."""
        from evaluate import load

        return load("recall")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Use zip to iterate over predictions and references
        for r in responses:
            formatted_predictions.append(1 if str(r.response).lower() == "true" else 0)
            formatted_references.append(
                1 if str(r.meta_data["reference_answer"]).lower() == "true" else 0
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions, references=formatted_references, average=self.average
        )
        if output is None or "recall" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=output["recall"],
            raw=output["recall"],
            misc={
                "average": self.average,
                "predictions": str(formatted_predictions),
                "references": str(formatted_references),
            },
        )


@register_metric("Accuracy")
class Accuracy(Metric):
    """Computes the Accuracy score."""

    def __init__(self, average="binary") -> None:
        super().__init__(
            name="accuracy",
            description="Accuracy for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.average = average
        self.metric = self.load_metric

    @cached_property
    def load_metric(self) -> Any:
        """Loads the Accuracy metric."""
        from evaluate import load

        return load("accuracy")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        # Initialize empty lists for formatted predictions and references
        formatted_predictions = []
        formatted_references = []

        # Use zip to iterate over predictions and references
        for r in responses:
            formatted_predictions.append(1 if str(r.response).lower() == "true" else 0)
            formatted_references.append(
                1 if str(r.meta_data["reference_answer"]).lower() == "true" else 0
            )

        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
        )
        if output is None or "accuracy" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=output["accuracy"],
            raw=output["accuracy"],
            misc={
                "predictions": str(formatted_predictions),
                "references": str(formatted_references),
            },
        )


@register_metric("DROP_F1")
class DropF1(Metric):
    """DROP-style F1 score with strict number match and multi-span alignment."""

    def __init__(self) -> None:
        super().__init__(
            name="drop_f1",
            description="DROP-style F1 score for numeric and multi-span answers",
            required_meta_data=["reference_answer"],
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Computes DROP-style F1 score."""
        self.validate_nested_keys(responses)

        def normalize_answer(s: str) -> str:
            """Lower text and remove punctuation, articles, and extra whitespace."""
            import string

            def remove_articles(text: str) -> str:
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text: str) -> str:
                return " ".join(text.split())

            def remove_punctuation(text: str) -> str:
                return "".join(ch for ch in text if ch not in set(string.punctuation))

            def lower(text: str) -> str:
                return text.lower()

            return white_space_fix(remove_articles(remove_punctuation(lower(s))))

        def get_tokens(s: str) -> list[str]:
            if not s:
                return []
            return normalize_answer(s).split()

        def compute_f1(a_gold: str, a_pred: str) -> float:
            gold_token = get_tokens(a_gold)
            pred_token = get_tokens(a_pred)
            common = set(gold_token) & set(pred_token)
            num_same = len(common)
            if len(gold_token) == 0 or len(pred_token) == 0:
                return float(gold_token == pred_token)
            if num_same == 0:
                return 0.0
            precision = num_same / len(pred_token)
            recall = num_same / len(gold_token)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        scores: list[float] = []
        for r in responses:
            prediction: str = str(r.response)
            references: Union[str, list[str]] = str(r.meta_data["reference_answer"]) or []
            if isinstance(references, str):
                references = [references]
            f1_scores = [compute_f1(ref, prediction) for ref in references]
            scores.append(max(f1_scores))

        return MetricOutput(score=float(np.mean(scores)), raw=scores)


@register_metric("ExactMatch")
class ExactMatch(Metric):
    """Computes the exact match for the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="exact_match",
            description="Exact match for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.metric = self.load_metric

    @cached_property
    def load_metric(self) -> Any:
        """Loads the Exact Match metric."""
        from evaluate import load

        return load("squad_v2")

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
                    "prediction_text": str(r.response),
                    "no_answer_probability": 0.0,
                }
            )
            formatted_references.append(
                {
                    "id": str(i),
                    "answers": [{"text": str(r.meta_data["reference_answer"]), "answer_start": 0}],
                }
            )
        # Call the compute function with formatted data
        output = self.metric.compute(
            predictions=formatted_predictions,
            references=formatted_references,
        )
        if output is None or "exact" not in output:
            return MetricOutput(score=0.0, raw=[])
        return MetricOutput(
            score=float(np.mean(output["exact"]) / 100),
            raw=output["exact"],
            misc={"output": output},
        )


class RetrievalMetric(Metric):
    """Base class for retrieval specific metrics."""

    def responses_to_trec(self, responses: ResponseWrapper) -> tuple:
        """Converts responses to TREC format."""
        qrels, run = {}, {}
        for response in responses:
            query_id = response.request_id
            ref_doc = response.meta_data["reference_document"]
            if ref_doc is None:
                ref_doc = Document(id=uuid.UUID(int=0), content="")
            elif isinstance(ref_doc, list):
                relevant = {str(doc.id if hasattr(doc, "id") else doc["id"]): 1 for doc in ref_doc}
            elif isinstance(ref_doc, Document):
                relevant = {str(ref_doc.id): 1}
            else:
                raise ValueError("reference_document must be a Document or list of Documents")
            retrieved = {
                str(document.id): document.score for document in response.context.documents
            }
            qrels[query_id] = relevant
            run[query_id] = retrieved
        return qrels, run


@register_metric("MeanReciprocalRank")
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
        return MetricOutput(score=float(np.mean(scores)), raw=scores)


@register_metric("MeanAveragePrecision")
class MeanAveragePrecision(RetrievalMetric):
    """Computes the Mean Average Precision (MAP) for the responses."""

    def __init__(self) -> None:
        super().__init__(
            name="map",
            description="Mean Average Precision (MAP) for the responses",
            required_meta_data=["reference_document"],
            required_documents=True,
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        qrels, run = self.responses_to_trec(responses)
        map_metric = ir_measures.AP()
        scores = [score.value for score in map_metric.iter_calc(qrels, run)]
        return MetricOutput(score=float(np.mean(scores)), raw=scores)


@register_metric("NDCG")
class NDCG(RetrievalMetric):
    """Computes the Normalized Discounted Cumulative Gain (NDCG) for the responses."""

    def __init__(self) -> None:
        super().__init__(
            name="ndcg",
            description="Normalized Discounted Cumulative Gain (NDCG) for the responses",
            required_meta_data=["reference_document"],
            required_documents=True,
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)
        qrels, run = self.responses_to_trec(responses)
        ndcg_metric = ir_measures.NDCG()
        scores = [score.value for score in ndcg_metric.iter_calc(qrels, run)]
        return MetricOutput(score=float(np.mean(scores)), raw=scores)


@register_metric("RecallAtK")
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
        return MetricOutput(score=float(np.mean(scores)), raw=scores)


@register_metric("HitRateAtK")
class HitRateAtK(RetrievalMetric):
    """Computes HitRate@k for the responses."""

    def __init__(self, k: int) -> None:
        super().__init__(
            name=f"hit{k}",
            description=(
                "Checks if at least one relevant document is in the top-k retrieved documents."
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
        return MetricOutput(score=float(np.mean(scores)), raw=scores)


@register_metric("SubstringEM")
class SubstringEM(Metric):
    """Computes the Substring Exact Match (EM) for the generated answers.

    This metric checks whether the normalized prediction is a substring of
    the normalized reference answer. Normalization includes lowercasing,
    removing punctuation, removing articles (a, an, the), and fixing whitespace.

    Supports single reference answers as well as multiple valid ground truths
    (as a list), returning the maximum score across all ground truths.
    """

    def __init__(self) -> None:
        super().__init__(
            name="substring_em",
            description="Substring Exact Match for the generated answers",
            required_meta_data=["reference_answer"],
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Computes the substring exact match score for all responses."""
        self.validate_nested_keys(responses)

        scores: list[int] = []
        for r in responses:
            prediction = str(r.response)
            ground_truths = r.meta_data["reference_answer"] or ""

            # Compute the best score across all valid ground truths
            score = self._max_score_over_ground_truths(prediction, ground_truths)
            scores.append(score)

        return MetricOutput(score=float(np.mean(scores)), raw=scores)

    def _normalize_answer(self, text: str) -> str:
        """Normalizes text for fuzzy matching."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Fix whitespace
        text = " ".join(text.split())

        return text

    def _is_substring_match(self, prediction: str, ground_truth: str) -> int:
        """Checks if normalized ground truth is a substring of normalized prediction."""
        normalized_pred = self._normalize_answer(prediction)
        normalized_gt = self._normalize_answer(ground_truth)
        return int(normalized_gt in normalized_pred)

    def _max_score_over_ground_truths(
        self, prediction: str, ground_truths: Union[str, list[str]]
    ) -> int:
        """Computes the maximum substring match score across all ground truths."""
        # Normalize ground_truths to a flat list
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        elif ground_truths and isinstance(ground_truths[0], list):
            # Flatten nested lists
            ground_truths = [gt for gt_list in ground_truths for gt in gt_list]

        # Return max score across all ground truths
        scores = [self._is_substring_match(prediction, gt) for gt in ground_truths]
        return max(scores) if scores else 0
