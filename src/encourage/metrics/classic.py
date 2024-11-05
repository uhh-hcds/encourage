"""Classic metrics for evaluating RAG."""

import evaluate
import ir_measures
import numpy as np
from nltk import word_tokenize

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput


class GeneratedAnswerLength(Metric):
    """Computes the average length of the generated answers."""

    def __init__(self):
        super().__init__(
            name="generated_answer_length", description="Average length of the generated answers."
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        lengths = [len(word_tokenize(r.response)) for r in responses]
        score = np.mean(lengths)
        return MetricOutput(score=score, raw=lengths)


class ReferenceAnswerLength(Metric):
    """Computes the average length of the reference answers."""

    def __init__(self):
        super().__init__(
            name="reference_answer_length",
            description="Average length of the reference answers",
            required_meta_data=["reference_answer"],
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_fields(responses)
        lengths = [len(word_tokenize(r.meta_data["reference_answer"])) for r in responses]
        score = np.mean(lengths)
        return MetricOutput(score=score, raw=lengths)


class ContextLength(Metric):
    """Computes the average length of the context."""

    def __init__(self):
        super().__init__(
            name="context_length",
            description="Average length of the context",
            required_context=["contexts"],
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_fields(responses)
        lengths = [
            sum(len(word_tokenize(context["content"])) for context in r.context["contexts"])
            for r in responses
        ]
        score = np.mean(lengths)
        return MetricOutput(score=score, raw=lengths)


class BLEU(Metric):
    """Computes the BLEU score for the generated answers."""

    def __init__(self):
        super().__init__(name="bleu", description="BLEU score for the generated answers")
        self.metric = evaluate.load("sacrebleu")

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        scores = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data["reference_answer"] for r in responses],
        )
        scores["score"] = scores["score"] / 100  # Normalize
        return MetricOutput(score=scores["score"], raw=scores)


class ROUGE(Metric):
    """Computes the ROUGE score for the generated answers."""

    def __init__(self, rouge_type: str):
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
        self.validate_nested_fields(responses)
        scores = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data["reference_answer"] for r in responses],
            rouge_types=[self.rouge_type],
            use_aggregator=False,
        )[self.rouge_type]
        agg = np.mean(scores)
        return MetricOutput(score=agg, raw=scores)


class BERTScore(Metric):
    """Computes the BERTScore for the generated answers."""

    def __init__(self, **metric_args):
        super().__init__(
            name="bertscore",
            description="BERTScore for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.metric = evaluate.load("bertscore")
        self.metric_args = metric_args

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_fields(responses)
        result = self.metric.compute(
            predictions=[r.response for r in responses],
            references=[r.meta_data["reference_answer"] for r in responses],
            **self.metric_args,
        )
        score = np.mean(result["f1"])
        return MetricOutput(
            score=score,
            raw=result["f1"],
            precision=result["precision"],
            recall=result["recall"],
            f1=result["f1"],
        )


class MeanReciprocalRank(Metric):
    """Computes the Mean Reciprocal Rank (MRR) for the responses."""

    def __init__(self):
        super().__init__(
            name="mrr",
            description="Mean Reciprocal Rank (MRR) for the responses",
            required_meta_data=["reference_document"],
            required_context=["contexts"],
        )

    def responses_to_trec(self, responses: ResponseWrapper) -> tuple:
        """Converts responses to TREC format."""
        qrels, run = {}, {}
        for response in responses:
            query_id = response.request_id
            relevant = {source: 1 for source in response.meta_data["reference_document"]}
            retrieved = {
                context["document"]: context["score"] for context in response.context["contexts"]
            }
            qrels[query_id] = relevant
            run[query_id] = retrieved
        return qrels, run

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_fields(responses)
        qrels, run = self.responses_to_trec(responses)
        mrr = ir_measures.MRR()
        scores = [score.value for score in mrr.iter_calc(qrels, run)]
        return MetricOutput(score=np.mean(scores), raw=scores)
