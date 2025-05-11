"""Replacement of ExactMatch metric for number and boolean answers."""

import math

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput
from encourage.metrics.registry import register_metric


@register_metric("NumberMatch")
class NumberMatch(Metric):
    """Computes the exact match for the generated answers."""

    def __init__(self, epsilon: float = 1e-2) -> None:
        super().__init__(
            name="number_match",
            description="Match numbers or booleans for the generated answers",
            required_meta_data=["reference_answer"],
        )
        self.epsilon = epsilon

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)

        scores = [
            int(self.compute_exact_score(r.response, r.meta_data["reference_answer"] or "", self.epsilon))
            for r in responses
        ]

        score = 100.0 * sum(scores) / len(scores) if scores else 0.0
        return MetricOutput(score=score, raw=scores)

    @staticmethod
    def compute_exact_score(prediction: str, ground_truth: str, epsilon: float = 1e-2) -> bool:
        """Given two strings pd and gt, find out whether pd is a correct answer to gt.

        Assume that gt is in correct format, i.e. either 'yes' 'no' or represents a float.
        Assume that pd is either 'yes' 'no' or represents a float, or 'None'.
        """
        if ground_truth in ["yes", "no"]:
            return ground_truth == prediction
        if prediction in ["yes", "no", "None"]:
            return False
        try:
            ground_truth_num = float(ground_truth)
            prediction_num = float(prediction)
        except ValueError:
            return False

        ground_truth_abs = abs(ground_truth_num)
        prediction_abs = abs(prediction_num)
        if ground_truth_abs < epsilon:
            return prediction_abs < epsilon
        if prediction_abs < epsilon:
            return False
        quotient = prediction_abs / ground_truth_abs
        quotient *= 0.1 ** math.floor(math.log10(quotient))
        # quotient should be around 1 or 10
        if quotient > 5:
            quotient *= 0.1
        return abs(quotient - 1) < epsilon
