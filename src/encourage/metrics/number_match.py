"""Replacement of ExactMatch metric for number and boolean answers."""

import math

from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput

EPSILON = 1e-10


class NumberMatch(Metric):
    """Computes the exact match for the generated answers."""

    def __init__(self) -> None:
        super().__init__(
            name="number_match",
            description="Match numbers or booleans for the generated answers",
            required_meta_data=["reference_answer"],
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Calls the metric calculation."""
        self.validate_nested_keys(responses)

        scores = [
            int(self.compute_score(r.response, r.meta_data["reference_answer"] or ""))
            for r in responses
        ]

        score = 100.0 * sum(scores) / len(scores) if scores else 0.0
        return MetricOutput(score=score, raw=scores)

    @staticmethod
    def compute_score(prediction: str, ground_truth: str) -> bool:
        """Check if the prediction matches the ground_truth.

        The ground_truth is assumed to be correctly formatted as 'yes', 'no', or a float.
        """
        if ground_truth in ["yes", "no"]:
            return ground_truth == prediction
        prediction = prediction.replace(" ", "")
        # when prediction contains %, there are two possible interpretations
        prediction_list = (
            [prediction.replace("%", "e-2"), prediction.replace("%", "")]
            if "%" in prediction
            else [prediction]
        )
        try:
            prediction_num = [float(x) for x in prediction_list]
        except ValueError:
            return False
        if "%" not in prediction:
            # there could be a "million"
            x = prediction_num[0]
            prediction_num += [x * 1e6, x * 1e-6]
        # there could be a minus sign
        prediction_num += [-x for x in prediction_num]
        ground_truth_num = float(ground_truth)
        ground_truth_abs = abs(ground_truth_num)
        if ground_truth_abs < EPSILON:
            return abs(prediction_num[0]) < EPSILON
        ratio = [x / ground_truth_num for x in prediction_num]
        if all(r < 0.6 or r > 2.1 for r in ratio):
            return False
        if any(0.999 <= r <= 1.001 for r in ratio):
            # this is good enough - financial people never care about more than 3 digits
            return True
        scale = 0.1 ** math.floor(math.log10(ground_truth_abs))
        # scale ground truth to something near 1, and try several possible roundings
        ground_truth_num *= scale
        if ground_truth_num >= 6:
            ground_truth_num *= 0.1
            scale *= 0.1
        prediction_num = [x * scale for x in prediction_num]
        for i in range(4):
            g_round = round(ground_truth_num, i)
            if any(abs(g_round - x) < EPSILON for x in prediction_num):
                return True
        return False
