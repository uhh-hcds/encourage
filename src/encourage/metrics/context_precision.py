"""Context Precision metric for evaluating the relevance of the context to the ground-truth."""

from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput, MetricTemplates
from encourage.prompts.prompt_collection import PromptCollection


class ContextPrecision(Metric):
    """How relevant the context is to the ground-truth answer."""

    def __init__(self, runner: BatchInferenceRunner) -> None:
        super().__init__(
            name="context-precision",
            description="Check how relevant the context is to the ground-truth answer.",
            runner=runner,
            required_context=["contexts"],
        )

    def _average_precision(self, labels: List[int]) -> float:
        """Computes average precision over a list of ranked results.

        Labels should be a list of binary labels, where 1 is relevant, and 0 is irrelevant.
        """
        total_relevant = sum(labels)
        if total_relevant == 0:
            return 0.0  # No relevant items, precision is undefined or zero

        numerator = sum((sum(labels[: i + 1]) / (i + 1)) * labels[i] for i in range(len(labels)))
        return numerator / total_relevant

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Check how relevant the context is to the ground-truth answer."""
        self.validate_nested_keys(responses)
        # Step 1: Prompts preparation
        contexts = [
            {
                "examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
                "task": {
                    "question": response.user_prompt,
                    "context": response.context,
                    "answer": response.response,
                },
                "output_model": Verdict,
            }
            for response in responses
        ]

        # Step 2: Prompt Collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in responses],
            contexts=contexts,
            template_name=MetricTemplates.LLAMA3_CONTEXT_PRECISION.value,
        )
        self.responses = self._runner.run(prompt_collection, schema=Verdict)
        return self._calculate_metric(responses)

    def _calculate_metric(self, input_responses: ResponseWrapper) -> MetricOutput:
        # Step 3: Precision computation
        precisions_per_questions = []
        all_labels = []
        current_idx = 0
        for response in input_responses:
            contexts_cnt = len(response.context["contexts"])
            verdicts = self.responses[current_idx : current_idx + contexts_cnt]  # type: ignore
            labels = [response.response.verdict for response in verdicts]  # type: ignore
            all_labels.append(labels)
            precision = self._average_precision(labels)
            precisions_per_questions.append(precision)
            current_idx += contexts_cnt

        agg = float(np.mean(precisions_per_questions))
        agg = agg if not np.isnan(agg) else 0.0

        # Step 4: Detailed Output
        return MetricOutput(
            score=agg, raw=precisions_per_questions, misc={"labeled_contexts": all_labels}
        )


class Verdict(BaseModel):
    """Verdict of the context precision task."""

    reason: str
    verdict: Literal[0, 1]


class Example(BaseModel):
    """Example for the context precision task."""

    question: str
    context: str
    answer: str
    verification: Optional[Verdict] = None


EXAMPLE_1 = Example(
    question="What is the tallest mountain in the world?",
    context="The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.",  # noqa: E501
    answer="Mount Everest.",
    verification=Verdict(
        reason="the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",  # noqa: E501
        verdict=0,
    ),
)
EXAMPLE_2 = Example(
    question="who won 2020 icc world cup?",
    context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",  # noqa: E501
    answer="England",
    verification=Verdict(
        reason="the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",  # noqa: E501
        verdict=1,
    ),
)

EXAMPLE_3 = Example(
    question="What can you tell me about albert Albert Einstein?",
    context="""Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",  # noqa: E501
    answer="Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895",  # noqa: E501
    verification=Verdict(
        reason="The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",  # noqa: E501
        verdict=1,
    ),
)
