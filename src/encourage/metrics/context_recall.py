"""Context Recall metric for evaluating the context completeness."""

from typing import List, Literal, Optional

import numpy as np
from pydantic import BaseModel

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput, MetricTemplates
from encourage.prompts.prompt_collection import PromptCollection


class ContextRecall(Metric):
    """How complete the context is for generating the ground-truth."""

    def __init__(self, runner: BatchInferenceRunner) -> None:
        super().__init__(
            name="context-recall",
            description="Check how complete the context is for generating the ground-truth",
            runner=runner,
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Check how complete the context is for generating the ground-truth."""
        # Step 1: Prompts preparation
        contexts = [
            {
                "examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
                "task": {
                    "question": response.user_prompt,
                    "context": response.context,
                    "answer": response.response,
                },
                "output_model": ClassifiedSentencesList,
            }
            for response in responses
        ]

        # Step 2: Prompt Collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in responses],
            contexts=contexts,
            template_name=MetricTemplates.LLAMA3_CONTEXT_RECALL.value,
        )
        self.responses = self._runner.run(prompt_collection, schema=ClassifiedSentencesList)
        return self._calculate_metric()

    def _calculate_metric(self) -> MetricOutput:
        all_sentences = [response.response.sentences for response in self.responses]  # type: ignore
        total = [len(response.response.sentences) for response in self.responses]  # type: ignore
        attributed = [
            sum(sent.label == 1 for sent in response.response.sentences)  # type: ignore
            for response in self.responses
        ]
        raw = [a / t if t > 0 else np.nan for a, t in zip(attributed, total)]

        # Avoid division by zero when sum(total) is 0
        total_sum = sum(total)
        score = sum(attributed) / total_sum if total_sum > 0 else 0.0

        return MetricOutput(
            score=score,
            raw=raw,
            misc={"total": total, "attributed": attributed, "sentences": all_sentences},
        )


class ClassifiedSentence(BaseModel):
    """A sentence with a classification label."""

    sentence: str
    reason: str
    label: Literal[0, 1]


class ClassifiedSentencesList(BaseModel):
    """A list of classified sentences."""

    sentences: List[ClassifiedSentence]


class Example(BaseModel):
    """An example for the context recall metric."""

    question: str
    context: str
    answer: str
    classification: Optional[ClassifiedSentencesList] = None


EXAMPLE_1 = Example(
    question="What can you tell me about albert Albert Einstein?",
    context="Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",  # noqa: E501
    answer="Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895",  # noqa: E501
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",  # noqa: E501
                reason="The date of birth of Einstein is mentioned clearly in the context.",
                label=1,
            ),
            ClassifiedSentence(
                sentence="He received the 1921 Nobel Prize in Physics for his services to theoretical physics.",  # noqa: E501
                reason="The exact sentence is present in the given context.",
                label=1,
            ),
            ClassifiedSentence(
                sentence="He published 4 papers in 1905.",
                reason="There is no mention about papers he wrote in the given context.",
                label=0,
            ),
            ClassifiedSentence(
                sentence="Einstein moved to Switzerland in 1895.",
                reason="There is no supporting evidence for this in the given context.",
                label=0,
            ),
        ]
    ),
)


EXAMPLE_2 = Example(
    question="who won 2020 icc world cup?",
    context="The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",  # noqa: E501
    answer="England",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="England",
                reason="From context it is clear that England defeated Pakistan to win the World Cup.",  # noqa: E501
                label=1,
            )
        ]
    ),
)

EXAMPLE_3 = Example(
    question="What is the primary fuel for the Sun?",
    context="",
    answer="Hydrogen",
    classification=ClassifiedSentencesList(
        sentences=[
            ClassifiedSentence(
                sentence="Hydrogen",
                reason="The context contains no information",
                label=0,
            )
        ]
    ),
)
