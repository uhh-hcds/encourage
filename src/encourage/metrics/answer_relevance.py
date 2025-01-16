"""Answer Relevance metric for LLMs."""

from typing import Optional

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput, MetricTemplates
from encourage.metrics.non_answer_critic import NonAnswerCritic
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context


class AnswerRelevance(Metric):
    """How relevant the answer is to the question."""

    def __init__(
        self,
        runner: BatchInferenceRunner,
        model_name: str = "all-mpnet-base-v2",
        n_claims: int = 3,
    ) -> None:
        super().__init__(
            name="answer-relevance",
            description="Check how relevant the answer is to the question",
            runner=runner,
        )
        self.embeddings_model = SentenceTransformer(model_name)
        # TODO: Add a parameter for the number of generated questions
        self.n_claims = 3
        self.non_answer_critic = NonAnswerCritic(runner)

    def _question_similarity(self, question: str, generated: list[str]) -> float:
        q_embedding = self.embeddings_model.encode([question])
        gen_embeddings = self.embeddings_model.encode(generated)
        similarities = self.embeddings_model.similarity(q_embedding, gen_embeddings)[0]
        return similarities.mean().item()

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Check how relevant the answer is to the question."""
        # Step 1: identify noncomittal answers
        self.non_answer_result: MetricOutput = self.non_answer_critic(responses)

        # 0 = answer
        # 1 = non-answer
        committal_responses = [
            response for response, label in zip(responses, self.non_answer_result.raw) if label == 0
        ]

        if not committal_responses:
            print("No committal responses found.")
            return MetricOutput(score=0.0, raw=[], misc=self.non_answer_result.misc)

        # Step 2: generate questions (only for valid answers)
        contexts = [
            Context.from_prompt_vars(
                {
                    "examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
                    "task": {
                        "context": response.context,
                        "answer": response.response,
                    },
                    "output_model": GeneratedQuestion,
                }
            )
            for response in committal_responses
        ]

        # Step 2: Prompt Collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in committal_responses],
            contexts=contexts,
            template_name=MetricTemplates.LLAMA3_ANSWER_RELEVANCE.value,
        )

        self.responses: ResponseWrapper = self._runner.run(prompt_collection, GeneratedQuestion)
        return self._calculate_metric(input_responses=responses)

    def _calculate_metric(self, input_responses: ResponseWrapper) -> MetricOutput:
        # Step 3: Relevance calculation
        question_list: list[GeneratedQuestion] = []
        for response in self.responses:
            question_list.append(GeneratedQuestion.model_validate_json(response.response))
        scores = [
            self._question_similarity(response.user_prompt, generated.question)  # type: ignore
            for response, generated in zip(input_responses, question_list)
        ]

        # Return scores for all responses, where non-answers have a None score.
        rationales = self.non_answer_result.misc["raw_output"]
        committal_ixs = [i for i, label in enumerate(self.non_answer_result.raw) if label == 0]

        full_scores: list[Optional[float]] = [None] * len(question_list)
        for i, score in zip(committal_ixs, scores):
            full_scores[i] = score

        return MetricOutput(
            score=float(np.mean(scores)),
            raw=full_scores,
            misc={
                "noncommittal": self.non_answer_result.raw,
                "rationales": rationales,
                "generated_questions": question_list,
            },
        )


class GeneratedQuestion(BaseModel):
    """A generated question for the answer relevance metric."""

    question: str


class Example(BaseModel):
    """An example for the answer relevance metric."""

    context: str
    answer: str
    output: Optional[GeneratedQuestion] = None


EXAMPLE_1 = Example(
    answer="Albert Einstein was born in Germany.",
    context="Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time",  # noqa: E501
    output=GeneratedQuestion(question="Where was Albert Einstein born?"),
)

EXAMPLE_2 = Example(
    answer="It can change its skin color based on the temperature of its environment.",
    context="A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.",  # noqa: E501
    output=GeneratedQuestion(
        question="What unique ability does the newly discovered species of frog have?",
    ),
)

EXAMPLE_3 = Example(
    answer="The tallest mountain on earth is Mt. Everest.",
    context="The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas.",  # noqa: E501
    output=GeneratedQuestion(question="What is the tallest mountain on Earth?"),
)
