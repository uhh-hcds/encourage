"""Answer Relevance metric for LLMs."""

from typing import Optional

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput, MetricTemplates
from encourage.metrics.non_answer_critic import NonAnswerCritic
from encourage.metrics.registry import register_metric
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context


@register_metric("answer-relevance")
class AnswerRelevance(Metric):
    """How relevant the answer is to the question."""

    @classmethod
    def requires_runner(cls) -> bool:
        """Return True if the metric requires an LLM runner."""
        return True

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
            response
            for response, output in zip(responses, self.non_answer_result.raw)
            if output.non_answer == 0  # noqa: E501
        ]

        if not committal_responses:
            print("No committal responses found.")
            return MetricOutput(score=0.0, raw=[], misc=self.non_answer_result.misc)

        # Step 2: Extract only the answers from the responses
        # We need this step because most of the time full answers contains questions too
        contexts_extract = []
        for response in committal_responses:
            context = Context.from_prompt_vars(
                {
                    "examples": [EXAMPLE_EXTRACT_1, EXAMPLE_EXTRACT_2, EXAMPLE_EXTRACT_3],
                    "task": {
                        # "context": response.context,
                        "answer": response.response,
                    },
                    "output_model": ExtractedAnswer,
                }
            )
            context.add_documents(response.context.documents)
            contexts_extract.append(context)

        prompt_collection_extract = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in committal_responses],
            contexts=contexts_extract,
            template_name=MetricTemplates.LLAMA3_ANSWER_EXTRACTION.value,
        )

        extracted_answers = self._runner.run(prompt_collection_extract, ExtractedAnswer)

        # Step 3: Build contexts with extracted answers (only for valid answers)
        contexts = []
        for response in extracted_answers:
            context = Context.from_prompt_vars(
                {
                    "examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
                    "task": {
                        "answer": response.response,
                    },
                    "output_model": GeneratedQuestion,
                }
            )
            context.add_documents(response.context.documents)
            contexts.append(context)

        # Step 4: Prompts for Answer Relevance metric calculation using the extracted answers
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in extracted_answers],
            contexts=contexts,
            template_name=MetricTemplates.LLAMA3_ANSWER_RELEVANCE.value,
        )

        self.responses: ResponseWrapper = self._runner.run(prompt_collection, GeneratedQuestion)
        # self.responses are the generated questions for the metric in the metric call

        return self._calculate_metric(input_responses=responses)
        # input_responses are the generated answers from the model that are passed to the metric

    def _calculate_metric(self, input_responses: ResponseWrapper) -> MetricOutput:
        # Step 3: Relevance calculation
        question_list: list[GeneratedQuestion] = []
        for response in self.responses:
            question_list.append(GeneratedQuestion.model_validate_json(response.response))
        scores = [
            self._question_similarity(response.meta_data["question"], generated.question)  # type: ignore
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


class ExtractedAnswer(BaseModel):
    """An extracted answer for the answer extraction prompt."""

    extracted_answer: str


class ExampleExtract(BaseModel):
    """An example for demonstrating answer extraction."""

    answer: str
    output: Optional[ExtractedAnswer] = None


EXAMPLE_EXTRACT_1 = ExampleExtract(
    answer="I see you asked 'Where was Albert Einstein from?' In the provided text, Albert Einstein is described as a German-born theoretical physicist. So, to answer the question directly: he was from Germany. Additionally, there's a note about his time in Switzerland, but that doesn't change his birthplace.",  # noqa: E501
    output=ExtractedAnswer(extracted_answer="He was from Germany."),
)


EXAMPLE_EXTRACT_2 = ExampleExtract(
    answer="Here's the entire conversation: The user asked, 'Who discovered penicillin?', and we found that Alexander Fleming made the discovery in 1928. The rest of the text talks about its impact on modern medicine. Therefore, the short answer is: Alexander Fleming.",  # noqa: E501
    output=ExtractedAnswer(extracted_answer="Alexander Fleming."),
)


EXAMPLE_EXTRACT_3 = ExampleExtract(
    answer="Answering the question 'Which planet is known as the Red Planet?' from the context: The Red Planet refers to Mars, which is often called the Red Planet due to its reddish appearance. They also mention Jupiter and Saturn, but those are gas giants, not 'red.'. So the best direct answer is: Mars.",  # noqa: E501
    output=ExtractedAnswer(extracted_answer="Mars."),
)
