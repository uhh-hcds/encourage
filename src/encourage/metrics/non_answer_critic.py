"""Non-answer critic metric for the LLM."""

from typing import Literal, Optional

from pydantic import BaseModel

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput, MetricTemplates
from encourage.prompts.context import Context
from encourage.prompts.prompt_collection import PromptCollection


class NonAnswerCritic(Metric):
    """Check if generated_answer is a non-answer."""

    def __init__(self, runner: BatchInferenceRunner) -> None:
        super().__init__(
            name="non-answer_critic",
            description="Check if generated_answer is a non-answer.",
            runner=runner,
        )

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Check if generated_answer is a non-answer."""
        # Step 1: Prompts preparation
        contexts = [
            Context.from_prompt_vars(
                {
                    "examples": [EXAMPLE_1, EXAMPLE_2, EXAMPLE_3],
                    "answer": response.response,
                    "output_model": ClassifiedAnswer,
                }
            )
            for response in responses
        ]

        # Step 2: Prompt Collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in responses],
            contexts=contexts,
            template_name=MetricTemplates.LLAMA3_NON_ANSWER_CRITIQUE.value,
        )
        self._runner.add_schema(ClassifiedAnswer)
        self.responses = self._runner.run(prompt_collection)
        return self._calculate_metric()

    def _calculate_metric(self) -> MetricOutput:
        if not self.responses:
            return MetricOutput(score=0.0, raw=[], misc={"raw_output": []})

        critic_list: list[ClassifiedAnswer] = []
        for response in self.responses:
            critic_list.append(ClassifiedAnswer.model_validate_json(response.response))
        good_answers = [not critic.non_answer for critic in critic_list]

        return MetricOutput(
            score=sum(good_answers) / len(self.responses),
            raw=critic_list,
            misc={"raw_output": [critic.rationale for critic in critic_list]},
        )


class ClassifiedAnswer(BaseModel):
    """Classify if the answer is a non-answer."""

    rationale: str
    non_answer: Literal[0, 1]


class Example(BaseModel):
    """Example for the non-answer critic metric."""

    answer: str
    classification: Optional[ClassifiedAnswer] = None


EXAMPLE_1 = Example(
    answer="The information provided does not mention anything about compulsory elective modules or the number of free modules a student can choose from in the M.Sc. Data Science program at Marburg University.",  # noqa: E501
    classification=ClassifiedAnswer(
        rationale="The response indicates a lack of information, which is characteristic of a non-answer.",  # noqa: E501
        non_answer=1,
    ),
)

EXAMPLE_2 = Example(
    answer="The question is not clear because there is no specific module mentioned. The text only talks about enrollment as a doctoral student and provides information about the required documents and the enrollment process. It does not mention a specific module.",  # noqa: E501
    classification=ClassifiedAnswer(
        rationale="The response states that specific information is not provided, making it a non-answer.",  # noqa: E501
        non_answer=1,
    ),
)


EXAMPLE_3 = Example(
    answer="Master of Science (M.Sc.)",
    classification=ClassifiedAnswer(
        rationale="The response provides a specific piece of information, making it a valid answer.",  # noqa: E501
        non_answer=0,
    ),
)
