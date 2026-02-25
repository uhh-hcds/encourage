"""Check how faithful the answer is to the question."""

import logging
import re
from typing import Iterator, Literal, Optional

import numpy as np
from pydantic import BaseModel, ValidationError

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.metrics.metric import Metric, MetricOutput, MetricTemplates
from encourage.metrics.registry import register_metric
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context

DEFAULT_NLI_SYS_PROMPT = (
    "Create verdicts for each statement based on the context. A verdict of 1 means the "
    "statement is supported by the context, while a verdict of 0 means it is not "
    "supported. Just return one output per statement, in the same order as the statements."
)

logger = logging.getLogger(__name__)


@register_metric("AnswerFaithfulness")
class AnswerFaithfulness(Metric):
    """Check how faithful the answer is to the question."""

    @classmethod
    def requires_runner(cls) -> bool:
        """Return True if the metric requires an LLM runner."""
        return True

    def __init__(
        self,
        runner: BatchInferenceRunner,
        split_examples: Optional[list["ExampleSplit"]] = None,
        nli_examples: Optional[list["ExampleNLI"]] = None,
        sys_prompt: Optional[str] = None,
        split_sys_prompt: str = "",
        nli_sys_prompt: str = DEFAULT_NLI_SYS_PROMPT,
    ) -> None:
        super().__init__(
            name="answer-faithfulness",
            description="Check how faithful the answer is to the question",
            runner=runner,
        )

        if sys_prompt is not None:
            split_sys_prompt = sys_prompt
            nli_sys_prompt = sys_prompt

        self._split_examples: list[ExampleSplit] = split_examples or [
            EXAMPLE_1_SPLIT,
            EXAMPLE_2_SPLIT,
        ]
        self._nli_examples: list[ExampleNLI] = nli_examples or [EXAMPLE_1_NLI, EXAMPLE_2_NLI]
        self._split_sys_prompt = split_sys_prompt
        self._nli_sys_prompt = nli_sys_prompt

    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        """Check how faithful the answer is to the question."""
        import nltk

        # Step 1: Split records into claims
        claim_prompts, response_indices = [], []
        for response_idx, response in enumerate(responses):
            text = re.sub(r"(\d+)\.", r"\1<NUM>", response.response)
            question = response.meta_data["question"] or response.user_prompt

            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                if not sent.strip():
                    continue
                tmp = {
                    "examples": self._split_examples,
                    "task": {
                        "question": question,
                        "answer": response.response,
                        "sentence": sent,
                    },
                    "output_model": OutputSplit,
                }
                claim_prompts.append(Context.from_prompt_vars(tmp))
                response_indices.append(response_idx)

        if not claim_prompts:
            return self._calculate_metric([OutputNLI(verdicts=[]) for _ in responses])

        # Step 2: Prompt Collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=self._split_sys_prompt,
            user_prompts=[" " for _ in claim_prompts],
            contexts=claim_prompts,
            template_name=MetricTemplates.ANSWER_FAITHFULNESS_SPLIT.value,
        )
        # Step 3: Generate claims
        responses_claims: ResponseWrapper = self._runner.run(prompt_collection, OutputSplit)

        # Step 4: Gather claims per record
        # TODO Understand what happens here
        response_to_claims: list[list] = [[] for _ in range(len(responses))]
        for response, response_idx in zip(responses_claims, response_indices):
            try:
                parsed_output = OutputSplit.model_validate_json(
                    response.response
                )  # manual parsing required here # noqa: E501
                response_to_claims[response_idx] += parsed_output.simpler_statements
            except ValidationError as ve:
                logger.warning("Validation error while parsing split response: %s", ve)

        assert len(responses_claims) == len(response_indices)

        # Step 5: Prepare NLI prompts using PromptCollection
        nli_contexts = []
        nli_response_indices = []
        for response_idx, (response, claims) in enumerate(zip(responses, response_to_claims)):
            if not claims:
                continue
            nli_contexts.append(
                Context.from_prompt_vars(
                    {
                        "examples": self._nli_examples,
                        "task": {
                            "context": response.context.to_string(),
                            "statements": claims,
                        },
                        "output_model": OutputNLI,
                    }
                )
            )
            nli_response_indices.append(response_idx)

        if not nli_contexts:
            return self._calculate_metric([OutputNLI(verdicts=[]) for _ in responses])

        nli_prompt_collection = PromptCollection.create_prompts(
            sys_prompts=self._nli_sys_prompt,
            user_prompts=[" " for _ in nli_contexts],
            contexts=nli_contexts,
            template_name=MetricTemplates.ANSWER_FAITHFULNESS_NLI.value,
        )

        # Step 6: Generate NLI responses
        nli_responses: ResponseWrapper = self._runner.run(
            nli_prompt_collection,
            OutputNLI,
        )
        verdicts_by_response: list[OutputNLI] = [OutputNLI(verdicts=[]) for _ in responses]

        for nli_response, response_idx in zip(nli_responses, nli_response_indices):
            try:
                verdicts_by_response[response_idx] = OutputNLI.model_validate_json(
                    nli_response.response
                )
            except Exception:
                logger.exception("Error processing NLI response: %s", nli_response)
                logger.debug("Raw NLI response payload: %s", nli_response.response)
                verdicts_by_response[response_idx] = OutputNLI(verdicts=[])

        return self._calculate_metric(verdicts_by_response)

    def _calculate_metric(self, verdicts_lists: list["OutputNLI"]) -> MetricOutput:
        # Step 7: Process results
        total = [len(verdict_list) for verdict_list in verdicts_lists]
        supported = [
            sum(v.verdict == 1 for v in verdicts_list.verdicts) for verdicts_list in verdicts_lists
        ]
        scores = [s / t if t > 0 else np.nan for s, t in zip(supported, total)]

        # micro-average over all responses
        agg = sum(supported) / sum(total) if sum(total) > 0 else 0.0

        return MetricOutput(
            score=agg,
            raw=scores,
            misc={"supported": supported, "total": total, "claims": verdicts_lists},
        )


class Verdict(BaseModel):
    """Verdict for a statement."""

    statement: str
    reason: str
    verdict: Literal[0, 1]


class OutputSplit(BaseModel):
    """Output model for the AnswerFaithfulness metric using the SPLIT task."""

    simpler_statements: list[str]


class OutputNLI(BaseModel):
    """Output model for the AnswerFaithfulness metric using the NLI task."""

    verdicts: list[Verdict]

    def __len__(self) -> int:
        return len(self.verdicts)

    def __iter__(self) -> Iterator[Verdict]:  # type: ignore
        return iter(self.verdicts)


class ExampleSplit(BaseModel):
    """Example for the SPLIT task."""

    question: str
    answer: str
    sentence: str
    output: Optional[OutputSplit] = None


class ExampleNLI(BaseModel):
    """Example for the NLI task."""

    context: str
    statements: list[str]
    output: Optional[OutputNLI] = None


EXAMPLE_1_SPLIT = ExampleSplit(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",  # noqa: E501
    sentence="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time.",  # noqa: E501
    output=OutputSplit(
        simpler_statements=[
            "Albert Einstein was a German-born theoretical physicist.",
            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",  # noqa: E501
        ]
    ),
)

EXAMPLE_2_SPLIT = ExampleSplit(
    question="Who was Albert Einstein and what is he best known for?",
    answer="He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",  # noqa: E501
    sentence="He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",  # noqa: E501
    output=OutputSplit(
        simpler_statements=[
            "Albert Einstein was best known for developing the theory of relativity.",
            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",  # noqa: E501
        ]
    ),
)


EXAMPLE_1_NLI = ExampleNLI(
    context="John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",  # noqa: E501
    statements=[
        "John is majoring in Biology.",
        "John is taking a course on Artificial Intelligence.",
        "John is a dedicated student.",
        "John has a part-time job.",
    ],
    output=OutputNLI(
        verdicts=[
            Verdict(
                statement="John is majoring in Biology.",
                reason="John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",  # noqa: E501
                verdict=0,
            ),
            Verdict(
                statement="John is taking a course on Artificial Intelligence.",
                reason="The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",  # noqa: E501
                verdict=0,
            ),
            Verdict(
                statement="John is a dedicated student.",
                reason="The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",  # noqa: E501
                verdict=1,
            ),
            Verdict(
                statement="John has a part-time job.",
                reason="There is no information given in the context about John having a part-time job.",  # noqa: E501
                verdict=0,
            ),
        ]
    ),
)

EXAMPLE_2_NLI = ExampleNLI(
    context="Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",  # noqa: E501
    statements=["Albert Einstein was a genius."],
    output=OutputNLI(
        verdicts=[
            Verdict(
                statement="Albert Einstein was a genius.",
                reason="The context and statement are unrelated",
                verdict=0,
            ),
        ]
    ),
)
