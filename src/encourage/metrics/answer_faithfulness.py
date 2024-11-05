"""Check how faithful the answer is to the question."""

from typing import Optional

import nltk
import numpy as np
from pydantic import BaseModel, conint

from encourage.llm.inference_runner import InferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.metric import LLMMetric, MetricTemplates
from encourage.prompts.prompt_collection import PromptCollection


class AnswerFaithfulness(LLMMetric):
    """Check how faithful the answer is to the question."""

    def __init__(self, runner: InferenceRunner) -> None:
        super().__init__(
            name="answer-faithfulness",
            description="Check how faithful the answer is to the question",
            runner=runner,
        )

    def __call__(self, responses: ResponseWrapper) -> dict:
        """Check how faithful the answer is to the question."""
        # Step 1: Split records into claims
        claim_prompts, response_indices = [], []
        for response_idx, response in enumerate(responses):
            sentences = nltk.sent_tokenize(response.response)
            # Step 2: generate questions (only for valid answers)
            contexts = [
                {
                    "examples": [EXAMPLE_1_SPLIT, EXAMPLE_2_SPLIT],
                    "task": {
                        "context": response.context,
                        "answer": response.response,
                        "sentence": sen,
                    },
                    "output_model": OutputSplit,
                }
                for sen in sentences
            ]
            claim_prompts += contexts
            response_indices += [response_idx] * len(sentences)

        # Step 2: Prompt Collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in contexts],
            contexts=contexts,
            template_name=MetricTemplates.LLAMA3_ANSWER_FAITHFULNESS_SPLIT.value,
        )

        # Step 2: Generate claims
        responses_claims: ResponseWrapper = self._runner.run(prompt_collection, schema=OutputSplit)

        # Step 3: Gather claims per record
        response_to_claims = [[] for _ in range(len(responses))]
        for response, response_idx in zip(responses_claims, response_indices):
            response_to_claims[response_idx] += response.response[1]

        # Step 4: Prepare NLI prompts using PromptCollection
        nli_contexts = [
            {
                "examples": [EXAMPLE_1_NLI, EXAMPLE_2_NLI],
                "task": {
                    "context": response.context,
                    "statements": claims,
                },
                "output_model": OutputNLI,
            }
            for response, claims in zip(responses, response_to_claims)
        ]

        nli_prompt_collection = PromptCollection.create_prompts(
            sys_prompts="",
            user_prompts=["" for _ in nli_contexts],
            contexts=nli_contexts,
            template_name=MetricTemplates.LLAMA3_ANSWER_FAITHFULNESS_NLI.value,
        )

        # Step 4: Generate NLI responses
        nli_responses: ResponseWrapper = self._runner.run(nli_prompt_collection, schema=OutputNLI)
        return self._calculate_metric(nli_responses)

    def _calculate_metric(self, nli_responses: ResponseWrapper) -> dict:
        # Step 6: Process results
        supported = [
            sum(v.verdict == 1 for v in response.response.verdicts) for response in nli_responses
        ]
        total = [len(response.response.verdicts) for response in nli_responses]
        scores = [s / t if t > 0 else np.nan for s, t in zip(supported, total)]
        claims = [response.response.verdicts for response in nli_responses]

        # micro-average over all responses
        agg = sum(supported) / sum(total)

        return {
            "score": agg,
            "raw": scores,
            "supported": supported,
            "total": total,
            "claims": claims,
        }


class Verdict(BaseModel):
    """Verdict for a statement."""

    statement: str
    reason: str
    verdict: conint(ge=0, le=1)  # type: ignore


class OutputSplit(BaseModel):
    """Output model for the AnswerFaithfulness metric using the SPLIT task."""

    simpler_statements: list[str]


class OutputNLI(BaseModel):
    """Output model for the AnswerFaithfulness metric using the NLI task."""

    verdicts: list[Verdict]


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
