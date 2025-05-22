import unittest
import uuid

from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics import (
    BLEU,
    F1,
    GLEU,
    ROUGE,
    BERTScore,
    ContextLength,
    ExactMatch,
    GeneratedAnswerLength,
    HitRateAtK,
    MeanReciprocalRank,
    RecallAtK,
    ReferenceAnswerLength,
)
from encourage.metrics.metric import MetricOutput
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample responses for testing
        self.responses_mock = [
            Response(
                request_id="1",
                prompt_id="p1",
                sys_prompt="System prompt example.",
                user_prompt="User prompt example.",
                response="This is a generated answer.",
                conversation_id=1,
                meta_data=MetaData(
                    tags={
                        "reference_answer": "This is a generated answer.",
                        "reference_document": Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content=""
                        ),
                    }
                ),
                context=Context.from_documents(
                    [
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"),
                            content="Here is an example content",
                            score=1.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"),
                            content="Here is example content",
                            score=0.5,
                        ),
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
            Response(
                request_id="2",
                prompt_id="p2",
                sys_prompt="Another system prompt.",
                user_prompt="Another user prompt.",
                response="Another generated answer.",
                conversation_id=2,
                meta_data=MetaData(
                    tags={
                        "reference_answer": "Another reference answer.",
                        "reference_document": Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content=""
                        ),
                    }
                ),
                context=Context.from_documents(
                    [
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"),
                            content="Here is example content",
                            score=1.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "4"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "3"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                        Document(
                            id=uuid.uuid5(uuid.NAMESPACE_DNS, "5"),
                            content="Here is an example content with extra",
                            score=0.0,
                        ),
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]
        self.responses = ResponseWrapper(self.responses_mock)  # Wrap the provided responses

    def test_generated_answer_length(self):
        metric = GeneratedAnswerLength()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 5.0)

    def test_reference_answer_length(self):
        metric = ReferenceAnswerLength()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 5.0)

    def test_context_length(self):
        metric = ContextLength()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 20.5)

    def test_bleu(self):
        metric = BLEU()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.76, places=2)

    def test_gleu(self):
        metric = GLEU()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.785, places=2)

    def test_rouge(self):
        metric = ROUGE(rouge_type="rouge1")
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.83, places=2)

        metric = ROUGE(rouge_type="rouge2")
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.5, places=2)

        metric = ROUGE(rouge_type="rougeLsum")
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.83, places=2)

    def test_bertscore(self):
        metric = BERTScore(lang="en", rescale_with_baseline=True)
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.808, places=2)

    def test_f1(self):
        metric = F1()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.83, places=2)

    def test_exact_match(self):
        metric = ExactMatch()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.5)

    def test_mean_reciprocal_rank(self):
        metric = MeanReciprocalRank()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.416, places=2)

    def test_recall(self):
        metric = RecallAtK(2)
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.5)

    def test_hit_rate(self):
        metric = HitRateAtK(2)
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.5)


if __name__ == "__main__":
    unittest.main()
