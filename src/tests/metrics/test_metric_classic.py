import unittest
import uuid
from unittest.mock import patch

from encourage.llm import ResponseWrapper
from encourage.metrics import (
    BLEU,
    GLEU,
    ROUGE,
    BERTScore,
    ContextLength,
    ExactMatch,
    GeneratedAnswerLength,
    HitRateAtK,
    MeanReciprocalRank,
    MetricOutput,
    RecallAtK,
    ReferenceAnswerLength,
)
from encourage.metrics.classic import F1, NDCG, F1Classification, MeanAveragePrecision
from encourage.prompts import Document, MetaData
from tests.fake_responses import create_responses


class TestMetrics(unittest.TestCase):
    def setUp(self):
        responses_content_list = [
            "This is a generated answer.",
            "Another generated answer.",
        ]
        document_list = [
            [
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"), content="Here is an example content"
                ),
                Document(id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content="Here is example content"),
            ],
            [
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"), content="Here is an example content"
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "4"),
                    content="Here is an example content with extra",
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "3"),
                    content="Here is an example content with extra",
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"),
                    content="Here is an example content with extra",
                ),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "5"),
                    content="Here is an example content with extra",
                ),
            ],
        ]
        meta_data_list = [
            MetaData(
                tags={
                    "reference_answer": "This is a generated answer.",
                    "reference_document": document_list[0][1],
                }
            ),
            MetaData(
                tags={
                    "reference_answer": "Another reference answer.",
                    "reference_document": document_list[0][0],
                }
            ),
        ]
        self.responses = ResponseWrapper(
            create_responses(2, responses_content_list, document_list, meta_data_list)
        )

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
        self.assertAlmostEqual(output.score, 21.0)

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
        with patch("encourage.metrics.BERTScore.__call__") as mock_bertscore:
            mock_bertscore.return_value = MetricOutput(score=0.808, raw=[])
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

    def test_f1_classification(self):
        metric = F1Classification(average="binary", pos_label="True")
        meta_data_list = [
            MetaData(
                tags={
                    "reference_answer": "True",
                }
            ),
            MetaData(
                tags={
                    "reference_answer": "False",
                }
            ),
            MetaData(
                tags={
                    "reference_answer": "True",
                }
            ),
            MetaData(
                tags={
                    "reference_answer": "False",
                }
            ),
        ]
        output = metric(
            ResponseWrapper(
                create_responses(
                    4, ["True", "False", "False", "True"], meta_data_list=meta_data_list
                )
            )
        )
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.5, places=2)

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
        self.assertAlmostEqual(output.score, 0.6, places=2)

    def test_mean_average_precision(self):
        document_list = [
            [
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"), content="Here is an example content"
                ),
            ],
            [
                Document(id=uuid.uuid5(uuid.NAMESPACE_DNS, "1"), content="Here is example content"),
                Document(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, "2"), content="Here is an example content"
                ),
            ],
        ]
        meta_data_list = [
            MetaData(
                tags={
                    "reference_answer": "This is a generated answer.",
                    "reference_document": document_list[0][:1],
                }
            ),
            MetaData(
                tags={
                    "reference_answer": "Another reference answer.",
                    "reference_document": document_list[1][1:],
                }
            ),
        ]
        responses = ResponseWrapper(
            create_responses(
                2,
                ["This is a generated answer.", "Another generated answer."],
                document_list,
                meta_data_list,
            )
        )
        metric = MeanAveragePrecision()
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.75, places=2)

    def test_normalized_discounted_cumulative_gain(self):
        metric = NDCG()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)
        self.assertAlmostEqual(output.score, 0.693, places=2)

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
