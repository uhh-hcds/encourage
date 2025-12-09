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
from encourage.metrics.classic import (
    F1,
    NDCG,
    F1Classification,
    MeanAveragePrecision,
    SubstringEM,
)
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

    def test_substring_em_exact_match(self):
        """Test SubstringEM when prediction exactly matches reference."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": "Paris"}),
            MetaData(tags={"reference_answer": "London"}),
        ]
        responses = ResponseWrapper(
            create_responses(2, ["Paris", "London"], meta_data_list=meta_data_list)
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 1.0)
        self.assertEqual(output.raw, [1, 1])

    def test_substring_em_substring_match(self):
        """Test SubstringEM when reference is a substring of prediction."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": "Paris"}),
            MetaData(tags={"reference_answer": "42"}),
        ]
        responses = ResponseWrapper(
            create_responses(
                2,
                ["The capital of France is Paris.", "The answer is 42 degrees."],
                meta_data_list=meta_data_list,
            )
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 1.0)
        self.assertEqual(output.raw, [1, 1])

    def test_substring_em_no_match(self):
        """Test SubstringEM when there is no match."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": "Paris"}),
            MetaData(tags={"reference_answer": "Berlin"}),
        ]
        responses = ResponseWrapper(
            create_responses(2, ["London", "Madrid"], meta_data_list=meta_data_list)
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 0.0)
        self.assertEqual(output.raw, [0, 0])

    def test_substring_em_normalization(self):
        """Test SubstringEM with normalization (case, punctuation, articles)."""
        metric = SubstringEM()
        meta_data_list = [
            # Case insensitivity
            MetaData(tags={"reference_answer": "PARIS"}),
            # Punctuation removal
            MetaData(tags={"reference_answer": "hello, world!"}),
            # Article removal
            MetaData(tags={"reference_answer": "the quick fox"}),
        ]
        responses = ResponseWrapper(
            create_responses(
                3,
                ["paris", "Hello World", "A quick fox jumps"],
                meta_data_list=meta_data_list,
            )
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 1.0)
        self.assertEqual(output.raw, [1, 1, 1])

    def test_substring_em_multiple_ground_truths(self):
        """Test SubstringEM with multiple valid ground truths (list)."""
        metric = SubstringEM()
        meta_data_list = [
            # Multiple valid answers, one matches
            MetaData(tags={"reference_answer": ["Paris", "City of Light"]}),
            # Multiple valid answers, none match
            MetaData(tags={"reference_answer": ["Berlin", "Munich"]}),
        ]
        responses = ResponseWrapper(
            create_responses(
                2,
                ["The answer is Paris", "The answer is London"],
                meta_data_list=meta_data_list,
            )
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertAlmostEqual(output.score, 0.5)
        self.assertEqual(output.raw, [1, 0])

    def test_substring_em_partial_match(self):
        """Test SubstringEM with mixed results."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": "correct"}),
            MetaData(tags={"reference_answer": "wrong"}),
            MetaData(tags={"reference_answer": "another"}),
            MetaData(tags={"reference_answer": "test"}),
        ]
        responses = ResponseWrapper(
            create_responses(
                4,
                [
                    "This is correct answer",
                    "This is incorrect",
                    "Something else",
                    "This is a test case",
                ],
                meta_data_list=meta_data_list,
            )
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        # 2 matches out of 4: "correct" in response 1, "test" in response 4
        self.assertAlmostEqual(output.score, 0.5)
        self.assertEqual(output.raw, [1, 0, 0, 1])

    def test_substring_em_none_reference(self):
        """Test SubstringEM when reference_answer is None."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": None}),
        ]
        responses = ResponseWrapper(
            create_responses(1, ["Any prediction"], meta_data_list=meta_data_list)
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        # None is converted to empty string "", and empty string is substring of any string
        self.assertAlmostEqual(output.score, 1.0)
        self.assertEqual(output.raw, [1])

    def test_substring_em_empty_string_reference(self):
        """Test SubstringEM when reference_answer is an empty string."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": ""}),
        ]
        responses = ResponseWrapper(
            create_responses(1, ["Any prediction"], meta_data_list=meta_data_list)
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        # Empty string normalized is empty, which is substring of any normalized string
        self.assertAlmostEqual(output.score, 1.0)
        self.assertEqual(output.raw, [1])

    def test_substring_em_empty_list_reference(self):
        """Test SubstringEM when reference_answer is an empty list."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": []}),
        ]
        responses = ResponseWrapper(
            create_responses(1, ["Any prediction"], meta_data_list=meta_data_list)
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        # Empty list is falsy, converted to "", and empty string is substring of any string
        self.assertAlmostEqual(output.score, 1.0)
        self.assertEqual(output.raw, [1])

    def test_substring_em_empty_prediction(self):
        """Test SubstringEM when prediction is an empty string."""
        metric = SubstringEM()
        meta_data_list = [
            MetaData(tags={"reference_answer": "Paris"}),
            MetaData(tags={"reference_answer": ""}),
        ]
        responses = ResponseWrapper(
            create_responses(2, ["", ""], meta_data_list=meta_data_list)
        )
        output = metric(responses)
        self.assertIsInstance(output, MetricOutput)
        # Empty prediction normalized is empty: "Paris" not in "" -> 0, "" in "" -> 1
        self.assertAlmostEqual(output.score, 0.5)
        self.assertEqual(output.raw, [0, 1])


if __name__ == "__main__":
    unittest.main()
