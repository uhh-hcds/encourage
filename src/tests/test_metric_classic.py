import unittest

from encourage.llm.response import Response
from encourage.metrics.classic import (
    BLEU,
    ROUGE,
    BERTScore,
    ContextLength,
    GeneratedAnswerLength,
    MeanReciprocalRank,
    ReferenceAnswerLength,
)
from encourage.metrics.metric import MetricOutput


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Sample responses for testing
        self.responses = [
            Response(
                request_id="1",
                prompt_id="p1",
                sys_prompt="System prompt example.",
                user_prompt="User prompt example.",
                response="This is a generated answer.",
                conversation_id=1,
                meta_data={
                    "reference_answer": "This is the reference answer.",
                    "reference_document": ["doc1"],  # Required field for MRR
                },
                context={
                    "contexts": [  # Required field for MRR
                        {"content": "Here is an example content", "document": "doc1", "score": 1.0},
                        {"content": "Here is example content", "document": "doc2", "score": 0.5},
                    ]
                },
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
                meta_data={
                    "reference_answer": "Another reference answer.",
                    "reference_document": ["doc2"],  # Required field for MRR
                },
                context={
                    "contexts": [  # Required field for MRR
                        {"content": "Here is an example content", "document": "doc2", "score": 1.0},
                        {"content": "Here is an example content", "document": "doc1", "score": 0.0},
                    ]
                },
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]

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
        self.assertAlmostEqual(output.score, 9.5)

    def test_bleu(self):
        metric = BLEU()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)

    def test_rouge(self):
        metric = ROUGE(rouge_type="rouge1")
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)

        metric = ROUGE(rouge_type="rouge2")
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)

        metric = ROUGE(rouge_type="rougeLsum")
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)

    def test_bertscore(self):
        metric = BERTScore(lang="en", rescale_with_baseline=True)
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)

    def test_mean_reciprocal_rank(self):
        metric = MeanReciprocalRank()
        output = metric(self.responses)
        self.assertIsInstance(output, MetricOutput)
        self.assertIsInstance(output.score, float)


if __name__ == "__main__":
    unittest.main()
