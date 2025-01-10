import unittest
from unittest.mock import patch

import numpy as np

from encourage.llm.response import Response
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.metrics.answer_similarity import AnswerSimilarity
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData


class TestAnswerSimilarity(unittest.TestCase):
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
                meta_data=MetaData(
                    tags={
                        "reference_answer": "This is a generated answer.",
                        "reference_document": Document(id="1", content=""),
                    }
                ),
                context=Context.from_documents(
                    [
                        {"id": 1, "content": "Here is an example content", "score": 1.0},
                        {"id": 0, "content": "Here is example content", "score": 0.5},
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
                        "reference_document": Document(id="0", content=""),
                    }
                ),
                context=Context.from_documents(
                    [
                        {"id": 1, "content": "Here is example content", "score": 1.0},
                        {"id": 2, "content": "Here is an example content with extra", "score": 0.0},
                        {"id": 3, "content": "Here is an example content with extra", "score": 0.0},
                        {"id": 4, "content": "Here is an example content with extra", "score": 0.0},
                        {"id": 0, "content": "Here is an example content with extra", "score": 0.0},
                    ]
                ),
                arrival_time=0.0,
                finished_time=1.0,
            ),
        ]
        # Create a mock runner
        self.responses = ResponseWrapper(self.responses)
        self.model_name = "mock-model"  # Specify a mock model name

    @patch("encourage.metrics.answer_similarity.SentenceTransformer", autospec=True)
    def test_similarity_computation(self, mock_sentence_transformer):
        # Mock the SentenceTransformer model
        mock_model = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)

        # Mock similarity computation to return fixed values
        mock_model.similarity.return_value = np.array([[0.8, 0.3], [0.3, 0.9]])

        # Instantiate AnswerSimilarity with the mock model
        metric = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model  # Use the mocked model

        # Call the metric and check the output
        result = metric(self.responses)

        # Assertions
        self.assertAlmostEqual(result.score, 0.85, places=2)  # Mean similarity (0.8+0.9)/2
        self.assertEqual(result.raw, [0.8, 0.9])  # Diagonal values of the similarity matrix

    @patch("encourage.metrics.answer_similarity.SentenceTransformer", autospec=True)
    def test_empty_responses(self, mock_sentence_transformer):
        # Setup an empty ResponseWrapper
        empty_responses = ResponseWrapper([])

        # Instantiate AnswerSimilarity with a mock model
        mock_model = mock_sentence_transformer.return_value
        metric = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model  # Use the mocked model

        # Call the metric with empty responses
        result = metric(empty_responses)

        # Assertions for empty responses
        self.assertEqual(result.score, 0.0)  # Score should be 0 when no responses
        self.assertEqual(result.raw, [])  # Raw similarities should be empty

    @patch("encourage.metrics.answer_similarity.SentenceTransformer", autospec=True)
    def test_single_response(self, mock_sentence_transformer):
        # Set up a single response scenario
        single_response = ResponseWrapper([self.responses[0]])

        # Mock the SentenceTransformer model
        mock_model = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.random.rand(len(texts), 768)

        # Set a fixed similarity value for single response testing
        mock_model.similarity.return_value = np.array([[0.7]])

        # Instantiate AnswerSimilarity with the mock model
        metric = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model

        # Call the metric with a single response
        result = metric(single_response)

        # Assertions for single response
        self.assertAlmostEqual(result.score, 0.7, places=2)
        self.assertEqual(result.raw, [0.7])

    @patch("encourage.metrics.answer_similarity.SentenceTransformer", autospec=True)
    def test_embedding_computation(self, mock_sentence_transformer):
        # Mock the SentenceTransformer encode method to return embeddings
        mock_model = mock_sentence_transformer.return_value
        mock_model.encode.side_effect = lambda texts: np.ones((len(texts), 768))

        # Instantiate AnswerSimilarity with the mock model
        metric = AnswerSimilarity(model_name=self.model_name)
        metric.model = mock_model

        # Call _get_embedding directly to test embedding output
        embedding = metric._get_embedding("This is a test response")

        # Assertions for embedding output
        self.assertEqual(embedding.shape, (1, 768))  # Check shape of single embedding
        self.assertTrue((embedding == 1).all())  # Check if all values are 1, as mocked
