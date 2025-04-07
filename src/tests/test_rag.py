import unittest
import uuid
from unittest.mock import create_autospec

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG


class TestBaseRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_collection_{uuid.uuid4()}"
        # Create test documents directly instead of dataframe
        self.documents = [
            Document(
                id=uuid.uuid4(), content="AI is a field of study.", meta_data=MetaData({"id": "1"})
            ),
            Document(
                id=uuid.uuid4(), content="ML is a subset of AI.", meta_data=MetaData({"id": "2"})
            ),
        ]

        # Initialize RAG with the new interface
        self.rag = BaseRAG(
            context_collection=self.documents,
            template_name="llama3_conv.j2",
            collection_name=self.collection_name,
            top_k=1,
            embedding_function="all-MiniLM-L6-v2",
            device="cpu",
        )

        # Keep a reference to the original queries for testing
        self.queries = ["What is AI?", "Define ML"]

    def tearDown(self):
        self.rag.client.delete_collection(self.collection_name)

    def test_init_db_document_count(self):
        count = self.rag.client.count_documents(self.collection_name)
        self.assertEqual(count, 2)

    def test_retrieve_contexts(self):
        query = ["What is AI?"]
        documents = self.rag.retrieve_contexts(query)
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], list)
        self.assertIsInstance(documents[0][0], Document)
        self.assertGreaterEqual(len(documents[0]), 1)

    def test_run_with_mocked_runner(self):
        runner = create_autospec(BatchInferenceRunner)
        mock_responses = ResponseWrapper(["Mock response 1", "Mock response 2"])
        runner.run.return_value = mock_responses

        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_instruction=["Define ML", "What is AI?"],
        )

        runner.run.assert_called_once()
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(result.response_data, ["Mock response 1", "Mock response 2"])


if __name__ == "__main__":
    unittest.main()
