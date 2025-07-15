import unittest
import uuid
from unittest.mock import create_autospec

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import Document, MetaData
from encourage.rag import BaseRAG
from encourage.rag.base.config import BaseRAGConfig
from tests.fake_responses import create_responses


class TestBaseRAGIntegration(unittest.TestCase):
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
        config = BaseRAGConfig(
            context_collection=self.documents,
            template_name="test_template.j2",
            collection_name=self.collection_name,
            top_k=1,
            embedding_function="all-MiniLM-L6-v2",
            device="cpu",
        )

        self.rag = BaseRAG(config)

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

        runner.run.return_value = self.responses

        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_queries=["Define ML", "What is AI?"],
        )

        runner.run.assert_called_once()
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(result.response_data, self.responses.response_data)


if __name__ == "__main__":
    unittest.main()
