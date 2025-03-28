import unittest
import uuid
from unittest.mock import create_autospec

import pandas as pd

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG


class TestBaseRAGIntegration(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_collection_{uuid.uuid4()}"
        self.df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "question": ["What is AI?", "Define ML"],
                "program_answer": ["Artificial Intelligence", "Machine Learning"],
                "context": ["AI is a field of study.", "ML is a subset of AI."],
            }
        )

        self.rag = BaseRAG(
            qa_dataset=self.df,
            template_name="llama3_conv.j2",
            collection_name=self.collection_name,
            top_k=1,
            embedding_function="all-MiniLM-L6-v2",
            meta_data_keys=["id"],
            device="cpu",
        )

    def tearDown(self):
        self.rag.client.delete_collection(self.collection_name)

    def test_create_context_id(self):
        df = self.rag.create_context_id(self.df.copy())
        self.assertIn("context_id", df.columns)
        self.assertEqual(len(df["context_id"].unique()), 2)

    def test_create_metadata(self):
        metadata = self.rag.create_metadata()
        self.assertEqual(len(metadata), 2)
        self.assertIsInstance(metadata[0], MetaData)

    def test_prepare_contexts_for_db(self):
        context = self.rag.prepare_contexts_for_db(["id"])
        self.assertIsInstance(context, Context)
        self.assertEqual(len(context.documents), 2)

    def test_get_contexts_from_db(self):
        query = ["What is AI?"]
        contexts = self.rag.retrieve_contexts(query)
        self.assertEqual(len(contexts), 1)
        self.assertIsInstance(contexts[0], Context)
        self.assertGreaterEqual(len(contexts[0].documents), 1)

    def test_init_db_document_count(self):
        count = self.rag.client.count_documents(self.collection_name)
        self.assertEqual(count, 2)

    def test_run_with_mocked_runner(self):
        runner = create_autospec(BatchInferenceRunner)
        mock_responses = ResponseWrapper(["Mock response 1", "Mock response 2"])
        runner.run.return_value = mock_responses

        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            retrieval_instruction=["Define ML", "What is AI?"],
        )

        runner.run.assert_called_once()
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(result.response_data, ["Mock response 1", "Mock response 2"])


if __name__ == "__main__":
    unittest.main()
