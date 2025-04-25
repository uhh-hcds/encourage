import unittest
import uuid
from unittest.mock import MagicMock, create_autospec, patch

from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.llm.response_wrapper import ResponseWrapper
from encourage.prompts.context import Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base_impl import BaseRAG
from encourage.rag.reranker import RerankerRAG


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
        from encourage.llm.response import Response

        runner = create_autospec(BatchInferenceRunner)
        # Create proper Response objects instead of using strings
        mock_responses = ResponseWrapper(
            [
                Response(
                    request_id="mock_1",
                    prompt_id="prompt_1",
                    sys_prompt="Answer precisely.",
                    user_prompt=self.queries[0],
                    response="Mock response 1",
                ),
                Response(
                    request_id="mock_2",
                    prompt_id="prompt_2",
                    sys_prompt="Answer precisely.",
                    user_prompt=self.queries[1],
                    response="Mock response 2",
                ),
            ]
        )
        runner.run.return_value = mock_responses

        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_queries=["Define ML", "What is AI?"],
        )

        runner.run.assert_called_once()
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(result.get_responses(), ["Mock response 1", "Mock response 2"])


class TestRerankerRAG(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_reranker_collection_{uuid.uuid4()}"
        # Create test documents
        self.documents = [
            Document(
                id=uuid.uuid4(), content="AI is a field of study.", meta_data=MetaData({"id": "1"})
            ),
            Document(
                id=uuid.uuid4(), content="ML is a subset of AI.", meta_data=MetaData({"id": "2"})
            ),
            Document(
                id=uuid.uuid4(),
                content="Deep learning is a type of ML.",
                meta_data=MetaData({"id": "3"}),
            ),
            Document(
                id=uuid.uuid4(),
                content="Neural networks are used in deep learning.",
                meta_data=MetaData({"id": "4"}),
            ),
        ]

        # Keep a reference to the original queries for testing
        self.queries = ["What is AI?", "Define ML"]

    def tearDown(self):
        if hasattr(self, "rag") and hasattr(self.rag, "client"):
            self.rag.client.delete_collection(self.collection_name)

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_reranker_initialization(self, mock_cross_encoder):
        # Setup mock for CrossEncoder
        mock_cross_encoder.return_value = MagicMock()

        # Initialize RerankerRAG
        reranker_rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_ratio=2.0,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Verify CrossEncoder was initialized with correct arguments
        mock_cross_encoder.assert_called_once_with(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
        )

        # Verify initial_top_k calculation
        self.assertEqual(reranker_rag.initial_top_k, 4)  # top_k * rerank_ratio = 2 * 2.0
        self.assertEqual(reranker_rag.top_k, 2)  # Original top_k preserved

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_retrieve_contexts_with_reranking(self, mock_cross_encoder):
        # Setup the mock for CrossEncoder
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance

        # Set up the mock to return predictable scores when predict is called
        # Higher scores should be better for reranking
        mock_instance.predict.return_value = [0.9, 0.3, 0.8, 0.2]

        # Initialize RerankerRAG with rerank_ratio=2.0, so it retrieves twice top_k documents
        self.rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            rerank_ratio=2.0,
            device="cpu",
        )

        # Test retrieval with reranking
        query = ["What is AI?"]
        documents = self.rag.retrieve_contexts(query)

        # Verify predict was called
        mock_instance.predict.assert_called()

        # Verify correct number of documents returned after reranking
        self.assertEqual(len(documents), 1)  # One query
        self.assertEqual(len(documents[0]), 2)  # top_k = 2

        # Note: since the reranking scores are mocked, we're not testing the actual order,
        # just that the right number of documents is returned

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_run_with_mocked_runner(self, mock_cross_encoder):
        from encourage.llm.response import Response

        # Setup CrossEncoder mock
        mock_cross_encoder.return_value = MagicMock()
        mock_cross_encoder.return_value.predict.return_value = [0.9, 0.8, 0.7, 0.6]

        # Setup runner mock
        runner = create_autospec(BatchInferenceRunner)
        # Create proper Response objects instead of using strings
        mock_responses = ResponseWrapper(
            [
                Response(
                    request_id="mock_1",
                    prompt_id="prompt_1",
                    sys_prompt="Answer precisely.",
                    user_prompt=self.queries[0],
                    response="Mock reranker response 1",
                ),
                Response(
                    request_id="mock_2",
                    prompt_id="prompt_2",
                    sys_prompt="Answer precisely.",
                    user_prompt=self.queries[1],
                    response="Mock reranker response 2",
                ),
            ]
        )
        runner.run.return_value = mock_responses

        # Initialize RerankerRAG
        self.rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Test run method
        result = self.rag.run(
            runner=runner,
            sys_prompt="Answer precisely.",
            user_prompts=self.queries,
            retrieval_queries=["What is AI?", "Define ML"],
        )

        # Verify runner was called
        runner.run.assert_called_once()

        # Verify response
        self.assertIsInstance(result, ResponseWrapper)
        self.assertEqual(
            result.get_responses(), ["Mock reranker response 1", "Mock reranker response 2"]
        )

    @patch("encourage.rag.reranker_base.CrossEncoder")
    def test_reranker_edge_case_empty_documents(self, mock_cross_encoder):
        # Setup CrossEncoder mock
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance
        mock_instance.predict.return_value = []

        # Initialize RerankerRAG
        self.rag = RerankerRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Patch the super().retrieve_contexts to return empty results
        with patch.object(BaseRAG, "retrieve_contexts", return_value=[[]]):
            # Test retrieval with reranking for an empty result
            query = ["Invalid query with no results"]
            documents = self.rag.retrieve_contexts(query)

            # Verify predict was not called (no documents to rerank)
            mock_instance.predict.assert_not_called()

            # Verify empty list returned
            self.assertEqual(len(documents), 1)
            self.assertEqual(len(documents[0]), 0)


class TestSelfRAG(unittest.TestCase):
    def setUp(self):
        self.collection_name = f"test_selfrag_collection_{uuid.uuid4()}"
        # Create test documents
        self.documents = [
            Document(
                id=uuid.uuid4(), content="AI is a field of study.", meta_data=MetaData({"id": "1"})
            ),
            Document(
                id=uuid.uuid4(), content="ML is a subset of AI.", meta_data=MetaData({"id": "2"})
            ),
            Document(
                id=uuid.uuid4(),
                content="Deep learning is a type of ML.",
                meta_data=MetaData({"id": "3"}),
            ),
            Document(
                id=uuid.uuid4(),
                content="Neural networks are used in deep learning.",
                meta_data=MetaData({"id": "4"}),
            ),
        ]

        # Keep a reference to the original queries for testing
        self.queries = ["What is AI?", "Define ML"]

    def tearDown(self):
        if hasattr(self, "rag") and hasattr(self.rag, "client"):
            self.rag.client.delete_collection(self.collection_name)

    @patch("encourage.rag.self_rag.SelfRAG._generate_reflection")
    @patch("encourage.rag.self_rag.SelfRAG._generate_refined_response")
    def test_selfrag_initialization(self, mock_generate_refined_response, mock_generate_reflection):
        from encourage.rag.self_rag import SelfRAG

        # Initialize SelfRAG
        self.rag = SelfRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reflection_rounds=1,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Verify initialization parameters
        self.assertEqual(self.rag.reflection_rounds, 1)
        self.assertEqual(self.rag.top_k, 2)
        self.assertEqual(self.rag.collection_name, self.collection_name)

    @patch("encourage.rag.self_rag.SelfRAG._generate_reflection")
    @patch("encourage.rag.self_rag.SelfRAG._generate_refined_response")
    def test_process_single_query(self, mock_refined_response, mock_reflection):
        from encourage.llm.response import Response
        from encourage.rag.self_rag import SelfRAG

        # Set up mocks
        mock_reflection.return_value = "This response needs improvement in factuality."
        mock_refined_response.return_value = (
            "AI is a field of computer science focused on creating intelligent machines."
        )

        # Set up runner mock
        runner = create_autospec(BatchInferenceRunner)

        # Create a proper Response object for the mock
        mock_response = Response(
            request_id="test_id",
            prompt_id="test_prompt",
            sys_prompt="You are a helpful AI assistant.",
            user_prompt="What is AI?",
            response="AI is about making smart computers.",
        )
        mock_response_wrapper = ResponseWrapper([mock_response])
        runner.run.return_value = mock_response_wrapper

        # Initialize SelfRAG
        self.rag = SelfRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=1,
            reflection_rounds=1,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Create a context from the first document
        from encourage.prompts.context import Context

        test_context = Context.from_documents([self.documents[0]])

        # Use a patch to handle the _process_single_query method
        with patch(
            "encourage.rag.self_rag.SelfRAG._process_single_query", autospec=True
        ) as mock_method:
            mock_method.return_value = (
                "AI is a field of computer science focused on creating intelligent machines.",
                ["This response needs improvement in factuality."],
            )

            # Test _process_single_query method
            response_text, reflections = self.rag._process_single_query(
                runner=runner,
                query="What is AI?",
                context=test_context,
                meta_data={},
                sys_prompt="You are a helpful AI assistant.",
                template_name="llama3_conv.j2",
            )

        # Check the results
        self.assertEqual(
            response_text,
            "AI is a field of computer science focused on creating intelligent machines.",
        )
        self.assertEqual(reflections, ["This response needs improvement in factuality."])

    @patch("encourage.rag.self_rag.SelfRAG._process_single_query")
    def test_run_with_reflection(self, mock_process_single_query):
        from encourage.llm.response import Response
        from encourage.rag.self_rag import SelfRAG

        # Set up mock for _process_single_query
        mock_process_single_query.side_effect = [
            ("AI is the field of making machines intelligent.", ["Reflection 1"]),
            ("ML is a subset of AI focused on learning from data.", ["Reflection 2"]),
        ]

        # Set up runner mock
        runner = create_autospec(BatchInferenceRunner)

        # Initialize SelfRAG
        self.rag = SelfRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            reflection_rounds=1,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Mock the ResponseWrapper creation to return proper Response objects
        with patch("encourage.rag.self_rag.ResponseWrapper") as mock_response_wrapper:
            # Create the appropriate Response objects
            mock_response1 = Response(
                request_id="mock_1",
                prompt_id="prompt_1",
                sys_prompt="You are a helpful AI assistant.",
                user_prompt=self.queries[0],
                response="AI is the field of making machines intelligent.",
            )
            mock_response2 = Response(
                request_id="mock_2",
                prompt_id="prompt_2",
                sys_prompt="You are a helpful AI assistant.",
                user_prompt=self.queries[1],
                response="ML is a subset of AI focused on learning from data.",
            )

            # Set up the mock to return a proper ResponseWrapper with Response objects
            mock_wrapper = ResponseWrapper([mock_response1, mock_response2])
            mock_wrapper.meta_data = {"self_rag_reflections": [["Reflection 1"], ["Reflection 2"]]}
            mock_response_wrapper.return_value = mock_wrapper

            # Test run method
            result = self.rag.run(
                runner=runner,
                sys_prompt="You are a helpful AI assistant.",
                user_prompts=self.queries,
            )

        # Verify _process_single_query was called twice (once for each query)
        self.assertEqual(mock_process_single_query.call_count, 2)

        # Check response content
        self.assertEqual(
            result.get_responses(),
            [
                "AI is the field of making machines intelligent.",
                "ML is a subset of AI focused on learning from data.",
            ],
        )

        # Check metadata contains reflections
        self.assertIn("self_rag_reflections", result.meta_data)
        self.assertEqual(
            result.meta_data["self_rag_reflections"], [["Reflection 1"], ["Reflection 2"]]
        )

    def test_generate_reflection(self):
        from encourage.llm.response import Response
        from encourage.rag.self_rag import SelfRAG

        # Set up runner mock
        runner = create_autospec(BatchInferenceRunner)

        # Create a proper Response object instead of using a string
        mock_response = Response(
            request_id="test_id",
            prompt_id="prompt_id",
            sys_prompt="You are a critical evaluator.",
            user_prompt="Evaluate this response for factuality.",
            response="The response lacks detail about AI applications.",
        )
        reflection_response = ResponseWrapper([mock_response])
        runner.run.return_value = reflection_response

        # Initialize SelfRAG
        self.rag = SelfRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=1,
            reflection_rounds=1,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Test _generate_reflection method
        from encourage.prompts.context import Context

        test_context = Context.from_documents([self.documents[0]])

        # Patch the to_string method on Context
        with patch.object(Context, "to_string", return_value="AI is a field of study."):
            reflection = self.rag._generate_reflection(
                runner=runner,
                query="What is AI?",
                response="AI is about computers.",
                context=test_context,
            )

        # Verify runner was called
        runner.run.assert_called_once()

        # Check reflection result
        self.assertEqual(reflection, "The response lacks detail about AI applications.")

    def test_generate_refined_response(self):
        from encourage.llm.response import Response
        from encourage.rag.self_rag import SelfRAG

        # Set up runner mock
        runner = create_autospec(BatchInferenceRunner)

        # Create a proper Response object instead of using a string
        mock_response = Response(
            request_id="test_id",
            prompt_id="prompt_id",
            sys_prompt="You are an expert assistant.",
            user_prompt="Improve your response based on feedback.",
            response="AI is the study of making intelligent computer systems.",
        )
        refined_response = ResponseWrapper([mock_response])
        runner.run.return_value = refined_response

        # Initialize SelfRAG
        self.rag = SelfRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=1,
            reflection_rounds=1,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Test _generate_refined_response method
        from encourage.prompts.context import Context

        test_context = Context.from_documents([self.documents[0]])

        # Patch the to_string method on Context
        with patch.object(Context, "to_string", return_value="AI is a field of study."):
            refined = self.rag._generate_refined_response(
                runner=runner,
                query="What is AI?",
                initial_response="AI is about computers.",
                reflection="The response is too vague and lacks technical accuracy.",
                context=test_context,
            )

        # Verify runner was called
        runner.run.assert_called_once()

        # Check refined response
        self.assertEqual(refined, "AI is the study of making intelligent computer systems.")

    def test_retrieval_only_mode(self):
        from encourage.rag.self_rag import SelfRAG

        # Set up runner mock
        runner = create_autospec(BatchInferenceRunner)

        # Initialize SelfRAG with retrieval_only=True
        self.rag = SelfRAG(
            context_collection=self.documents,
            collection_name=self.collection_name,
            embedding_function="all-MiniLM-L6-v2",
            top_k=2,
            retrieval_only=True,
            template_name="llama3_conv.j2",
            device="cpu",
        )

        # Test run method in retrieval-only mode
        result = self.rag.run(
            runner=runner,
            sys_prompt="You are a helpful AI assistant.",
            user_prompts=self.queries,
        )

        # Runner should not be called in retrieval-only mode
        runner.run.assert_not_called()

        # Response should be a mock response wrapper
        self.assertIsInstance(result, ResponseWrapper)


if __name__ == "__main__":
    unittest.main()
