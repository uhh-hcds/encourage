"""Module containing various RAG method implementations as classes."""

import logging
from typing import Any, override

from chromadb.utils import embedding_functions
from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.config import BaseRAGConfig
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base.interface import RAGMethodInterface
from encourage.utils.llm_mock import create_mock_response_wrapper
from encourage.vector_store import ChromaClient, VectorStore

logger = logging.getLogger(__name__)


@RAGFactory.register(RAGMethod.Base, BaseRAGConfig)
class BaseRAG(RAGMethodInterface):
    """Base implementation of RAG.

    BaseRAG is a foundational implementation of a Retrieval-Augmented Generation (RAG) method.
    It integrates retrieval-based context generation with language model inference to provide
    answers based on a given dataset and context.

    Attributes:
        template_name (str): The name of the prompt template to use.
        collection_name (str): The name of the collection in the vector database.
        top_k (int): The number of top results to retrieve from the database.
        embedding_function (Any): The embedding function used for vectorization.
        where (dict[str, str] | None): Optional filtering conditions for retrieval.
        retrieval_only (bool): If True, skips LLM inference and only retrieves contexts.
        runner (BatchInferenceRunner | None): The inference runner for batch processing.
        additional_prompt (str): Additional prompt text to append to the generated prompts.
        client (VectorStore): The vector database client for context retrieval.

    Methods:
        init_db(context_collection) -> VectorStore:
            Initializes the vector database with the provided context collection.
        retrieve_contexts(query_list, **kwargs) -> list[list[Document]]:
            Retrieves relevant contexts from the database based on the provided queries.
        run(runner, sys_prompt, user_prompts=[], meta_datas=[], retrieval_queries=[])
        -> ResponseWrapper:
            Executes the RAG pipeline, including context retrieval and LLM inference,
            and returns the generated responses.

    """

    def __init__(self, config: BaseRAGConfig, **kwargs: Any) -> None:
        """Initialize BaseRAG from BaseRAGConfig."""
        self.collection_name = config.collection_name
        self.top_k = config.top_k
        self.embedding_function = self.get_embedding_model(
            config.embedding_function, device=config.device
        )
        self.retrieval_only = config.retrieval_only
        self.runner = config.runner
        self.additional_prompt = config.additional_prompt
        self.where = config.where
        self.template_name = config.template_name
        self.context_collection = self.filter_duplicates(config.context_collection)
        self.batch_size_insert = config.batch_size_insert
        self.batch_size_query = config.batch_size_query
        self.client = self.init_db()

    def get_embedding_model(self, name: str, device: str = "cuda") -> Any:
        """Return embedding model based on name."""
        # TODO: Add support for other embedding models in the future.
        # key_map = {
        #     "Google": (
        #         "GOOGLE_API_KEY",
        #         embedding_functions.GoogleGenerativeAiEmbeddingFunction,
        #         "models/text-embedding-004",
        #     ),
        #     "OpenAI": (
        #         "OPENAI_API_KEY",
        #         embedding_functions.OpenAIEmbeddingFunction,
        #         "text-embedding-3-large",
        #     ),
        # }

        # if name in key_map:
        #     env_var, fn, model = key_map[name]
        #     key = os.getenv(env_var)
        #     if not key:
        #         raise ValueError(f"{env_var} not set.")
        #     self.check_quota = True
        #     return fn(api_key=key, model_name=model)
        # self.check_quota = False
        return embedding_functions.SentenceTransformerEmbeddingFunction(name, device=device)

    def filter_duplicates(self, context_collection: list[Document]) -> list[Document]:
        """Filter out duplicate documents from the context collection."""
        unique_documents = {}
        for document in context_collection:
            if document.id not in unique_documents:
                unique_documents[document.id] = document
        return list(unique_documents.values())

    def init_db(
        self,
    ) -> VectorStore:
        """Initialize the database with the contexts."""
        chroma_client = ChromaClient()
        logger.info(f"Creating collection {self.collection_name}.")
        chroma_client.create_collection(
            self.collection_name, overwrite=True, embedding_function=self.embedding_function
        )
        logger.info(f"Inserting {len(self.context_collection)} documents into the database.")
        chroma_client.insert_documents(
            collection_name=self.collection_name,
            documents=self.context_collection,
            embedding_function=self.embedding_function,
            batch_size=self.batch_size_insert,
        )
        logger.info("Database initialized.")
        return chroma_client

    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve relevant contexts from the database."""
        return self.client.query(
            collection_name=self.collection_name,
            query=query_list,
            top_k=self.top_k,
            embedding_function=self.embedding_function,
            where=self.where if self.where else None,
            batch_size=self.batch_size_query,
        )

    @override
    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        meta_datas: list[MetaData] = [],
        retrieval_queries: list[str] = [],
        response_format: type[BaseModel] | str | None = None,
    ) -> ResponseWrapper:
        """Execute the complete RAG pipeline and return responses."""
        # Generate queries and retrieve contexts
        if retrieval_queries:
            logger.info(f"Generating {len(retrieval_queries)} retrieval queries.")
            retrieved_documents = self.retrieve_contexts(retrieval_queries)
            self.contexts = [Context.from_documents(documents) for documents in retrieved_documents]
        else:
            logger.info("No context retrieval queries provided. Using no context.")
            self.contexts = []

        template_name = self.template_name
        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=meta_datas,
            template_name=template_name,
        )

        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            return create_mock_response_wrapper(prompt_collection)
        else:
            # Run inference with the LLM
            return runner.run(prompt_collection, response_format=response_format)
