"""Module containing various RAG method implementations as classes."""

import logging
from typing import Any, override

from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from pydantic import BaseModel

from encourage.llm import BatchInferenceRunner, ResponseWrapper
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.prompts.meta_data import MetaData
from encourage.rag.base.rag_interface import RAGMethodInterface
from encourage.utils.llm_mock import create_mock_response_wrapper
from encourage.vector_store import ChromaClient, VectorStore

logger = logging.getLogger(__name__)


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
        __init__(context_collection, template_name, collection_name, embedding_function,
                 top_k, device="cuda", where=None, retrieval_only=False, runner=None,
                 additional_prompt="", **kwargs):
            Initializes the BaseRAG instance with the provided configuration.
        init_db(context_collection) -> VectorStore:
            Initializes the vector database with the provided context collection.
        retrieve_contexts(query_list, **kwargs) -> list[list[Document]]:
            Retrieves relevant contexts from the database based on the provided queries.
        run(runner, sys_prompt, user_prompts=[], meta_datas=[], retrieval_queries=[])
        -> ResponseWrapper:
            Executes the RAG pipeline, including context retrieval and LLM inference,
            and returns the generated responses.

    """

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: Any,
        top_k: int,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        where: dict[str, str] | None = None,
        device: str = "cuda",
        template_name: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize RAG method with configuration."""
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            embedding_function, device=device
        )
        self.retrieval_only = retrieval_only
        self.runner = runner
        self.additional_prompt = additional_prompt
        self.where = where
        self.template_name = template_name
        self.context_collection = self.filter_duplicates(context_collection)
        self.client = self.init_db()

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
