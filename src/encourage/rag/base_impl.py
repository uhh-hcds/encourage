"""Module containing various RAG method implementations as classes."""

import logging
import uuid
from typing import Any

import pandas as pd
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)

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
    The class also modifies the `qa_dataset` by adding additional columns such as `context_id`.

    Attributes:
        qa_dataset (pd.DataFrame): The QA dataset with questions, answers, and contexts.
        template_name (str): The name of the prompt template to use.
        collection_name (str): The name of the collection in the vector database.
        top_k (int): The number of top results to retrieve from the database.
        embedding_function (Any): The embedding function used for vectorization.
        meta_data_keys (list[str]): Keys for metadata extraction from the dataset.
        context_key (str): The column name for context in the dataset. Defaults to "context".
        question_key (str): The column name for questions in the dataset. Defaults to "question".
        answer_key (str): The column name for answers in the dataset. Defaults to "program_answer".
        device (str): The device to use for computation (e.g., "cuda" or "cpu"). Defaults to "cuda".
        where (dict[str, str] | None): Optional filtering conditions for retrieval.
        retrieval_only (bool): If True, skips LLM inference and only retrieves contexts.
        runner (BatchInferenceRunner | None): The inference runner for batch processing.
        additional_prompt (str): Additional prompt text to append to the generated prompts.
        metadata (list[MetaData]): Metadata extracted from the dataset.
        user_prompts (pd.Series): User prompts extracted from the dataset.
        client (VectorStore): The vector database client for context retrieval.

    Methods:
        __init__(qa_dataset, template_name, collection_name, top_k, embedding_function,
                 meta_data_keys, context_key, question_key, answer_key, device, where,
                 retrieval_only, runner, additional_prompt, **kwargs):
            Initializes the BaseRAG instance with the provided configuration.
        create_context_id(qa_dataset, context_key="context") -> pd.DataFrame:
            Adds a unique `context_id` column to the dataset based on the context values.
        create_metadata(answer_key="program_answer") -> list[MetaData]:
            Generates metadata objects from the dataset for use in context retrieval.
        prepare_contexts_for_db(meta_data_keys) -> Context:
            Prepares the contexts and metadata for insertion into the vector database.
        init_db(context_collection) -> VectorStore:
            Initializes the vector database with the provided context collection.
        retrieve_contexts(query_list, **kwargs) -> list[Context]:
            Retrieves relevant contexts from the database based on the provided queries.
        run(runner, sys_prompt, user_prompts=[], retrieval_instruction=[]) -> ResponseWrapper:
            Executes the RAG pipeline, including context retrieval and LLM inference,
            and returns the generated responses.

    """

    def __init__(
        self,
        qa_dataset: pd.DataFrame,
        template_name: str,
        collection_name: str,
        top_k: int,
        embedding_function: Any,
        meta_data_keys: list[str],
        context_key: str = "context",
        question_key: str = "question",
        answer_key: str = "program_answer",
        device: str = "cuda",
        where: dict[str, str] | None = None,
        retrieval_only: bool = False,
        runner: BatchInferenceRunner | None = None,
        additional_prompt: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize RAG method with configuration."""
        self.qa_dataset = self.create_context_id(qa_dataset, context_key)
        self.template_name = template_name
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            embedding_function, device=device
        )
        self.context_key = context_key
        self.where = where
        self.retrieval_only = retrieval_only
        self.runner = runner
        self.additional_prompt = additional_prompt
        self.metadata = self.create_metadata(answer_key)
        self.user_prompts = qa_dataset[question_key]
        context_collection = self.prepare_contexts_for_db(meta_data_keys)
        self.client = self.init_db(context_collection)

    def create_context_id(
        self, qa_dataset: pd.DataFrame, context_key: str = "context"
    ) -> pd.DataFrame:
        """Create context id from dataset."""
        unique_values = list(qa_dataset[context_key].unique())
        uuid_mapping = {val: str(uuid.uuid4()) for val in unique_values}
        qa_dataset["context_id"] = qa_dataset[context_key].map(uuid_mapping)
        return qa_dataset

    def create_metadata(self, answer_key: str = "program_answer") -> list[MetaData]:
        """Create metadata from dataset."""
        meta_datas = []
        for i in range(len(self.qa_dataset)):
            meta_data = MetaData(
                {
                    "reference_answer": self.qa_dataset[answer_key][i],
                    "id": self.qa_dataset["id"][i],
                    "reference_document": Document(
                        id=uuid.UUID(self.qa_dataset["context_id"][i]),
                        content=self.qa_dataset["context"][i],
                    ),
                }
            )
            meta_datas.append(meta_data)
        return meta_datas

    def prepare_contexts_for_db(self, meta_data_keys: list[str]) -> Context:
        """Prepare contexts for the QA dataset."""
        df = self.qa_dataset[[*meta_data_keys, self.context_key, "context_id"]]
        df = df.drop_duplicates(subset=[self.context_key])
        meta_datas = [
            MetaData(tags={key: row[key] for key in meta_data_keys}) for _, row in df.iterrows()
        ]
        return Context.from_documents(
            df[self.context_key].tolist(), meta_datas, df["context_id"].tolist()
        )

    def init_db(self, context_collection: Context) -> VectorStore:
        """Initialize the database with the contexts."""
        chroma_client = ChromaClient()
        print(f"Creating collection {self.collection_name}.")
        chroma_client.create_collection(
            self.collection_name, overwrite=True, embedding_function=self.embedding_function
        )
        print(f"Inserting {len(context_collection.documents)} documents into the database.")
        chroma_client.insert_documents(
            self.collection_name,
            vector_store_document=context_collection,  # type: ignore
            embedding_function=self.embedding_function,
        )
        print("Database initialized.")
        return chroma_client

    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[Context]:
        """Retrieve relevant contexts from the database."""
        results = self.client.query(
            collection_name=self.collection_name,
            query=query_list,
            top_k=self.top_k,
            embedding_function=self.embedding_function,
            where=self.where if self.where else None,
        )
        return [Context.from_documents(document_list) for document_list in results]

    def run(
        self,
        runner: BatchInferenceRunner,
        sys_prompt: str,
        user_prompts: list[str] = [],
        retrieval_instruction: list[str] = [],
    ) -> ResponseWrapper:
        """Execute the complete RAG pipeline and return responses."""
        # Generate queries and retrieve contexts
        if retrieval_instruction:
            logger.info(f"Generating {len(retrieval_instruction)} retrieval queries.")
            self.contexts = self.retrieve_contexts(retrieval_instruction)
        else:
            logger.info("No context retrieval queries provided. Using no context.")
            self.contexts = []

        user_prompts = user_prompts if user_prompts else self.user_prompts
        # Create prompt collection
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=self.metadata,
            template_name=self.template_name,
        )

        if self.retrieval_only:
            logger.info("Retrieval-only mode: Skipping LLM inference.")
            return create_mock_response_wrapper(prompt_collection)
        else:
            # Run inference with the LLM
            return runner.run(prompt_collection)
