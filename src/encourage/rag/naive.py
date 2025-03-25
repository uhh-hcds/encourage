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
from encourage.vector_store import ChromaClient

logger = logging.getLogger(__name__)


class NaiveRAG:
    """Base implementation of RAG."""

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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize RAG method with configuration."""
        self.qa_dataset = self.create_context_id(qa_dataset, context_key)
        self.template_name = template_name
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            embedding_function, device="cuda"
        )
        self.context_key = context_key

        self.metadata = self.create_metadata(answer_key)
        self.user_prompts = qa_dataset[question_key]
        context_collection = self.prepare_contexts_for_db(meta_data_keys)
        self.client = self.init_db(context_collection)

    def create_context_id(self, qa_dataset: pd.DataFrame, context_key: str = "context") -> Any:
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

    def prepare_contexts_for_db(self, meta_datas_keys: list[str]) -> Context:
        """Prepare contexts for the QA dataset."""
        df = self.qa_dataset[[*meta_datas_keys, self.context_key, "context_id"]]
        df = df.drop_duplicates(subset=[self.context_key])
        meta_datas = [
            MetaData(tags={key: row[key] for key in meta_datas_keys}) for _, row in df.iterrows()
        ]
        return Context.from_documents(
            df[self.context_key].tolist(), meta_datas, df["context_id"].tolist()
        )

    def init_db(self, context_collection: Context) -> ChromaClient:
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

    def _get_contexts_from_db(
        self,
        query_list: list[str],
        meta_datas: list[MetaData] = [MetaData()],
    ) -> list[Context]:
        """Get the contexts from the database."""
        if meta_datas != [MetaData()]:
            raise NotImplementedError("Handling of meta_datas is not implemented yet.")
        results = self.client.query(
            self.collection_name, query_list, self.top_k, self.embedding_function
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
            self.contexts = self._get_contexts_from_db(retrieval_instruction)
        else:
            logger.info("No context retrieval queries provided. Using no context.")
            self.contexts = []
        user_prompts = user_prompts if user_prompts else self.user_prompts

        # Run inference
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=sys_prompt,
            user_prompts=user_prompts,
            contexts=self.contexts,
            meta_datas=self.metadata,
            template_name=self.template_name,
        )
        return runner.run(prompt_collection)
