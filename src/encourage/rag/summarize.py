"""Module containing various RAG method implementations as classes."""

import logging
from typing import Any, override

from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context, Document
from encourage.rag.base.config import (
    SummarizationContextRAGConfig,
    SummarizationRAGConfig,
)
from encourage.rag.base.enum import RAGMethod
from encourage.rag.base.factory import RAGFactory
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)


@RAGFactory.register(RAGMethod.Summarization, SummarizationRAGConfig)
class SummarizationRAG(BaseRAG):
    """Implementation of RAG for summarization."""

    def __init__(self, config: SummarizationRAGConfig) -> None:
        """Initialize KnownContext with provided configuration.

        Args:
            config (KnownContextConfig): Configuration object with parameters.
            **kwargs: Additional arguments passed to BaseRAG.

        """
        """Initialize RAG method with configuration."""
        context_collection = super().filter_duplicates(config.context_collection)
        if not isinstance(config.runner, BatchInferenceRunner):
            raise TypeError("config.runner must be an instance of BatchInferenceRunner")
        summaries = self.create_summaries(
            config.runner, config.additional_prompt, context_collection, config.template_name
        )
        super().__init__(config.model_copy(update={"context_collection": summaries}))

    def create_summaries(
        self,
        runner: BatchInferenceRunner,
        additional_prompt: str,
        context_collection: list[Document],
        template_name: str,
    ) -> list[Document]:
        """Create summary from the QA dataset."""
        user_prompts = [context.content for context in context_collection]
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=additional_prompt,
            user_prompts=user_prompts,
            template_name=template_name,
        )
        responses = runner.run(prompt_collection)
        sum_mapping = {
            context.content: response.response
            for response, context in zip(responses, context_collection)
        }
        summarized_documents = []
        for doc in context_collection:
            summarized_documents.append(
                Document(
                    content=sum_mapping[doc.content],
                    meta_data=doc.meta_data,
                    id=doc.id,
                )
            )
        return summarized_documents


@RAGFactory.register(RAGMethod.SummarizationContextRAG, SummarizationContextRAGConfig)
class SummarizationContextRAG(SummarizationRAG):
    """Implementation of RAG with context preserving summarization."""

    def __init__(self, config: SummarizationContextRAGConfig):
        """Initialize RAG method with configuration."""
        self.original_context = self.transform_contexts_to_documents(config.context_collection)
        super().__init__(config)

    @override
    def retrieve_contexts(
        self,
        query_list: list[str],
        **kwargs: Any,
    ) -> list[list[Document]]:
        """Retrieve contexts from the database with context preservation.

        This method overrides the parent implementation to ensure that the
        original context is preserved in the returned documents.
        """
        # Get contexts using the parent implementation
        document_lists = super().retrieve_contexts(query_list, **kwargs)

        for document_list in document_lists:
            for doc in document_list:
                original = next(
                    (
                        orig_doc
                        for orig_doc in self.original_context
                        if str(orig_doc.id) == str(doc.id)
                    ),
                    None,
                )
                if original:
                    doc.content = original.content
                    doc.meta_data = original.meta_data

        return document_lists

    def transform_contexts_to_documents(self, contexts: list[Context]) -> list[Document]:
        """Transform Context objects to Document objects."""
        documents = []
        for context in contexts:
            for doc in context.documents:
                documents.append(doc)
        return documents
