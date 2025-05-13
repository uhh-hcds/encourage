"""Module containing various RAG method implementations as classes."""

import logging
from typing import Any, override

from encourage.llm import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Document
from encourage.rag.base_impl import BaseRAG

logger = logging.getLogger(__name__)


class SummarizationRAG(BaseRAG):
    """Implementation of RAG for summarization."""

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: str,
        top_k: int,
        runner: BatchInferenceRunner,
        retrieval_only: bool = False,
        where: dict[str, str] | None = None,
        additional_prompt: str = "",
        template_name: str = "",
        **kwargs: Any,
    ):
        """Initialize RAG method with configuration."""
        self.template_name = template_name
        summaries = self.create_summaries(runner, additional_prompt, context_collection)
        super().__init__(
            context_collection=summaries,
            template_name=template_name,
            collection_name=collection_name,
            top_k=top_k,
            embedding_function=embedding_function,
            where=where,
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
        )

    def create_summaries(
        self,
        runner: BatchInferenceRunner,
        additional_prompt: str,
        context_collection: list[Document],
    ) -> list[Document]:
        """Create summary from the QA dataset."""
        user_prompts = [context.content for context in context_collection]
        prompt_collection = PromptCollection.create_prompts(
            sys_prompts=additional_prompt,
            user_prompts=user_prompts,
            template_name=self.template_name,
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


class SummarizationContextRAG(SummarizationRAG):
    """Implementation of RAG with context preserving summarization."""

    def __init__(
        self,
        context_collection: list[Document],
        collection_name: str,
        embedding_function: str,
        top_k: int,
        runner: BatchInferenceRunner,
        where: dict[str, str] | None = None,
        retrieval_only: bool = False,
        additional_prompt: str = "",
        template_name: str = "",
    ):
        """Initialize RAG method with configuration."""
        self.original_context = context_collection
        super().__init__(
            context_collection=context_collection,
            template_name=template_name,
            collection_name=collection_name,
            top_k=top_k,
            embedding_function=embedding_function,
            where=where,
            retrieval_only=retrieval_only,
            runner=runner,
            additional_prompt=additional_prompt,
        )

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
