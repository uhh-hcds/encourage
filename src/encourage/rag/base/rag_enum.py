"""Module containing various RAG method implementations as classes."""

import logging
from enum import Enum

from encourage.rag.base.rag_interface import RAGMethodInterface
from encourage.rag.known_context import KnownContext
from encourage.rag.naive import NaiveRAG
from encourage.rag.retrieval_only import RetrievalOnlyRAG
from encourage.rag.summarize import ContextPreservingSummarizationRAG, SummarizationRAG

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """Enum for supported RAG methods."""

    Naive = "Naive"
    KnownContext = "KnownContext"
    Summarization = "Summarization"
    ContextPreservingSummarization = "ContextPreservingSummarization"
    RetrievalOnly = "RetrievalOnly"

    def get_class(self) -> type[RAGMethodInterface]:
        """Get the implementation class for this RAG method."""
        method_classes = {
            RAGMethod.Naive: NaiveRAG,
            RAGMethod.KnownContext: KnownContext,
            RAGMethod.Summarization: SummarizationRAG,
            RAGMethod.ContextPreservingSummarization: ContextPreservingSummarizationRAG,
            RAGMethod.RetrievalOnly: RetrievalOnlyRAG,
        }
        return method_classes[self]
