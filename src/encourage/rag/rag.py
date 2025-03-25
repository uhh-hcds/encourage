"""Module containing various RAG method implementations as classes."""

import logging
from enum import Enum

from encourage.rag.known_context import KnownContext
from encourage.rag.naive import NaiveRAG
from encourage.rag.summarize import SummarizationRAG

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """Enum for supported RAG methods."""

    Naive = "Naive"
    KnownContext = "KnownContext"
    Summarization = "Summarization"

    def get_class(self) -> type:
        """Get the implementation class for this RAG method."""
        method_classes = {
            RAGMethod.Naive: NaiveRAG,
            RAGMethod.KnownContext: KnownContext,
            RAGMethod.Summarization: SummarizationRAG,
        }
        return method_classes[self]
