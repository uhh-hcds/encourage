"""Module containing various RAG method implementations as classes."""

import logging
from enum import Enum

from encourage.rag.base.rag_interface import RAGMethodInterface
from encourage.rag.hyde import HydeRAG
from encourage.rag.known_context import KnownContext
from encourage.rag.naive import BaseRAG
from encourage.rag.summarize import SummarizationContextRAG, SummarizationRAG

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """Enum for supported RAG methods."""

    Hyde = "Hyde"
    Base = "Base"
    KnownContext = "KnownContext"
    Summarization = "Summarization"
    SummarizationContextRAG = "SummarizationContextRAG"

    def get_class(self) -> type[RAGMethodInterface]:
        """Get the implementation class for this RAG method."""
        method_classes = {
            RAGMethod.Hyde: HydeRAG,
            RAGMethod.Naive: BaseRAG,
            RAGMethod.KnownContext: KnownContext,
            RAGMethod.Summarization: SummarizationRAG,
            RAGMethod.SummarizationContextRAG: SummarizationContextRAG,
        }
        return method_classes[self]  # type: ignore
