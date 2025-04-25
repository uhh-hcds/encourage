"""Module containing various RAG method implementations as classes."""

import logging
from enum import Enum

from encourage.rag.base.rag_interface import RAGMethodInterface
from encourage.rag.base_impl import BaseRAG
from encourage.rag.hybrid_bm25 import HybridBM25RAG
from encourage.rag.hyde import HydeRAG
from encourage.rag.hyde_reranker import HydeRerankerRAG
from encourage.rag.known_context import KnownContext
from encourage.rag.reranker import RerankerRAG
from encourage.rag.summarize import SummarizationContextRAG, SummarizationRAG

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """Enum for supported RAG methods."""

    Hyde = "Hyde"
    Base = "Base"
    KnownContext = "KnownContext"
    Summarization = "Summarization"
    SummarizationContextRAG = "SummarizationContextRAG"
    Reranker = "Reranker"
    HydeReranker = "HydeReranker"
    HybridBM25 = "HybridBM25"

    def get_class(self) -> type[RAGMethodInterface]:
        """Get the implementation class for this RAG method."""
        method_classes = {
            RAGMethod.Hyde: HydeRAG,
            RAGMethod.Base: BaseRAG,
            RAGMethod.KnownContext: KnownContext,
            RAGMethod.Summarization: SummarizationRAG,
            RAGMethod.SummarizationContextRAG: SummarizationContextRAG,
            RAGMethod.Reranker: RerankerRAG,
            RAGMethod.HydeReranker: HydeRerankerRAG,
            RAGMethod.HybridBM25: HybridBM25RAG,
        }
        return method_classes[self]  # type: ignore
