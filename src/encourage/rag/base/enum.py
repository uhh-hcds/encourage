"""Module containing various RAG method implementations as classes."""

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class RAGMethod(Enum):
    """Enum for supported RAG methods."""

    Hyde = "Hyde"
    Base = "Base"
    KnownContext = "KnownContext"
    NoContext = "NoContext"
    Summarization = "Summarization"
    SummarizationContextRAG = "SummarizationContextRAG"
    Reranker = "Reranker"
    HydeReranker = "HydeReranker"
    HybridBM25 = "HybridBM25"
    BM25 = "BM25"
    SelfRAG = "SelfRAG"
