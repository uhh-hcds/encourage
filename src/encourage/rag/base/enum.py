"""Module containing various RAG method implementations as classes."""

import logging
import importlib

# Import Enum explicitly from the standard library enum module to avoid
# any potential self-import if this module's name shadows stdlib `enum`.
_stdlib_enum = importlib.import_module("enum")
Enum = _stdlib_enum.Enum

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
