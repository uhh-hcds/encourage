"""Module implementing a reranker that uses a cross-encoder model to improve retrieval quality."""

import logging
from abc import abstractmethod
from enum import Enum

from encourage.prompts.context import Document

logger = logging.getLogger(__name__)


class RerankerModel(str, Enum):
    """Predefined reranker model names."""

    MS_MARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    JINA_V3 = "jinaai/jina-reranker-v3"


class Reranker:
    """A reranker that uses a cross-encoder model to improve retrieval quality.

    This class provides common functionality for reranking documents based on relevance
    scoring using a cross-encoder model.

    Attributes:
        reranker_model (str): Name or path of the cross-encoder reranker model
        rerank_ratio (float): Multiplier for how many documents to initially retrieve compared
        to top_k
        reranker (CrossEncoder): The cross-encoder model used for reranking
        device (str): Device to use for the reranker model

    """

    def __init__(
        self,
        rerank_ratio: float = 1.0,
        device: str = "cuda",
    ) -> None:
        """Initialize a Reranker with configuration.

        Args:
            reranker_model: Name or path of the cross-encoder reranker model
            rerank_ratio: How many times more documents to retrieve initially for reranking
                (e.g., 1.0 means retrieve 1*top_k documents initially)
            device: Device to use for reranker model

        """
        self.rerank_ratio = rerank_ratio
        self.device = device

    def calculate_initial_top_k(self, top_k: int) -> int:
        """Calculate how many documents to retrieve initially for reranking."""
        return max(int(top_k * self.rerank_ratio), top_k)

    @abstractmethod
    def rerank_documents(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Rerank documents based on relevance to the query."""
        pass
