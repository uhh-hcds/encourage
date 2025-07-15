"""Base configuration for RAG (Retrieval-Augmented Generation) components."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from encourage.llm.inference_runner import InferenceRunner


class BaseRAGConfig(BaseModel):
    """Configuration for BaseRAG method.

    Attributes:
        context_collection: List of Documents or context items.
        collection_name: Name of the vector collection.
        embedding_function: Embedding function or reference.
        top_k: Number of top results to retrieve.
        retrieval_only: If True, skip LLM inference.
        runner: Optional batch inference runner.
        additional_prompt: Extra prompt text appended to queries.
        where: Optional filtering dictionary for retrieval.
        device: Device string, e.g., 'cuda' or 'cpu'.
        template_name: Name of prompt template to use.
        batch_size_insert: Batch size for inserting contexts.
        batch_size_query: Batch size for querying contexts.

    """

    context_collection: list[Any]
    collection_name: str
    embedding_function: Any
    top_k: int
    retrieval_only: bool = False
    runner: Optional[InferenceRunner] = None
    additional_prompt: str = ""
    where: Optional[dict[str, str]] = None
    device: str = "cuda"
    template_name: str = ""
    batch_size_insert: int = 2000
    batch_size_query: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HybridBM25RAGConfig(BaseRAGConfig):
    """Configuration for HybridBM25RAG method, extending BaseRAGConfig.

    Adds:
        alpha: Weight for dense retrieval scores (0 to 1).
        beta: Weight for sparse retrieval scores (0 to 1).
    """

    alpha: float = 0.5
    beta: float = 0.5


class RerankerRAGConfig(BaseRAGConfig):
    """Configuration for RerankerRAG method extending BaseRAGConfig.

    Adds reranker_model and rerank_ratio parameters.
    """

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_ratio: float = 3.0


class HydeRAGConfig(BaseRAGConfig):
    """Configuration for HydeRAG method, extending BaseRAGConfig.

    HYDE generates hypothetical answers to queries and uses those answers as search vectors.
    """

    pass


class NoContextConfig(BaseRAGConfig):
    """Configuration class for NoContext RAG method,inheriting from BaseRAGConfig."""

    pass


class KnownContextConfig(BaseRAGConfig):
    """Configuration class for KnownContext RAG method, inheriting from BaseRAGConfig."""

    pass


class SummarizationRAGConfig(BaseRAGConfig):
    """Configuration for SummarizationRAG method extending BaseRAGConfig.

    No additional parameters beyond BaseRAGConfig.
    """

    pass


class SummarizationContextRAGConfig(SummarizationRAGConfig):
    """Configuration for SummarizationContextRAG method."""

    pass


class SelfRAGConfig(BaseRAGConfig):
    """Configuration for SelfRAG method, extending BaseRAGConfig."""

    relevance_sys_prompt: str = (
        "You are a critical evaluator employed by a RAG system. Analyze the retrieved context for "
        "factuality, relevance, coherence, and information completeness."
        "Identify any hallucinations or missing important information from the retrieved context."
        "Relevant or Irrelevant? Answer with one of those two words only."
    )
    support_sys_prompt: str = (
        "Given the response and the context, determine if the response is supported by the context."
        "Label as Fully supported, Partially supported or"
        "No support – output exactly one of those only."
    )
    utility_sys_prompt: str = (
        "Given the query and the response, rate the utility of the response from 1 to 5. "
        "Output the number only."
    )
