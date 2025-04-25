# RAG Methods

Encourage provides several RAG (Retrieval-Augmented Generation) methods to enhance LLM responses with relevant context. Each method has unique characteristics and is suitable for different use cases.

## Available RAG Methods

### KnownContext

`KnownContext` allows you to provide predefined context documents that you know are relevant to your queries. It doesn't perform dynamic retrieval but instead uses exactly the context you specify.

**Use case**: When you have pre-selected relevant information that you want to use as context for all queries.

### Base

`BaseRAG` is the foundational implementation of RAG. It provides basic vector-based retrieval of context documents based on query similarity. It integrates retrieval-based context generation with language model inference to provide answers from a given dataset.

**Use case**: General purpose RAG when you need a simple, reliable approach for context retrieval.

### Hyde (Hypothetical Document Embeddings)

`HydeRAG` generates hypothetical answers to queries and uses those answers as search vectors for retrieving relevant documents. This can improve retrieval quality compared to using the original query directly.

**Use case**: When you need better semantic search results, especially for complex or ambiguous queries.

**Reference**: [Hypothetical Document Embeddings (HYDE)](https://arxiv.org/abs/2212.10496)

### Summarization

`SummarizationRAG` summarizes the context documents before adding them to the vector store. This approach can help reduce the context size while preserving the most important information.

**Use case**: When dealing with lengthy documents or when you need to fit more context within token limits.

### SummarizationContextRAG

`SummarizationContextRAG` is similar to `SummarizationRAG`, but it preserves the original context during retrieval. It uses summarized documents for retrieval but returns the original full documents when providing context to the LLM.

**Use case**: When you want the efficiency of searching through summaries but need the completeness of original documents in your responses.

### Reranker

`RerankerRAG` uses a two-stage retrieval approach. First, it retrieves a larger set of potentially relevant documents using vector search, then uses a reranking model to score and filter the results based on relevance to the query.

**Use case**: When precision is critical and you want to improve the relevance of retrieved documents.

### HybridBM25

`HybridBM25RAG` combines dense vector retrieval with sparse BM25 (keyword-based) retrieval for a hybrid search approach. This can capture both semantic similarity and keyword relevance.

**Use case**: When you need to balance semantic understanding with keyword matching, particularly useful for technical or specialized domains.

### HydeReranker

`HydeRerankerRAG` combines the strengths of both HYDE and Reranker approaches. It first generates hypothetical answers using an LLM, uses these as search vectors for initial retrieval, and then applies cross-encoder reranking to further improve relevance.

**Use case**: When dealing with complex queries that benefit from both hypothetical document generation and precision reranking, especially in domains where both semantic understanding and precise ranking matter.

## Using RAG Methods

You can select and instantiate a RAG method using the `RAGMethod` enum:

```python
from encourage.rag import RAGMethod

# Get the class for a specific RAG method
rag_class = RAGMethod.Hyde.get_class()

# Instantiate the RAG method
rag_instance = rag_class(
    context_collection=documents,
    collection_name="my_collection",
    embedding_function="sentence-transformers/all-mpnet-base-v2",
    top_k=5
)

# Use the RAG instance for retrieval and generation
response = rag_instance.run(
    runner=inference_runner,
    sys_prompt="Answer based on the provided context",
    user_prompts=["What is RAG?"]
)
```

Refer to the API documentation of each method for specific parameters and customization options.