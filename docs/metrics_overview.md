# Metrics Overview

Encourage provides a comprehensive set of metrics to evaluate the quality and performance of LLM responses. This document provides an overview of all available metrics, their purpose, and how to use them.

## Metrics Registration System

Encourage uses a registration system for metrics that makes them easy to access and use. The registration system works as follows:

1. Each metric class is decorated with `@register_metric("MetricName")` which adds it to the central registry
2. Metrics can be accessed by name using `get_metric_from_registry("metric_name")`
3. This design allows for easy extension with custom metrics

Example of registering a custom metric:

```python
from encourage.metrics import Metric, MetricOutput, register_metric
from encourage.llm import ResponseWrapper

@register_metric("MyCustomMetric")
class MyCustomMetric(Metric):
    """My custom metric implementation."""
    
    def __init__(self):
        super().__init__(
            name="my-custom-metric",
            description="A custom metric for specific evaluation needs"
        )
    
    def __call__(self, responses: ResponseWrapper) -> MetricOutput:
        # Custom metric calculation logic
        # ...
        return MetricOutput(score=score, raw=raw_scores)
```

## Available Metrics

| Category | Metric | Requires LLM | Description | Use Case |
|----------|--------|--------------|-------------|----------|
| **Quality** | AnswerFaithfulness | Yes | Measures how factually accurate the response is to the provided context | Fact-checking and hallucination detection |
| | AnswerRelevance | Yes | Evaluates how relevant the response is to the question | Ensuring responses address the query |
| | NonAnswerCritic | Yes | Identifies when responses fail to properly answer the question | Detecting non-answers and unhelpful responses |
| **Similarity** | AnswerSimilarity | No | Measures semantic similarity between responses and reference answers | Comparing responses to ground truth |
| | BLEU | No | Bilingual Evaluation Understudy score for text similarity | Machine translation evaluation |
| | GLEU | No | Google's variant of BLEU | Text generation evaluation |
| | ROUGE | No | Recall-Oriented Understudy for Gisting Evaluation | Summarization evaluation |
| | BERTScore | No | Measures similarity using BERT embeddings | Semantic similarity evaluation |
| | F1 | No | Harmonic mean of precision and recall | Information retrieval evaluation |
| | ExactMatch | No | Measures exact matches between responses and references | QA accuracy assessment |
| **Context** | ContextPrecision | Yes | Evaluates if retrieved context contains relevant information | RAG retrieval quality assessment |
| | ContextRecall | Yes | Checks if all needed information for an answer is in the context | RAG completeness assessment |
| | ContextLength | No | Counts the average length of context documents | Context size monitoring |
| **Retrieval** | MeanReciprocalRank | No | Measures rank position of first relevant document | Search quality evaluation |
| | RecallAtK | No | Percentage of relevant documents in top K results | Retrieval completeness assessment |
| | HitRateAtK | No | Whether any relevant document is in top K results | Retrieval success measurement |
| **Statistics** | GeneratedAnswerLength | No | Counts the average length of generated responses | Output length monitoring |
| | ReferenceAnswerLength | No | Counts the average length of reference answers | Reference length monitoring |
| | NumberMatch | No | Checks numerical accuracy in responses | Financial/quantitative accuracy |

## Using Metrics

### Basic Usage

```python
from encourage.metrics import AnswerRelevance, BLEU
from encourage.llm import BatchInferenceRunner

# For metrics that don't require an LLM runner
bleu_metric = BLEU()
bleu_result = bleu_metric(responses)
print(f"BLEU Score: {bleu_result.score}")

# For metrics that require an LLM runner
runner = BatchInferenceRunner(...)
relevance_metric = AnswerRelevance(runner=runner)
relevance_result = relevance_metric(responses)
print(f"Relevance Score: {relevance_result.score}")
```

### Using the Registry

```python
from encourage.metrics import get_metric_from_registry

# Get a metric from the registry (for metrics that don't require an LLM runner)
bleu = get_metric_from_registry("bleu")
result = bleu(responses)

# For metrics requiring an LLM runner
relevance = get_metric_from_registry("answerrelevance", runner=llm_runner)
result = relevance(responses)
```

## Metric Results

All metrics return a `MetricOutput` object with the following attributes:

- `score`: The aggregated score (typically between 0 and 1)
- `raw`: List of individual scores for each response
- `misc`: Additional metric-specific information

```python
result = metric(responses)
print(f"Overall score: {result.score}")
print(f"Individual scores: {result.raw}")
print(f"Additional info: {result.misc}")
```

## Advanced Usage: Custom Metrics

You can create custom metrics by extending the `Metric` base class and implementing the required methods:

1. Inherit from `Metric`
2. Implement `__call__` method
3. Register with `@register_metric`

See the registration example above for implementation details.

## LLM-Based vs. Traditional Metrics

Encourage provides two types of metrics:

1. **Traditional metrics**: Statistical measures like BLEU, ROUGE, etc., that don't require an LLM to compute
2. **LLM-based metrics**: Advanced evaluation using LLMs to assess aspects like faithfulness and relevance

LLM-based metrics require an LLM runner to be passed during initialization and generally provide more nuanced evaluation but may be more computationally expensive.