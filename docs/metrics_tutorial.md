# Metrics Usage

## Importing the Metrics

To use the metrics, you first need to import them into your project. You can import them individually or as a group depending on your needs.

```python
from encourage.metrics import GeneratedAnswerLength, ReferenceAnswerLength, ContextLength, BLEU, GLEU, ROUGE, BERTScore, F1, ExactMatch, MeanReciprocalRank
```

## Calling the Metrics

Once the metrics are imported, you can call them by instantiating each metric and passing a ResponseWrapper object containing the responses you want to evaluate. For example:

```python
# Instantiate the metric
metric = GeneratedAnswerLength()

# Prepare the responses (assuming you have a ResponseWrapper object)
responses = ...  # Your ResponseWrapper containing responses

# Call the metric
result = metric(responses)

# Access the score and raw values
print(f"Score: {result.score}")
print(f"Raw values: {result.raw}")
```

Each metric returns a `MetricResult` object containing the computed score and raw values. You can access these values using the `score` and `raw` attributes, respectively. Additional computed information are included in the `misc` attribute.

### Notes

- Each metric requires a ResponseWrapper object that contains the responses to evaluate.
- Some metrics, like `ROUGE` or `BERTScore`, might need additional arguments (e.g., rouge_type or metric_args). Ensure to provide them when calling the metric.

```python
# Example for ROUGE
rouge_metric = ROUGE(rouge_type="rouge1")
result = rouge_metric(responses)

# Example for BERTScore with additional arguments
bertscore_metric = BERTScore(lang="en")
result = bertscore_metric(responses)
```
