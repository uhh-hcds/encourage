# Metric Explanations

This document explains the different metrics used to evaluate responses, their associated calculations, and the arguments that can be used when calling the metrics.

---

## 1. **Generated Answer Length**

### Explanation

This metric computes the average length of the generated answers by counting the number of words in each response.

### Computation

Let \( r_1, r_2, \dots, r_n \) be the generated responses. The length of each response is computed as the number of tokens (words) in it. The average length \( L_{\text{gen}} \) is then:

\[
L_{\text{gen}} = \frac{1}{n} \sum_{i=1}^{n} \text{len}(r_i)
\]

where \( \text{len}(r_i) \) represents the number of words in response \( r_i \).

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
  
No additional arguments are needed when calling this metric.

---

## 2. **Reference Answer Length**

### Explanation

This metric computes the average length of the reference answers by counting the number of words in each reference.

### Computation

Let \( r_1, r_2, \dots, r_n \) be the generated responses, and \( ref_1, ref_2, \dots, ref_n \) be the corresponding reference answers. The length of each reference answer is computed as the number of tokens (words) in it. The average length \( L_{\text{ref}} \) is then:

\[
L_{\text{ref}} = \frac{1}{n} \sum_{i=1}^{n} \text{len}(ref_i)
\]

where \( \text{len}(ref_i) \) represents the number of words in reference answer \( ref_i \).

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
  
No additional arguments are needed when calling this metric.

---

## 3. **Context Length**

### Explanation

This metric computes the average length of the context, where the context is a list of documents or sentences associated with each response.

### Computation

Let \( c_1, c_2, \dots, c_n \) represent the contexts for each response, and \( \text{len}(c_i) \) the length of each context. The total length for each response is the sum of the lengths of all context items. The average length \( L_{\text{context}} \) is computed as:

\[
L_{\text{context}} = \frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{m_i} \text{len}(c_{ij})
\]

where \( m_i \) is the number of context items for response \( r_i \), and \( \text{len}(c_{ij}) \) is the length of context item \( j \) in context \( c_i \).

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses and their associated contexts to evaluate.

No additional arguments are needed when calling this metric.

---

## 4. **BLEU Score**

### Explanation

This metric computes the BLEU (Bilingual Evaluation Understudy) score to evaluate the quality of the generated responses based on reference answers.

### Computation

The BLEU score \( \text{BLEU}(P, R) \) is computed based on the n-grams in the prediction \( P \) and the reference \( R \). It is calculated as:

\[
\text{BLEU}(P, R) = BP \times \exp\left(\sum_{n=1}^{N} w_n \cdot \log p_n(P, R)\right)
\]

where \( BP \) is the brevity penalty, \( p_n(P, R) \) is the precision for n-grams, and \( w_n \) is the weight for n-grams.

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
  
No additional arguments are needed when calling this metric.

---

## 5. **GLEU Score**

### Explanation

This metric computes the GLEU score, which is a variant of BLEU, designed to evaluate text generation by matching the generated text with references based on n-grams.

### Computation

The GLEU score is computed similarly to BLEU, but it focuses on matching n-grams while also considering precision and recall:

\[
\text{GLEU}(P, R) = \frac{\text{Precision}(P, R) \times \text{Recall}(P, R)}{\text{Precision}(P, R) + \text{Recall}(P, R)}
\]

where \( P \) is the generated response and \( R \) is the reference.

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
  
No additional arguments are needed when calling this metric.

---

## 6. **ROUGE Score**

### Explanation

This metric computes the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score, which evaluates the overlap between n-grams in the predicted and reference responses. Variants of ROUGE include ROUGE-1, ROUGE-2, and ROUGE-L.

### Computation

For ROUGE-1, the score is computed as the ratio of overlapping unigrams (words) between the prediction and reference:

\[
\text{ROUGE-1} = \frac{\sum_{i=1}^{n} \text{Count}_{\text{overlap}}(P_i, R_i)}{\sum_{i=1}^{n} \text{Count}(R_i)}
\]

where \( \text{Count}(R_i) \) is the total occurrences of a word in the reference, and \( \text{Count}_{\text{overlap}}(P_i, R_i) \) is the number of overlapping words between the prediction and reference.

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
- `rouge_type`: A string specifying the type of ROUGE score to compute, such as `"rouge1"`, `"rouge2"`, `"rougeL"`, or `"rougeLsum"`.
  
---

## 7. **BERTScore**

### Explanation

This metric computes the BERTScore, a similarity measure based on contextualized word embeddings from BERT.

### Computation

The BERTScore is computed by calculating cosine similarity between the embeddings of each word in the prediction and the reference:

\[
\text{BERTScore}(P, R) = \frac{1}{N} \sum_{i=1}^{N} \text{cosine}(P_i, R_i)
\]

where \( P_i \) and \( R_i \) are the embeddings for the i-th word in the prediction and reference, and \( N \) is the number of words.

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
- `metric_args`: A dictionary of additional arguments for the BERTScore, such as `lang` for specifying the language model to use.
  
---

## 8. **F1 Score**

### Explanation

The F1 score measures the balance between precision and recall for the generated answers compared to reference answers.

### Computation

The F1 score is computed as:

\[
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

where precision is the ratio of correct positive predictions to all positive predictions, and recall is the ratio of correct positive predictions to all actual positives.

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.

No additional arguments are needed when calling this metric.

---

## 9. **Exact Match**

### Explanation

This metric computes the percentage of exact matches between the generated answers and the reference answers.

### Computation

The Exact Match score is calculated as:

\[
\text{Exact Match} = \frac{\text{Count of Exact Matches}}{n}
\]

where \( n \) is the total number of responses and "Exact Matches" refers to responses that exactly match their corresponding reference answers.

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.

No additional arguments are needed when calling this metric.

---

## 10. **Mean Reciprocal Rank (MRR)**

### Explanation

This metric computes the Mean Reciprocal Rank (MRR), which evaluates the rank position of the first relevant document in a ranked list of responses.

### Computation

The MRR is computed as the mean of the reciprocal ranks \( r_1, r_2, \dots, r_n \) for each query:

\[
MRR = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{r_i}
\]

where \( r_i \) is the rank position of the first relevant response for query \( i \).

### Arguments

- `responses`: A `ResponseWrapper` object containing the responses to evaluate.
  
No additional arguments are needed when calling this metric.
