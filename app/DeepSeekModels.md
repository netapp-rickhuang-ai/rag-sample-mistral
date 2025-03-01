# Model Comparison

## Model Set: Copilot

### Model gpt-4o
- **Accuracy:** 90%
- **Speed:** Fast
- **Complexity:** Low
pipeline:
  name: custom_mistral_rag_pipeline
  stages:
    - name: retrieval
      type: dense
      model: "Mistral AI 1.0"
    - name: generation
      type: seq2seq
      model: "gpt-4o"
### Model BERT
- **Accuracy:** 88%
- **Speed:** Medium
- **Complexity:** Medium

### T5 (t2t Transfer-Transfomer)
- **Accuracy:** 85%
- **Speed:** Slow
- **Complexity:** High

## Model Set 2: Mistral

### Mistral AI 1.0
- **Accuracy:** 85%
- **Speed:** Medium
- **Complexity:** High

### RoBERTa
- **Accuracy:** 87%
- **Speed:** Fast
- **Complexity:** Low

### XLNET
- **Accuracy:** 90%
- **Speed:** Fast
- **Complexity:** Medium