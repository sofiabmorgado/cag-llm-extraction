# cag-llm-extraction

Official implementation for the paper:

**"Can Large Language Models Reliably Extract Physiology Index Values from Coronary Angiography Reports?"**

This repository evaluates different strategies for extracting coronary physiology indices (FFR, iFR) from unstructured coronary angiography (CAG) reports, including regex baselines, unconstrained LLMs, and constrained LLM generation.

---

## Repository Structure

### Extraction Methods

- `extractor_baseline_regex.py`  
  Rule-based extraction using regular expressions (deterministic baseline).

- `extractor_baseline_llms.py`  
  LLM-based extraction without structured output constraints.

- `extractor_constrained_llms.py`  
  LLM-based extraction with schema-constrained generation.

- `backbone_extractor_constrained_llms.py`  
  Core logic for constrained extraction (prompting, schema enforcement, parsing).

---

### Postprocessing 

- `postprocessing_constrained.py`  
  Cleans constrained outputs.

- `postprocessing_implausible.py`  
  Handles physiologically implausible outputs.

- `postprocessing_regex.py`  
  Regex-based confirmation layer.

---

### Evaluation

- `Evaluation.py`  
  Framework for computing metrics and comparing extraction methods.

---

## Important Files

- `output_schema.json`  
  JSON schema used for constrained generation.

- `data/examples.csv`  
  Few-shot examples for constrained extraction.

- `data/examples_implausible.csv`  
  Few-shot examples including implausible values.

---


## Data

Due to GDPR regulations, real-world clinical data associated with this project cannot be made publicly available.

---

## Reproducibility

To reproduce results:

1. Run one of the extractor scripts.
2. Apply the corresponding postprocessing pipeline.
3. Evaluate results using `Evaluation.py`.

