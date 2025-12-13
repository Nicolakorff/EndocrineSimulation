(... In process)
# Endocrine-Modulated Language Models

This repository contains the code, data, and experiments developed for the Master's Thesis (TFM):

**“Endocrine Simulation Applied to LLMs: An Artificial Neuromodulation Framework for Text Generation”**

Master’s Degree in Large Language Models and Computational Linguistics.
Tutor and co-author: Matías Nuñez

---

## Overview

This work proposes an **endocrine-inspired neuromodulation mechanism** for language models, in which simulated hormonal signals (e.g., dopamine, cortisol, oxytocin, serotonin, adrenaline) dynamically modulate the generation process of an open-weight language model.

The modulation operates **directly at the logit level**, prior to sampling, allowing systematic control over creativity, stability, diversity, and empathic behavior. The framework is evaluated empirically using linguistic metrics, empathy classifiers, and statistical analysis.

---

## Key Contributions

- A **hormone-inspired control vector** for language generation
- **Logit-level neuromodulation** applied to an open-weight LLM
- Static and dynamic hormonal profiles
- Empathy-aware lexical biasing using embeddings
- Comprehensive evaluation:
  - Lexical diversity
  - Perplexity
  - Self-BLEU
  - Empathy classifiers (RoBERTa)
- Formal statistical analysis:
  - ANOVA
  - Tukey HSD post-hoc tests
  - Hormone–metric correlations
- Automatic export of results to **LaTeX tables and figures**

---

## Repository Structure (in process)

EndocrineSimulation/

├── README.md

├── requirements.txt

├── notebooks/

│ └── TFM_Endocrine_LLM_Final.ipynb

├── data/

│ └── prompts/

├── results/

└── src/

---

## Reproducibility

All experiments and analyses can be reproduced by running: `notebooks/TFM_Endocrine_LLM_Final.ipynb`

The notebook includes:
- experiment execution
- metric computation
- statistical analysis
- LaTeX export utilities

---

## Requirements

See `requirements.txt`.

All experiments were run using Python 3.10 in Google Colab.

---

## Limitations

This framework requires **direct access to model logits** and intermediate representations.  
Therefore, it is applicable only to **open-weight language models** (e.g., GPT-2, OPT, LLaMA-based models) and **cannot be directly applied to proprietary API-based LLMs** (e.g., GPT-4, Claude, Gemini).

This limitation is methodological and reflects current access constraints rather than an inherent restriction of the proposed approach.

---

## License

This project is released under the MIT License.
