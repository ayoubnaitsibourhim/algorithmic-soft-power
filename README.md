# algorithmic-soft-power
Official repository for "Algorithmic Soft Power?" (UNICART 2026 / Quantrade Journal). A 5-step basis-free audit pipeline (RSA, CKA, baseline-adjusted permutation tests) measuring semantic contamination and cultural resilience across multilingual LLM alignment stages (OLMo-2, Qwen2.5).
# Algorithmic Soft Power? Measuring Semantic Contamination in Multilingual Latent Spaces

[![Conference](https://img.shields.io/badge/UNICART_2026-University_of_Palermo-teal)](#)
[![Journal](https://img.shields.io/badge/Target-Quantrade_Journal_of_Complex_Systems-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

Official code, lexicon, and diagnostic suite for the paper:  
**"Algorithmic Soft Power? Measuring Semantic Contamination of Culturally Situated Concepts Under Preference Alignment in Multilingual Latent Spaces"**  
*Presented at the 11th UNICART International Conference (University of Palermo, 12 November 2026).*

**Authors:** Ayoub Nait Si Bourhim & Mohammed Ali Boulaich  
*LAREMEF Lab, École Nationale de Commerce et de Gestion de Fès (ENCG Fès), Sidi Mohamed Ben Abdellah University*

---

## Overview

When multilingual Large Language Models (LLMs) undergo human preference alignment—such as Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)—do they quietly assimilate culturally situated concepts into dominant Western-centric baselines (e.g., pulling *zakat* toward *charity*, *shura* toward *democracy*, or *banlieue* toward *ghetto*)?

This repository provides a reproducible **five-step geometric audit** to test the semantic contamination hypothesis across open-weight model checkpoints (**OLMo-2 1B** and **Qwen2.5 1.5B**). 

Crucially, we show that **naive cross-checkpoint cosine comparisons yield up to an 85% false-positive rate** due to coordinate-system rotations and baseline similarity confounds. By pairing rotation-invariant structural diagnostics (**RSA** and **Linear CKA**) with baseline-adjusted permutation tests and within-term paired indices, this pipeline isolates genuine conceptual displacement from general cross-lingual alignment.

---

## Key Findings

1. **Coordinate Breaks Mask Structural Preservation:** While raw centered cosine similarity across base-to-instruct checkpoints drops near zero (`cos_centered ≈ 0`), basis-free metrics prove that internal relational geometry is strongly preserved across layers (**RSA ≈ 0.88–0.97**, **CKA ≈ 0.92–0.98**).
2. **Where Geometry Changes:** Preference alignment stages (`SFT → DPO` and `DPO → Instruct`) alter relational term geometry **30–50× less** than the initial `Base → SFT` transition.
3. **General Cross-Lingual Pull vs. Conceptual Assimilation:** While Arabic cultural terms exhibit a small, statistically reliable shift toward dominant analogues during the DPO stage (`+0.0037`, Holm *p* < .05 in OLMo-2), they shift by an almost identical magnitude toward their literal English translations (`+0.0035`). The paired within-term index confirms no preferential convergence toward Western cultural analogues over literal glosses.

---

## The Five-Step Audit Pipeline

The repository is structured to run end-to-end in ~30 minutes on a single Google Colab T4 GPU:

1. **Anchor Construction & Audit (`lexicon.csv`, `build_lexicon.py`, `build_audit.py`)**
   * Curated dataset of culturally situated terms in Arabic and French paired with dual English anchors (literal translation gloss vs. dominant-culture analogue) and trilingual neutral controls (`ar`, `fr`, `en`).
2. **Contextual Embedding Extraction (`extract.py`)**
   * Extracts hidden-state representations across all transformer layers using split-half prompt templates (`E_A`, `E_B`) across Base, SFT, DPO, and Instruct checkpoints.
3. **Coordinate & Reliability Diagnostics (`diagnose.py`)**
   * Computes layer-by-layer uncentered/centered cosine, Representational Similarity Analysis (RSA), Linear CKA, split-half template reliability ceilings, and top-1 dimension variance (`top1_var`) to detect massive activations.
4. **Statistical Testing (`run_analysis.py`, `scanalysis.py`)**
   * Executes baseline-adjusted permutation tests (with Holm-Bonferroni step-down correction) for profile drift, analogue shift, and the paired within-term differential ($\Delta_{\text{analogue}} - \Delta_{\text{gloss}}$).
5. **Validation & Sensitivity (`kappa.py`)**
   * Synthetic null validation suite (400 runs per cell) and inter-annotator agreement tooling for lexical audit sensitivity checks.

---

## Quick Start

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/semantic-contamination.git
cd semantic-contamination

# 2. Run the full extraction and statistical pipeline
bash run_all.sh

# 3. Run structural and basis-continuity diagnostics on extracted embeddings
python diagnose.py base=emb/olmo_base.npz sft=emb/olmo_sft.npz instruct=emb/olmo_instruct.npz
