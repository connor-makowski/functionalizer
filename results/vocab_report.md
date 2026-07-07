# Vocabulary Analysis Report

This report compares the vocabularies learned by each tokenizer configuration across datasets and target vocab sizes, and measures how efficiently each one compresses held-out validation text.

**Metrics:**

* **Actual Vocab** - number of vocab entries actually learned by BPE training (capped at the target size).
* **Unique Concepts** - count of distinct "root" tokens after lowercasing, stripping diacritics/PUA opcodes, and collapsing repeated characters; a proxy for how much semantic redundancy is baked into the vocab.
* **Concept Density (%)** - Unique Concepts / Actual Vocab; higher means less redundancy per vocab slot.
* **Chars / Token** - average characters produced per token on the validation sample; higher means better compression.
* **Token Inflation vs Baseline (%)** - for Functionalizer configs, the change in chars/token relative to the non-Functionalizer tokenizer using the same pre-tokenization split.
* **Vocab Size Δ vs Baseline (%)** - for Functionalizer configs, the change in actual vocab size relative to the baseline, only reported when both configs trained under the target vocab size budget.

## Dataset: Wikitext

### Target Vocab Size: 128k

| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |
|---|---|---|---|---|---|---|
| **Split Only** | 51780 | 44421 | 85.79% | 2.4704 | - | - |
| **Split + Functionalizer** | 43700 | 43283 | 99.05% | 2.2659 | +9.02% | -15.60% |
| **Split + Functionalizer (Repeat)** | 52013 | 44487 | 85.53% | 2.4750 | -0.19% | +0.45% |
| **Llama Split Only** | 53262 | 43910 | 82.44% | 4.2189 | - | - |
| **Llama Split + Functionalizer** | 45714 | 42762 | 93.54% | 3.6215 | +16.50% | -14.17% |

## Dataset: Python-Codes

### Target Vocab Size: 128k

| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |
|---|---|---|---|---|---|---|
| **Split Only** | 33399 | 24281 | 72.70% | 2.1984 | - | - |
| **Split + Functionalizer** | 28017 | 23533 | 84.00% | 2.3070 | -4.71% | -16.11% |
| **Split + Functionalizer (Repeat)** | 33445 | 24196 | 72.35% | 2.4362 | -9.76% | +0.14% |
| **Llama Split Only** | 37181 | 17172 | 46.18% | 3.7522 | - | - |
| **Llama Split + Functionalizer** | 31280 | 16426 | 52.51% | 3.2600 | +15.10% | -15.87% |

## Dataset: TinyStories

### Target Vocab Size: 128k

| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |
|---|---|---|---|---|---|---|
| **Split Only** | 14280 | 12492 | 87.48% | 2.2937 | - | - |
| **Split + Functionalizer** | 12853 | 12494 | 97.21% | 2.1553 | +6.42% | -9.99% |
| **Split + Functionalizer (Repeat)** | 14665 | 12695 | 86.57% | 2.2919 | +0.08% | +2.70% |
| **Llama Split Only** | 16981 | 12955 | 76.29% | 4.2100 | - | - |
| **Llama Split + Functionalizer** | 15639 | 12847 | 82.15% | 3.7533 | +12.17% | -7.90% |

## Dataset: CSN-Python

### Target Vocab Size: 128k

| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |
|---|---|---|---|---|---|---|
| **Split Only** | 94605 | 69308 | 73.26% | 1.8928 | - | - |
| **Split + Functionalizer** | 82575 | 67596 | 81.86% | 2.5848 | -26.77% | -12.72% |
| **Split + Functionalizer (Repeat)** | 94243 | 69035 | 73.25% | 2.7177 | -30.35% | -0.38% |
| **Llama Split Only** | 90834 | 39032 | 42.97% | 4.0416 | - | - |
| **Llama Split + Functionalizer** | 77785 | 37657 | 48.41% | 3.4345 | +17.67% | -14.37% |

## Dataset: CSN-Java

### Target Vocab Size: 128k

| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |
|---|---|---|---|---|---|---|
| **Split Only** | 62436 | 49826 | 79.80% | 2.1356 | - | - |
| **Split + Functionalizer** | 54371 | 49085 | 90.28% | 2.3325 | -8.44% | -12.92% |
| **Split + Functionalizer (Repeat)** | 62257 | 49650 | 79.75% | 2.6693 | -19.99% | -0.29% |
| **Llama Split Only** | 84810 | 44849 | 52.88% | 4.2262 | - | - |
| **Llama Split + Functionalizer** | 76664 | 44779 | 58.41% | 3.1723 | +33.22% | -9.60% |

## Dataset: CSN-Go

### Target Vocab Size: 128k

| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |
|---|---|---|---|---|---|---|
| **Split Only** | 15735 | 14719 | 93.54% | 1.9750 | - | - |
| **Split + Functionalizer** | 15281 | 14894 | 97.47% | 1.8464 | +6.97% | -2.89% |
| **Split + Functionalizer (Repeat)** | 15672 | 14606 | 93.20% | 1.9699 | +0.26% | -0.40% |
| **Llama Split Only** | 20458 | 15332 | 74.94% | 2.6201 | - | - |
| **Llama Split + Functionalizer** | 20125 | 15394 | 76.49% | 2.3779 | +10.19% | -1.63% |

