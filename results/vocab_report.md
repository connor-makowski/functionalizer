# Vocabulary Analysis Report

This report compares the vocabularies learned by each tokenizer configuration across datasets and target vocab sizes, and measures how efficiently each one compresses held-out validation text.

**Metrics:**

* **Actual Vocab** - number of vocab entries actually learned by BPE training (capped at the target size).
* **Unique Concepts** - count of distinct "root" tokens after lowercasing, stripping diacritics/PUA opcodes, and collapsing repeated characters; a proxy for how much semantic redundancy is baked into the vocab.
* **Concept Density (%)** - Unique Concepts / Actual Vocab; higher means less redundancy per vocab slot.
* **Density Δ vs Baseline (%)** - for Functionalizer configs, the relative percentage change in concept density relative to the baseline tokenizer using the same pre-tokenization split.
* **Exhausted** - whether BPE ran out of merges before reaching the target vocab size. The vocabulary-reduction claim is only meaningful when this is `yes` for BOTH configs; a `NO` means the run was capped by the target and Vocab Size Δ is suppressed.
* **Chars / Token (val)** - average characters per token on the held-out sample; higher means better compression.
* **Chars / Token (train)** - the same measure on a sample of the tokenizer's own training text. Reported alongside the held-out figure so any train/held-out gap is visible directly rather than surfacing later as an unexplained discrepancy against the training pipeline.
* **Chars/Token Δ vs Baseline (%)** - for Functionalizer configs, the relative change in character density relative to the non-Functionalizer tokenizer using the same pre-tokenization split (positive = compression gain, negative = token inflation).
* **Vocab Size Δ vs Baseline (%)** - for Functionalizer configs, the change in actual vocab size relative to the baseline, only reported when both configs exhausted below the target vocab size budget.
* **Roundtrip** - counts of (verified / failed / bypassed) sequences. Verified decode matched the NFC-normalized input exactly, failed sequences had decoding mismatches without OOV/UNK, and bypassed sequences contained characters out-of-vocabulary or unknown to the tokenizer.

## Dataset: Wikitext

### Target Vocab Size: 4096k

Tokenizer training corpus: **23,767 documents**.

| Tokenizer Type | Actual Vocab | Exhausted | Unique Concepts | Concept Density (%) | Chars / Token (val) | Chars / Token (train) | Chars/Token Δ vs Baseline (%) | Density Δ vs Baseline (%) | Vocab Size Δ vs Baseline (%) | Roundtrip (verified / failed / bypassed) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama Split Only** | 106023 | yes | 86356 | 81.45% | 4.6045 | 4.7738 | - | - | - | 197 / 0 / 3 |
| **Llama Split + Functionalizer** | 90531 | yes | 84712 | 93.57% | 3.9464 | 4.1007 | -14.29% | +14.88% | -14.61% | 196 / 0 / 4 |

## Dataset: Python-Codes

### Target Vocab Size: 4096k

Tokenizer training corpus: **44,626 documents**.

| Tokenizer Type | Actual Vocab | Exhausted | Unique Concepts | Concept Density (%) | Chars / Token (val) | Chars / Token (train) | Chars/Token Δ vs Baseline (%) | Density Δ vs Baseline (%) | Vocab Size Δ vs Baseline (%) | Roundtrip (verified / failed / bypassed) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama Split Only** | 68471 | yes | 29728 | 43.42% | 4.2030 | 4.0709 | - | - | - | 200 / 0 / 0 |
| **Llama Split + Functionalizer** | 57012 | yes | 28745 | 50.42% | 3.4806 | 3.4347 | -17.19% | +16.13% | -16.74% | 200 / 0 / 0 |

## Dataset: FineWeb-Edu

### Target Vocab Size: 4096k

Tokenizer training corpus: **100,000 documents**.

| Tokenizer Type | Actual Vocab | Exhausted | Unique Concepts | Concept Density (%) | Chars / Token (val) | Chars / Token (train) | Chars/Token Δ vs Baseline (%) | Density Δ vs Baseline (%) | Vocab Size Δ vs Baseline (%) | Roundtrip (verified / failed / bypassed) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama Split Only** | 1214684 | yes | 716927 | 59.02% | 5.0092 | 5.0392 | - | - | - | 197 / 0 / 3 |
| **Llama Split + Functionalizer** | 975169 | yes | 699607 | 71.74% | 4.3642 | 4.3842 | -12.88% | +21.55% | -19.72% | 196 / 0 / 4 |

## Dataset: GitHub-Code-Python

### Target Vocab Size: 4096k

Tokenizer training corpus: **100,000 documents**.

| Tokenizer Type | Actual Vocab | Exhausted | Unique Concepts | Concept Density (%) | Chars / Token (val) | Chars / Token (train) | Chars/Token Δ vs Baseline (%) | Density Δ vs Baseline (%) | Vocab Size Δ vs Baseline (%) | Roundtrip (verified / failed / bypassed) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama Split Only** | 4071598 | yes | 2394740 | 58.82% | 4.2961 | 4.2454 | - | - | - | 198 / 1 / 1 |
| **Llama Split + Functionalizer** | 3356761 | yes | 2315220 | 68.97% | 3.5343 | 3.4082 | -17.73% | +17.27% | -17.56% | 198 / 0 / 2 |

