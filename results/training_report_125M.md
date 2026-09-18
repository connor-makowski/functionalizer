# Tokenizer Functionalizer Comparison Report

This report compares the performance and compute usage of a GPT-2 model (12 layers, 768 embedding dim, 12 attention heads, ~125M parameters) trained over 5 seed(s) with different tokenizers:
1. **Llama Split**
2. **Llama Split + Functionalizer**

Predefined seed list used: [1, 2, 3, 4, 5]

Trained under a standard next-token objective. `± ` denotes the population standard deviation across seeds.

## Dataset: FineWeb-Edu

| Tokenizer Type | Vocab Size | Chars/Token | Chars/Token Δ vs Baseline | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |
|---|---|---|---|---|---|---|---|---|
| **Llama Split** | 16,000.0 ± 0.0 | 4.178 ± 0.000 | 0.00% | 33640.0 ± 7690.9 | 140549.4 ± 32133.0 | 3.3954 ± 0.0110 | 29.83 ± 0.33 | 2.2662 ± 0.0060 |
| **Llama Split + Functionalizer** | 16,000.0 ± 0.0 | 3.847 ± 0.000 | -7.91% ± 0.00% | 33651.1 ± 7496.0 | 129468.1 ± 28839.7 | 3.1261 ± 0.0044 | 22.79 ± 0.10 | 2.2656 ± 0.0026 |

## Dataset: GitHub-Code-Python

| Tokenizer Type | Vocab Size | Chars/Token | Chars/Token Δ vs Baseline | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |
|---|---|---|---|---|---|---|---|---|
| **Llama Split** | 16,000.0 ± 0.0 | 1.767 ± 0.000 | 0.00% | 68908.9 ± 72872.2 | 121767.4 ± 128770.8 | 0.8056 ± 0.0961 | 2.25 ± 0.22 | 1.5697 ± 0.0850 |
| **Llama Split + Functionalizer** | 16,000.0 ± 0.0 | 1.487 ± 0.000 | -15.84% ± 0.00% | 30642.5 ± 465.3 | 45568.0 ± 692.0 | 0.6525 ± 0.0013 | 1.92 ± 0.00 | 1.5328 ± 0.0013 |

