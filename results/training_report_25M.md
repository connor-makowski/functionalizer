# Tokenizer Functionalizer Comparison Report

This report compares the performance and compute usage of a GPT-2 model (6 layers, 512 embedding dim, 8 attention heads, ~25M parameters) trained over 5 seed(s) with different tokenizers:
1. **Metaspace + Split**: Character-based BPE (using Custom Regex split).
2. **Metaspace + Split + Functionalizer**: BPE with the Custom Regex split + Functionalizer pre-tokenizer and decoder (all parameters enabled).
3. **Functionalizer (Repeat)**: BPE with the Custom Regex split + Functionalizer where only repeat=True is enabled (capitalize/serialize/operators disabled).
4. **Split Only**: Split without Metaspace.
5. **Split + Functionalizer**: Split + Functionalizer.
6. **Llama Split Only**: Llama split pattern without Metaspace.
7. **Llama Split + Functionalizer**: Llama split + Functionalizer.
8. **Metaspace + Whitespace**: Metaspace + Whitespace.
9. **Metaspace + Whitespace + Functionalizer**: Metaspace + Whitespace + Functionalizer.

Predefined seed list used: [10, 42, 100, 2026, 7]

## Dataset: TinyStories

| Tokenizer Type | Vocab Size | Chars/Token | Inflation | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |
|---|---|---|---|---|---|---|---|---|
| **Split** | 14,280.0 ± 0.0 | 2.295 ± 0.000 | 0.00% | 55498.7 ± 661.8 | 127354.4 ± 1518.5 | 1.4980 ± 0.0048 | 4.47 ± 0.02 | 1.9209 ± 0.0040 |
| **Split + Functionalizer** | 12,853.0 ± 0.0 | 2.156 ± 0.000 | +6.45% ± 0.00% | 53675.9 ± 1568.6 | 115705.6 ± 3381.3 | 1.4402 ± 0.0011 | 4.22 ± 0.00 | 1.9506 ± 0.0010 |
| **Split + Functionalizer (Repeat)** | 14,665.0 ± 0.0 | 2.294 ± 0.000 | +0.03% ± 0.00% | 54220.7 ± 2187.0 | 124387.2 ± 5017.1 | 1.5125 ± 0.0023 | 4.54 ± 0.01 | 1.9335 ± 0.0020 |

## Dataset: CSN-Python

| Tokenizer Type | Vocab Size | Chars/Token | Inflation | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |
|---|---|---|---|---|---|---|---|---|
| **Split** | 16,000.0 ± 0.0 | 1.901 ± 0.000 | 0.00% | 52237.9 ± 17.3 | 99306.9 ± 33.0 | 2.5551 ± 0.0041 | 12.87 ± 0.05 | 3.8345 ± 0.0082 |
| **Split + Functionalizer** | 16,000.0 ± 0.0 | 2.620 ± 0.000 | -27.44% ± 0.00% | 54584.6 ± 19.9 | 143018.9 ± 52.3 | 3.1686 ± 0.0040 | 23.77 ± 0.10 | 3.3512 ± 0.0052 |
| **Split + Functionalizer (Repeat)** | 16,000.0 ± 0.0 | 2.771 ± 0.000 | -31.39% ± 0.00% | 55470.3 ± 25.3 | 153699.3 ± 70.0 | 3.3496 ± 0.0049 | 28.49 ± 0.14 | 3.3498 ± 0.0059 |

## Dataset: CSN-Go

| Tokenizer Type | Vocab Size | Chars/Token | Inflation | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |
|---|---|---|---|---|---|---|---|---|
| **Split** | 15,735.0 ± 0.0 | 2.226 ± 0.000 | 0.00% | 60367.7 ± 23.3 | 134352.3 ± 51.8 | 2.8526 ± 0.0093 | 17.33 ± 0.16 | 3.6029 ± 0.0151 |
| **Split + Functionalizer** | 15,281.0 ± 0.0 | 1.959 ± 0.000 | +13.63% ± 0.00% | 60554.2 ± 36.4 | 118598.1 ± 71.3 | 2.6738 ± 0.0066 | 14.50 ± 0.10 | 3.9166 ± 0.0133 |
| **Split + Functionalizer (Repeat)** | 15,672.0 ± 0.0 | 2.224 ± 0.000 | +0.07% ± 0.00% | 60627.7 ± 30.9 | 134833.6 ± 68.8 | 2.8677 ± 0.0085 | 17.60 ± 0.15 | 3.6308 ± 0.0139 |

