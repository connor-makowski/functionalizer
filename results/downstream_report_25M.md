# Downstream Tasks Evaluation Report

This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) for the models trained under the different tokenization configurations across 5 seed(s).

Predefined seed list used: [10, 42, 100, 2026, 7]

## Dataset: TinyStories (Natural Language Generation)

| Tokenizer Type | Avg Tokens Generated | Avg Tokens Pre-Collapse | Avg Chars Generated | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | Coherence (1 - Repetition) (%) |
|---|---|---|---|---|---|---|---|
| **Split Only** | 7.6 ± 0.5 | 3.6 ± 0.5 | 17.4 ± 1.6 | 303.0 ± 16.0 | 1117.9 ± 42.8 | 1.0 ± 0.0% | 99.00 ± 0.00% |
| **Split + Functionalizer** | 15.7 ± 1.1 | 11.7 ± 1.1 | 40.1 ± 4.0 | 311.0 ± 3.3 | 954.5 ± 28.2 | 2.2 ± 1.2% | 97.80 ± 1.17% |
| **Split + Functionalizer (Repeat)** | 7.4 ± 0.3 | 3.4 ± 0.3 | 16.6 ± 0.9 | 307.2 ± 6.1 | 1134.5 ± 25.2 | 1.0 ± 0.0% | 99.00 ± 0.00% |

## Dataset: CSN-Python (Code Generation)

| Tokenizer Type | Avg Tokens Generated | Avg Tokens Pre-Collapse | Avg Chars Generated | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | Syntax Success Rate (%) |
|---|---|---|---|---|---|---|---|
| **Split Only** | 6.0 ± 1.3 | 2.0 ± 1.3 | 7.9 ± 2.4 | 313.3 ± 6.0 | 626.7 ± 79.1 | 52.0 ± 25.7% | 0.00 ± 0.00% |
| **Split + Functionalizer** | 9.1 ± 1.1 | 5.1 ± 1.1 | 15.6 ± 3.1 | 316.8 ± 2.9 | 507.6 ± 200.1 | 77.6 ± 11.6% | 9.20 ± 9.00% |
| **Split + Functionalizer (Repeat)** | 7.3 ± 0.2 | 3.3 ± 0.2 | 12.1 ± 0.6 | 317.5 ± 4.1 | 769.8 ± 25.2 | 92.6 ± 2.9% | 0.00 ± 0.00% |

## Dataset: CSN-Go (Code Generation)

| Tokenizer Type | Avg Tokens Generated | Avg Tokens Pre-Collapse | Avg Chars Generated | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | Syntax Success Rate (%) |
|---|---|---|---|---|---|---|---|
| **Split Only** | 8.5 ± 3.7 | 4.6 ± 3.8 | 10.6 ± 6.1 | 314.6 ± 3.3 | 401.1 ± 74.1 | 73.2 ± 23.7% | 2.20 ± 3.12% |
| **Split + Functionalizer** | 10.7 ± 7.6 | 7.0 ± 8.2 | 13.6 ± 11.3 | 314.6 ± 1.9 | 431.5 ± 28.4 | 71.8 ± 26.3% | 4.00 ± 4.65% |
| **Split + Functionalizer (Repeat)** | 9.9 ± 3.9 | 6.1 ± 4.0 | 13.4 ± 7.4 | 312.2 ± 4.7 | 436.9 ± 81.8 | 67.6 ± 21.7% | 2.80 ± 4.17% |

