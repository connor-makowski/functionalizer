# Downstream Tasks Evaluation Report

This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) for the 125M models trained under the different tokenization configurations across 5 seed(s).

Predefined seed list used: [1, 2, 3, 4, 5]

Greedy generation budget: **256 new tokens** per prompt (stops early on `[SEP]`).

### Note on Token Collapse
When small models finish generating content without emitting an explicit end token (`[SEP]`), they frequently fall into repetition loops. Collapse is judged on the **generated token stream**: generation **stops early** once a cycle of up to **20 tokens** has occurred **4 times** back to back (a single token emitted 4 times in a row, `a b a b ...`, and so on). The rule is applied to tokens rather than to decoded characters on purpose: the failure being detected is a decoding loop, which lives in the token stream, and a character-length rule silently missed loops built from short or zero-width tokens as well as every multi-token cycle. The trade-off is that a token-count rule is not perfectly comparable across tokenizers, since the arms differ in chars/token by construction; a repeating cycle is degenerate regardless of how many characters it prints, and **Avg Chars Pre-Collapse** still reports the per-arm character yield. The cutoff is the **end of the cycle's first occurrence**, so the one legitimate emission of the pattern is kept and only the repeats are discarded: `[1, 2, 3, 6, 6, 6, 6]` counts as **4 tokens pre-collapse**. **Avg Tokens Pre-Collapse** and **Avg Chars Pre-Collapse** measure that prefix, and generation throughput (`Tokens/Sec`, `Chars/Sec`) is computed exclusively over it. Totals over the full generated stream are no longer reported: since generation is killed at the collapse point, a total would be the pre-collapse figure plus a fixed slice of loop, which measures the cutoff rule rather than the model.

**% Collapsed** reports how often the early stop fired, as opposed to `[SEP]` or the token budget. **On code datasets a collapsed generation is scored as a syntax failure automatically**, without being handed to the syntax checker: the model stopped emitting code and started looping, so the fragment before the loop is not a completed function. **% Auto-Failed (Collapse)** reports the share of prompts failed this way; it is a component of the failures behind the syntax success rate, not an additional penalty on top of it.

Generation uses a KV cache, so per-token decode cost is O(1) against the cache rather than a full forward pass over the context. `Tokens/Sec` is therefore **not comparable to earlier runs** that recomputed the whole sequence each step.

### Note on Syntax Checking
Syntax checkers used in this run: `python-ast`. `gofmt` means real Go parsing; `bracket-fallback` means balanced-bracket matching only, which is far more permissive and NOT comparable to the Python results. **Non-Trivial Syntax** (Python only) additionally requires the parsed function body to contain a statement that is not `pass`, a docstring, or a bare identifier.

### Note on Repetition
**Repetition (%)** measures how much of the returned text is repeated content. For each n in 2..4, every overlapping word n-gram is built and the share that is a duplicate is reported (`1 - unique/total`); the headline figure averages those across n, and **Repetition 2-gram (%)** shows n=2 alone. For example `the the the cat went to the the the door` has 9 bigrams of which 6 are distinct, so rep-2 = 33.3%.

This is complementary to collapse detection rather than a restatement of it: `the the the` repeats only three times, well under the 4-repeat collapse rule, yet it is clearly degenerate. Collapse catches hard short-cycle loops; repetition catches soft degeneracy inside otherwise-varied output. Text is split on whitespace, so runs of indentation do not count as repeated content (those are already covered by **% Collapsed** and **% Empty**). Completions too short to contain two n-grams are excluded from the average rather than scored as 0%.

### Note on Decode Fallback
**% Decode Fallback** is the share of prompts where the tokenizer's decoder rejected the generated stream and raw token strings were used instead. This can only occur for Functionalizer configurations (malformed opcode streams); a non-zero value means the corresponding character metrics are approximate.

## Dataset: FineWeb-Edu (Natural Language Generation)

| Tokenizer Type | Avg Tokens Pre-Collapse | Avg Chars Pre-Collapse | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | % Decode Fallback | % Past Context | % Collapsed | Repetition (%) | Repetition 2-gram (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Llama Split Only** | 87.4 ± 3.8 | 357.9 ± 17.5 | 417.8 ± 0.8 | 1710.2 ± 15.4 | 0.1 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 70.3 ± 2.4% | 66.0 ± 0.4 | 68.1 ± 0.3 |
| **Llama Split + Functionalizer** | 76.2 ± 4.2 | 217.7 ± 13.6 | 417.9 ± 0.4 | 1194.2 ± 17.5 | 7.1 ± 0.5% | 0.0 ± 0.0% | 0.0 ± 0.0% | 78.4 ± 1.4% | 55.8 ± 1.5 | 55.8 ± 1.6 |

## Dataset: github-code-python (Code Generation - Python)

| Tokenizer Type | Avg Tokens Pre-Collapse | Avg Chars Pre-Collapse | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | % Decode Fallback | % Past Context | % Collapsed | % Auto-Failed (Collapse) | Repetition (%) | Repetition 2-gram (%) | Syntax Success Rate (%) | Min Syntax Success Rate (%) | Max Syntax Success Rate (%) | Non-Trivial Syntax (%) | Syntax (% of Non-Empty) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Llama Split Only** | 227.7 ± 17.2 | 375.6 ± 23.2 | 420.1 ± 4.0 | 693.8 ± 18.7 | 0.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 15.5 ± 8.2% | 15.5 ± 8.2% | 25.5 ± 6.4 | 31.4 ± 5.4 | 7.70 ± 2.63% | 4.70% | 12.60% | 5.94 ± 3.00 | 7.70 ± 2.63 |
| **Llama Split + Functionalizer** | 231.9 ± 7.0 | 319.0 ± 10.1 | 428.9 ± 0.7 | 589.9 ± 3.4 | 2.1 ± 3.8% | 0.0 ± 0.0% | 0.0 ± 0.0% | 15.5 ± 3.4% | 15.5 ± 3.4% | 17.9 ± 1.0 | 22.8 ± 1.4 | 9.12 ± 1.22% | 8.10% | 11.30% | 6.40 ± 2.32 | 9.32 ± 1.15 |

