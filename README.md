# Setup and Execution Guide for Tokenizer Training & Downstream Evaluation

This guide outlines the environment setup process, the correct execution order of the scripts, and key parameters for running experiments. 

The modified `tokenizers` library is installed via the steps below, and can be explored here: https://github.com/connor-makowski/tokenizers/tree/functionalizer

---

## 1. Setup Process

To run the training and evaluation suite successfully, you must configure a virtual environment, install dependencies, and build the custom Rust-backed local `tokenizers` bindings in editable mode.

### Steps:

1. **Activate your virtual environment** (create one if needed):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   pip install git+https://github.com/connor-makowski/tokenizers.git@functionalizer#subdirectory=bindings/python
   ```
   *(Note: This requires a Rust toolchain `cargo` to be installed on your system to compile the source code).*

3. **Ensure the Go language is installed (Optional)**:
   The Go benchmarking in `downstream_tests.py` relies on the `gofmt` command. If this isn't present it falls back to basic bracket matching which gives unrealistically high success rates.

4. **Ensure your Hugging Face token is set (Optional)**:
   If streaming datasets with high rate limits or accessing restricted datasets, authenticate with Hugging Face via `huggingface-cli login` or by exporting the `HF_TOKEN` environment variable.

---

## 2. Execution Order

Running a full evaluation cycle consists of profiling the tokenizer first, followed by training the downstream models and evaluating their generation metrics.

### Step 1: Profile Tokenizer Configurations
Execute the `vocab.py` script to run isolated tokenizer-level experiments. This script profiles vocab size exhaustion limits, vocabulary density, and token-to-character compression/inflation ratios across various datasets (e.g., Wikitext, Python-Codes, FineWeb-Edu, GitHub-Code) and configurations (Llama Split, etc.).
```bash
python vocab.py
```
* **Output**: Prints a tokenizer comparison summary table to stdout and writes detailed results to `./results/vocab_results.json` and `./results/vocab_report.md`.

### Step 2: Run Training & Baseline Experiments
Execute the `run_experiments.py` script to train the model variations across the selected datasets and seeds. This creates the required tokenizer configurations and saved model checkpoints.
```bash
python run_experiments.py --model_size 125M --num_seeds 1
```
* **Output**: Writes result summary JSONs for each configuration to `./results/{model_size}/results_*_seed{seed}_{model_size}.json` and aggregates metrics across seeds into the comparison report at `./results/training_report_{model_size}.md`. Model/tokenizer checkpoints are saved in `./checkpoints/{dataset}_{tokenizer}_seed{seed}_{model_size}/`.

### Step 3: Run Downstream Evaluations
Once training checkpoints are populated, run the `downstream_tests.py` script to test task syntax success rates, tokenization efficiency, and generation throughput. Make sure to specify the matching `--model_size` to locate the correct checkpoints.
```bash
python downstream_tests.py --model_size 125M --num_seeds 1
```
* **Output**: Generates a downstream latency and throughput report at `./results/downstream_report_{model_size}.md`, raw per-seed cached results at `./results/downstream_results_{model_size}.json`, and sample completions at `./results/downstream_samples_{model_size}.md`.

---

## 3. Important CLI Flags

### Highlighted Flag: `--model_size`
* **Option**: `--model_size [25M | 125M | 2B]`
* **Default**: `125M`
* **Description**: Sets the training capacity class. Automatically configures the GPT-2 architecture and step count, partitioning checkpoints, reports, and results by model size:
  * **25M**: 6 layers, 8 heads, 512 embedding dimensions, 512 context window, and 3,000 steps.
  * **125M**: 12 layers, 12 heads, 768 embedding dimensions, 512 context window, and 50,000 steps.
  * **2B**: 24 layers, 32 heads, 2560 embedding dimensions, 1024 context window, and 100,000 steps.

### Other Key Flags

#### Common Flags (Used in both training and downstream scripts)
* `--num_seeds`: Sets the number of seeds to test (from `1` to `5`). Utilizes the predefined list of seeds (`[1, 2, 3, 4, 5]`) to ensure repeatable results and generates statistical reports showing `mean ± std`.
* `--model_size`: Distinguishes checkpoint retrieval paths and generated reports between runs of different sizes (`25M`, `125M`, `2B`).
* `--run_tag`: Optional string suffix appended to checkpoint folders and result files to prevent overwriting artifacts across experimental runs.

#### Training Flags (For `run_experiments.py` and `train.py`)
* `--max_steps`: Overrides the default number of training steps.
* `--eval_steps`: Overrides the default frequency of validation loss checks.
* `--batch_size`: Sets per-device training batch size (default: `16`).
* `--lr`: Configures learning rate (defaults by model size: `5e-4` for 25M, `4e-4` for 125M, `1.5e-4` for 2B).
* `--context_length`: Manually overrides the context window length.
* `--python_bin`: (*`run_experiments.py` only*) Path to the Python interpreter binary to use for launching sub-processes (default: `./venv/bin/python`).

#### Downstream Flags (For `downstream_tests.py`)
* `--num_prompts`: Sets the number of prompts loaded from the validation split for testing generation metrics (default: `1000`).
* `--max_new_tokens`: Maximum new tokens to generate per prompt (default: `256`).
* `--no_checkpoint`: Ignores existing evaluation JSON results and re-runs evaluations from scratch.
