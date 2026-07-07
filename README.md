# Setup and Execution Guide for Tokenizer Training & Downstream Evaluation

This guide outlines the environment setup process, the correct execution order of the scripts, and key parameters for running experiments.

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
   ```
   *(Note: This requires a Rust toolchain `cargo` to be installed on your system to compile the source code).*

3. **Ensure the Go language is installed (Optional)**:
   The go benchmarking in `downstream_tests.py` relies on the `gofmt` command. If this isn't present it falls back to basic bracket matching which gives unrealistically high success rates.

---

## 2. Execution Order

Running a full evaluation cycle consists of profiling the tokenizer first, followed by training the downstream models and evaluating their generation metrics.

### Step 1: Profile Tokenizer Configurations
Execute the `vocab.py` script to run isolated tokenizer-level experiments. This script profiles vocab size exhaustion limits, vocabulary density, and token-to-character compression/inflation ratios across various datasets (e.g., Wikitext, TinyStories, CSN) and configurations (Metaspace + Split, Split Only, Llama Split, etc.).
```bash
python vocab.py
```
* **Output**: Prints a tokenizer comparison summary table to stdout and writes detailed results to `./results/vocab_results.json`.

### Step 2: Run Training & Baseline Experiments
Execute the `run_experiments.py` script to train the model variations across the selected datasets and seeds. This creates the required tokenizer configurations and saved model checkpoints.
```bash
python run_experiments.py --model_size 25M --num_seeds 1
```
* **Output**: Writes result summary JSONs for each configuration to `./results/results_*_{model_size}.json` and aggregates metrics across seeds into the comparison report at `./results/training_report_{model_size}.md`. Model/tokenizer checkpoints are saved in `./checkpoints/*_{model_size}/`.

### Step 3: Run Downstream Evaluations
Once training checkpoints are populated, run the `downstream_tests.py` script to test task syntax success rates, tokenization efficiency, and generation throughput. Make sure to specify the matching `--model_size` to locate the correct checkpoints.
```bash
python downstream_tests.py --num_seeds 1 --model_size 25M
```
* **Output**: Generates a downstream latency and coherence report at `./results/downstream_report_{model_size}.md`.

---

## 3. Important CLI Flags

### Highlighted Flag: `--model_size`
* **Option**: `--model_size [25M | 125M]`
* **Default**: `25M`
* **Description**: Sets the training capacity class. Choosing `125M` automatically switches the model configuration to a larger GPT-2 scale and ensures all checkpoints, output reports, and JSON results are partitioned distinctly by model size (e.g., suffixed with `_125M`):
  * **25M**: 6 layers, 8 heads, 512 embedding dimensions, 256 context window, and 3,000 steps.
  * **125M**: 12 layers, 12 heads, 768 embedding dimensions, 1024 context window, and 50,000 steps.

### Other Key Flags

#### Common Flags (Used in both training and downstream scripts)
* `--num_seeds`: Sets the number of seeds to test (from `1` to `5`). Utilizes the predefined list of seeds (`[42, 100, 2026, 7, 999]`) to ensure repeatable results and generates statistical reports showing `mean ± std`.
* `--model_size`: Utilized to distinguish checkpoint retrieval paths and generated reports between runs of different sizes (e.g., `_25M` vs `_125M`).

#### Training Flags (For `run_experiments.py` and `train.py`)
* `--max_steps`: Overrides the default number of training steps.
* `--eval_steps`: Overrides the default frequency of validation loss checks.
* `--batch_size`: Sets training batch size (default: `32`).
* `--lr`: Configures learning rate (default: `5e-4`).
* `--context_length`: Manually overrides the context window length.
* `--python_bin`: (*`run_experiments.py` only*) Path to the Python interpreter binary to use for launching sub-processes (default: `./venv/bin/python`).

#### Downstream Flags (For `downstream_tests.py`)
* `--num_prompts`: Sets the number of prompts loaded from the validation split for testing generation metrics (default: `100`).
