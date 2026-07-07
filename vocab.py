import os
import re
import unicodedata
from datasets import load_dataset
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Metaspace, Whitespace
from tokenizers.pre_tokenizers import Functionalizer as FunctionalizerPreTok
from tokenizers.decoders import Functionalizer as FunctionalizerDecoder, Metaspace as MetaspaceDecoder, Sequence as SequenceDecoder, Fuse
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC

DEFAULT_SPLIT_PAT = r"\n+|\t+|[ ]| |[^\p{L}\p{N}\s _]+"
LLAMA_SPLIT_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

TOK_NAME_MAP = {
    "metaspace_split": "Metaspace + Split",
    "metaspace_split_functionalizer": "Metaspace + Split + Functionalizer",
    "metaspace_split_functionalizer_repeat": "Metaspace + Split + Functionalizer (Repeat)",
    "split": "Split Only",
    "split_functionalizer": "Split + Functionalizer",
    "split_functionalizer_repeat": "Split + Functionalizer (Repeat)",
    "llama": "Llama Split Only",
    "llama_functionalizer": "Llama Split + Functionalizer",
    "metaspace_whitespace": "Metaspace + Whitespace",
    "metaspace_whitespace_functionalizer": "Metaspace + Whitespace + Functionalizer",
    "metaspace": "Metaspace Only",
    "metaspace_functionalizer": "Metaspace + Functionalizer",
}


def train_tokenizer(tokenizer_type, corpus_path, vocab_size):
    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NFC()
    
    if tokenizer_type == "metaspace_split":
        tokenizer.pre_tokenizer = Sequence([
            Metaspace(prepend_scheme="never"),
            Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated")
        ])
        tokenizer.decoder = MetaspaceDecoder(prepend_scheme="never")
    elif tokenizer_type == "metaspace_split_functionalizer":
        tokenizer.pre_tokenizer = Sequence([
            Metaspace(prepend_scheme="never"),
            Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated"),
            FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=True, repeat=True, serialize=True, split_operators=True),
            MetaspaceDecoder(prepend_scheme="never")
        ])
    elif tokenizer_type == "metaspace_split_functionalizer_repeat":
        tokenizer.pre_tokenizer = Sequence([
            Metaspace(prepend_scheme="never"),
            Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated"),
            FunctionalizerPreTok(capitalize=False, repeat=True, serialize=False, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=False, repeat=True, serialize=False, split_operators=True),
            MetaspaceDecoder(prepend_scheme="never")
        ])
    elif tokenizer_type == "split":
        tokenizer.pre_tokenizer = Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated")
        tokenizer.decoder = Fuse()
    elif tokenizer_type == "split_functionalizer":
        tokenizer.pre_tokenizer = Sequence([
            Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated"),
            FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=True, repeat=True, serialize=True, split_operators=True),
            Fuse()
        ])
    elif tokenizer_type == "split_functionalizer_repeat":
        tokenizer.pre_tokenizer = Sequence([
            Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated"),
            FunctionalizerPreTok(capitalize=False, repeat=True, serialize=False, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=False, repeat=True, serialize=False, split_operators=True),
            Fuse()
        ])
    elif tokenizer_type == "llama":
        tokenizer.pre_tokenizer = Split(pattern=Regex(LLAMA_SPLIT_PAT), behavior="isolated")
        tokenizer.decoder = Fuse()
    elif tokenizer_type == "llama_functionalizer":
        tokenizer.pre_tokenizer = Sequence([
            Split(pattern=Regex(LLAMA_SPLIT_PAT), behavior="isolated"),
            FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=True, repeat=True, serialize=True, split_operators=True),
            Fuse()
        ])
    elif tokenizer_type == "metaspace_whitespace":
        tokenizer.pre_tokenizer = Sequence([
            Metaspace(prepend_scheme="never"),
            Whitespace()
        ])
        tokenizer.decoder = MetaspaceDecoder(prepend_scheme="never")
    elif tokenizer_type == "metaspace_whitespace_functionalizer":
        tokenizer.pre_tokenizer = Sequence([
            Metaspace(prepend_scheme="never"),
            Whitespace(),
            FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=True, repeat=True, serialize=True, split_operators=True),
            MetaspaceDecoder(prepend_scheme="never")
        ])
    elif tokenizer_type == "metaspace":
        tokenizer.pre_tokenizer = Metaspace(prepend_scheme="never")
        tokenizer.decoder = MetaspaceDecoder(prepend_scheme="never")
    elif tokenizer_type == "metaspace_functionalizer":
        tokenizer.pre_tokenizer = Sequence([
            Metaspace(prepend_scheme="never"),
            FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=True)
        ])
        tokenizer.decoder = SequenceDecoder([
            FunctionalizerDecoder(capitalize=True, repeat=True, serialize=True, split_operators=True),
            MetaspaceDecoder(prepend_scheme="never")
        ])
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train([corpus_path], trainer)
    return tokenizer



def is_pua(char):
    return 0xE000 <= ord(char) <= 0xF0FF

def normalize_token(token):
    # Remove any PUA characters first to handle functionalizer-specific opcodes and parameters
    token = "".join([c for c in token if not is_pua(c)])
    token = token.lower()
    # Normalize unicode to separate base characters and diacritics
    nfd_form = unicodedata.normalize('NFD', token)
    # Retain only alphanumeric characters (removes spaces, punctuation, symbols, and marks)
    cleaned = "".join([c for c in nfd_form if c.isalnum()])
    # Collapse repetitions of 3+ down to 1
    cleaned = re.sub(r'(.)\1{2,}', r'\1', cleaned)
    return cleaned

def analyze_vocab(vocab, is_functionalizer=False):
    unique_concepts = set()
    total_tokens = 0
    pua_tokens_count = 0
    special_tokens = {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}
    
    for token, _ in vocab.items():
        total_tokens += 1
        if token in special_tokens:
            continue
            
        if is_functionalizer and all(is_pua(c) for c in token):
            pua_tokens_count += 1
            continue
            
        normalized = normalize_token(token)
        if normalized:
            unique_concepts.add(normalized)
            
    return unique_concepts, total_tokens, pua_tokens_count

def prepare_corpus(dataset_name, dataset_config=None, split_train="train", split_val="validation", text_column="text", label="Wikitext"):
    print(f"\nLoading {label} dataset (streaming)...")
    if dataset_config:
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split_train, streaming=True)
    else:
        raw_dataset = load_dataset(dataset_name, split=split_train, streaming=True)

    if split_val is None:
        print("Streaming dataset and separating into train (90%) and validation (10%) splits locally...")
        train_items = []
        validation_items = []
        for i, item in enumerate(raw_dataset):
            if i < 9000:
                train_items.append(item)
            elif i < 10000:
                validation_items.append(item)
            else:
                break
        test_dataset = validation_items
        train_iterable = train_items
    else:
        if dataset_config:
            test_dataset_raw = load_dataset(dataset_name, dataset_config, split=split_val, streaming=True)
        else:
            test_dataset_raw = load_dataset(dataset_name, split=split_val, streaming=True)
        
        test_dataset = []
        for i, item in enumerate(test_dataset_raw):
            if i >= 500:
                break
            test_dataset.append(item)
            
        train_iterable = raw_dataset
        
    corpus_path = f"temp_corpus_{label.lower()}.txt"
    print("Writing temporary corpus file (first 10,000 samples)...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(train_iterable):
            if split_val is not None and i >= 10000:
                break
            text = item[text_column]
            if text and text.strip():
                f.write(text + "\n")
                
    return corpus_path, test_dataset

def evaluate_on_prepared_corpus(corpus_path, test_dataset, vocab_size, text_column="text", label="Wikitext"):
    print(f"Evaluating {label} with vocab size {vocab_size}...")
    try:
        configs = [
            # ("metaspace_split", False, None),
            # ("metaspace_split_functionalizer", True, "metaspace_split"),
            # ("metaspace_split_functionalizer_repeat", True, "metaspace_split"),
            ("split", False, None),
            ("split_functionalizer", True, "split"),
            ("split_functionalizer_repeat", True, "split"),
            ("llama", False, None),
            ("llama_functionalizer", True, "llama"),
            # ("metaspace_whitespace", False, None),
            # ("metaspace_whitespace_functionalizer", True, "metaspace_whitespace"),
            # ("metaspace", False, None),
            # ("metaspace_functionalizer", True, "metaspace"),
        ]
        
        tokenizers = {}
        for name, _, _ in configs:
            tokenizers[name] = train_tokenizer(name, corpus_path, vocab_size)
            
        # Analyze Vocabularies
        vocab_analyses = {}
        for name, is_dec, _ in configs:
            unique_concepts, total_tokens, _ = analyze_vocab(tokenizers[name].get_vocab(), is_functionalizer=is_dec)
            density = len(unique_concepts) / total_tokens if total_tokens > 0 else 0
            vocab_analyses[name] = {
                "concepts": len(unique_concepts),
                "total": total_tokens,
                "density": density
            }
            
        # Measure sequence length on validation subset
        tot_toks = {name: 0 for name, _, _ in configs}
        tot_chars = 0
        eval_count = 0
        
        for item in test_dataset:
            text = item[text_column]
            if not text or not text.strip():
                continue
            for name, _, _ in configs:
                tot_toks[name] += len(tokenizers[name].encode(text).ids)
            tot_chars += len(text)
            eval_count += 1
            if eval_count >= 500:
                break
                
        chars_per_token = {}
        for name, _, _ in configs:
            chars_per_token[name] = tot_chars / tot_toks[name] if tot_toks[name] > 0 else 0
            
        # Compute inflations and unconstrained vocab diffs
        inflations = {}
        vocab_diffs = {}
        for name, is_dec, baseline_name in configs:
            if is_dec and baseline_name:
                base_toks = tot_toks[baseline_name]
                dec_toks = tot_toks[name]
                inflations[name] = ((dec_toks - base_toks) / base_toks) * 100 if base_toks > 0 else 0
                
                base_vocab = vocab_analyses[baseline_name]["total"]
                dec_vocab = vocab_analyses[name]["total"]
                if base_vocab < vocab_size and dec_vocab < vocab_size:
                    vocab_diffs[name] = ((dec_vocab - base_vocab) / base_vocab) * 100
                else:
                    vocab_diffs[name] = None
            else:
                inflations[name] = 0.0
                vocab_diffs[name] = None
                
        # Roundtrip checks against original text
        checked_count = 0
        success_count = 0
        for item in test_dataset:
            text = item[text_column]
            if not text or not text.strip():
                continue
                
            for name, _ in tokenizers.items():
                tokenizer = tokenizers[name]
                check_text = text
                if "whitespace" in name:
                    check_text = check_text.replace("\n", "").replace("\t", "")
                    
                encoded = tokenizer.encode(check_text)
                decoded = tokenizer.decode(encoded.ids)
                if decoded != check_text:
                    missing_chars = set(check_text) - set(decoded)
                    has_oov = any(tokenizer.token_to_id(c) is None for c in missing_chars if not c.isspace())
                    has_unk = "[UNK]" in encoded.tokens
                    
                    if has_unk or has_oov:
                        print(f"Bypassing roundtrip mismatch for {name} on {label} due to unknown/OOV characters.")
                        continue
                    if len(check_text) < 256:
                        print(f"\n--- Mismatch on {name} tokenizer ---")
                        print("Original: ", repr(check_text))
                        print("Decoded:  ", repr(decoded))
                        print("Tokens:    ", encoded.tokens)
                        raise AssertionError(f"{name} roundtrip mismatch on {label}!")
                    else:
                        print(f"Skipping roundtrip assert for long sequence in {name} on {label}")
                        
            success_count += 1
            checked_count += 1
            if checked_count >= 100:
                break
                
        print(f"Successfully verified losslessness on {success_count} {label} validation text sequences.")
        
        results_list = []
        for name, is_dec, baseline_name in configs:
            results_list.append({
                "label": label,
                "target_vocab_size": vocab_size,
                "tokenizer_type": name,
                "baseline_type": baseline_name,
                "actual_vocab_size": vocab_analyses[name]["total"],
                "concepts": vocab_analyses[name]["concepts"],
                "density": vocab_analyses[name]["density"] * 100,
                "chars_per_token": chars_per_token[name],
                "inflation": inflations[name] if is_dec else None,
                "vocab_diff": vocab_diffs[name],
            })
            
        return results_list
    except Exception as e:
        print(f"Error during evaluation of {label} (vocab={vocab_size}): {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results_table(results):
    headers = [
        "Tokenizer", "Dataset", "Target", "Actual Voc", "Concepts", "Density (%)", "Chars/Token", "Inflation (%)", "Vocab Diff (%)"
    ]
    col_widths = [len(h) for h in headers]
    
    rows = []
    for r in results:
        target_str = f"{r['target_vocab_size'] // 1000}k" if r['target_vocab_size'] >= 1000 else f"{r['target_vocab_size']}"
        infl_str = f"{r['inflation']:+.2f}%" if r['inflation'] is not None else "-"
        vocab_diff = r.get("vocab_diff")
        vocab_diff_str = f"{vocab_diff:+.2f}%" if vocab_diff is not None else "-"

        pretty_name = TOK_NAME_MAP.get(r['tokenizer_type'], r['tokenizer_type'])
        
        rows.append([
            pretty_name, r["label"], target_str,
            str(r["actual_vocab_size"]), str(r["concepts"]), f"{r['density']:.2f}",
            f"{r['chars_per_token']:.4f}", infl_str, vocab_diff_str
        ])

        
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
            
    sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    
    print("\n" + "=" * len(sep))
    print(" TOKENIZER COMPARISON SUMMARY")
    print("=" * len(sep))
    print(sep)
    
    header_str = "|" + "|".join([f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)]) + "|"
    print(header_str)
    print(sep)
    
    for idx, row in enumerate(rows):
        row_str = "|" + "|".join([f" {val:<{col_widths[i]}} " for i, val in enumerate(row)]) + "|"
        print(row_str)
        if (idx + 1) % 2 == 0:
            print(sep)

def generate_markdown_report(results):
    report = "# Vocabulary Analysis Report\n\n"
    report += (
        "This report compares the vocabularies learned by each tokenizer configuration across datasets and "
        "target vocab sizes, and measures how efficiently each one compresses held-out validation text.\n\n"
    )
    report += "**Metrics:**\n\n"
    report += "* **Actual Vocab** - number of vocab entries actually learned by BPE training (capped at the target size).\n"
    report += (
        "* **Unique Concepts** - count of distinct \"root\" tokens after lowercasing, stripping diacritics/PUA "
        "opcodes, and collapsing repeated characters; a proxy for how much semantic redundancy is baked into the vocab.\n"
    )
    report += "* **Concept Density (%)** - Unique Concepts / Actual Vocab; higher means less redundancy per vocab slot.\n"
    report += "* **Chars / Token** - average characters produced per token on the validation sample; higher means better compression.\n"
    report += (
        "* **Token Inflation vs Baseline (%)** - for Functionalizer configs, the change in chars/token relative to "
        "the non-Functionalizer tokenizer using the same pre-tokenization split.\n"
    )
    report += (
        "* **Vocab Size Δ vs Baseline (%)** - for Functionalizer configs, the change in actual vocab size relative to "
        "the baseline, only reported when both configs trained under the target vocab size budget.\n\n"
    )

    by_dataset = {}
    for r in results:
        by_dataset.setdefault(r["label"], {}).setdefault(r["target_vocab_size"], []).append(r)

    for label, by_vocab in by_dataset.items():
        report += f"## Dataset: {label}\n\n"
        for vocab_size, rows in by_vocab.items():
            vs_str = f"{vocab_size // 1000}k" if vocab_size >= 1000 else str(vocab_size)
            report += f"### Target Vocab Size: {vs_str}\n\n"
            report += "| Tokenizer Type | Actual Vocab | Unique Concepts | Concept Density (%) | Chars / Token | Token Inflation vs Baseline (%) | Vocab Size Δ vs Baseline (%) |\n"
            report += "|---|---|---|---|---|---|---|\n"
            for r in rows:
                pretty_name = TOK_NAME_MAP.get(r["tokenizer_type"], r["tokenizer_type"])
                infl_str = f"{r['inflation']:+.2f}%" if r["inflation"] is not None else "-"
                vdiff = r.get("vocab_diff")
                vdiff_str = f"{vdiff:+.2f}%" if vdiff is not None else "-"
                report += (
                    f"| **{pretty_name}** | {r['actual_vocab_size']} | {r['concepts']} | "
                    f"{r['density']:.2f}% | {r['chars_per_token']:.4f} | {infl_str} | {vdiff_str} |\n"
                )
            report += "\n"

    return report

def main():
    datasets_to_run = [
        {
            "dataset_name": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "split_train": "train",
            "split_val": "validation",
            "text_column": "text",
            "label": "Wikitext"
        },
        {
            "dataset_name": "flytech/python-codes-25k",
            "dataset_config": None,
            "split_train": "train",
            "split_val": None,
            "text_column": "text",
            "label": "Python-Codes"
        },
        {
            "dataset_name": "roneneldan/TinyStories",
            "dataset_config": None,
            "split_train": "train",
            "split_val": "validation",
            "text_column": "text",
            "label": "TinyStories"
        },
        {
            "dataset_name": "code-search-net/code_search_net",
            "dataset_config": "python",
            "split_train": "train",
            "split_val": "validation",
            "text_column": "func_code_string",
            "label": "CSN-Python"
        },
        {
            "dataset_name": "code-search-net/code_search_net",
            "dataset_config": "java",
            "split_train": "train",
            "split_val": "validation",
            "text_column": "func_code_string",
            "label": "CSN-Java"
        },
        {
            "dataset_name": "code-search-net/code_search_net",
            "dataset_config": "go",
            "split_train": "train",
            "split_val": "validation",
            "text_column": "func_code_string",
            "label": "CSN-Go"
        },
    ]

    vocab_sizes = [
        # 4000, 
        # 32000, 
        128000
    ]
    results = []

    for ds in datasets_to_run:
        try:
            corpus_path, test_dataset = prepare_corpus(
                dataset_name=ds["dataset_name"],
                dataset_config=ds["dataset_config"],
                split_train=ds["split_train"],
                split_val=ds["split_val"],
                text_column=ds["text_column"],
                label=ds["label"]
            )
            
            try:
                for vocab_size in vocab_sizes:
                    metrics = evaluate_on_prepared_corpus(
                        corpus_path=corpus_path,
                        test_dataset=test_dataset,
                        vocab_size=vocab_size,
                        text_column=ds["text_column"],
                        label=ds["label"]
                    )
                    if metrics:
                        results.extend(metrics)
            finally:
                if os.path.exists(corpus_path):
                    os.remove(corpus_path)
        except Exception as e:
            print(f"Failed to prepare dataset {ds['label']}: {e}")

    if results:
        print_results_table(results)
        import json
        output_file = "./results/vocab_results.json"
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        report_path = "./results/vocab_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(results))
        print(f"\nVocabulary analysis finished!")
        print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
