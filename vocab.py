import os
import re
import json
import argparse
import unicodedata
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Metaspace, Whitespace
from tokenizers.pre_tokenizers import Functionalizer as FunctionalizerPreTok
from tokenizers.decoders import Functionalizer as FunctionalizerDecoder, Metaspace as MetaspaceDecoder, Sequence as SequenceDecoder, Fuse
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC

DEFAULT_SPLIT_PAT = r"\n+|\t+|\u2581+| +|[^\p{L}\p{N}\s _]+"
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


def train_tokenizer(tokenizer_type, corpus_iter, vocab_size, length=None):
    """
    `corpus_iter` is any iterator over strings (see train_from_iterator below).
    """
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
    # Train straight from an iterator instead of a temp corpus file. The file was never
    # required -- it existed only because tokenizer.train() takes paths -- and at full-corpus
    # scale it meant writing a multi-GB duplicate of data already cached on disk, with no
    # progress output while it happened. `length` gives the trainer a real progress bar.
    tokenizer.train_from_iterator(corpus_iter, trainer, length=length)
    return tokenizer



# Unicode Private Use Area block used by the Functionalizer for opcodes and parameters.
# Params occupy U+E000-U+E0FF and operators U+E100-U+EFFF, so the whole block is U+E000-U+EFFF.
# Keep this identical to downstream_tests.py -- the two files previously disagreed.
PUA_START = 0xE000
PUA_END = 0xEFFF


def is_pua(char):
    return PUA_START <= ord(char) <= PUA_END

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

def prepare_corpus(dataset_name, dataset_config=None, split_train="train", split_val="validation",
                   text_column="text", label="Wikitext", corpus_samples=1000000, eval_samples=5000,
                   shuffle_seed=42, shuffle_buffer=10000):
    """
    Build a REPEATABLE corpus source plus two evaluation sample sets.

    Returns (make_corpus_iter, val_texts, train_texts, corpus_doc_count), where
    `make_corpus_iter()` returns a fresh iterator over the corpus text. It must be repeatable
    because each dataset trains several tokenizers over the same corpus -- that requirement is
    the only reason a temp file ever existed.

    Two modes:

    * corpus_samples is None (full corpus): loads NON-streaming, so the split is downloaded once
      into the HuggingFace cache as memory-mapped Arrow. Re-iterating is then cheap and needs no
      extra RAM and no multi-GB temp text file, and the count is known up front so progress bars
      work. Subsequent runs reuse the cache instead of re-downloading.
    * corpus_samples set (capped): keeps streaming, so a small calibration run does not pull the
      entire split. The capped sample is small by definition, so it is held in memory.

    Note the corpus itself is NOT shuffled: BPE merge training counts word frequencies over the
    whole corpus and is order-independent, so shuffling it would be pure overhead. Shuffling
    matters only for the evaluation samples and for subsetting under a cap.
    """
    def _load_stream(split):
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_name, split=split, streaming=True)
        return ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)

    def _load_full(split):
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=split)
        return load_dataset(dataset_name, split=split)

    def _texts_from(ds, limit=None):
        out = []
        for batch in ds.iter(batch_size=1000):
            for text in batch[text_column]:
                if text and text.strip():
                    out.append(text)
                    if limit is not None and len(out) >= limit:
                        return out
        return out

    # ---------------- capped mode: stream, stay small ----------------
    if corpus_samples is not None:
        print(f"\nLoading {label} (streaming, capped at {corpus_samples:,} documents)...")
        train_stream = _load_stream(split_train)
        val_texts = []

        if split_val is None:
            print(f"No validation split for {label}; holding out {eval_samples:,} shuffled train documents.")
            stream_iter = iter(train_stream)
            for item in stream_iter:
                text = item[text_column]
                if text and text.strip():
                    val_texts.append(text)
                    if len(val_texts) >= eval_samples:
                        break
            corpus_source = stream_iter
        else:
            for item in _load_stream(split_val):
                text = item[text_column]
                if text and text.strip():
                    val_texts.append(text)
                    if len(val_texts) >= eval_samples:
                        break
            corpus_source = iter(train_stream)

        corpus_texts = []
        for item in tqdm(corpus_source, total=corpus_samples, desc=f"Collecting {label} corpus"):
            text = item[text_column]
            if not text or not text.strip():
                continue
            corpus_texts.append(text)
            if len(corpus_texts) >= corpus_samples:
                break

        train_texts = corpus_texts[:eval_samples]
        print(f"{label}: corpus={len(corpus_texts):,} docs | held-out eval={len(val_texts):,} | train eval={len(train_texts):,}")
        return (lambda: iter(corpus_texts)), val_texts, train_texts, len(corpus_texts)

    # ---------------- full mode: cached Arrow, memory-mapped ----------------
    print(f"\nLoading FULL {label} corpus (non-streaming; downloads once, then cached)...")
    train_ds = _load_full(split_train)

    if split_val is None:
        # Exact disjoint split by index -- no streaming-order guesswork.
        print(f"No validation split for {label}; holding out {eval_samples:,} shuffled train documents.")
        shuffled = train_ds.shuffle(seed=shuffle_seed)
        n_hold = min(eval_samples, len(shuffled))
        val_texts = _texts_from(shuffled.select(range(n_hold)), limit=eval_samples)
        corpus_ds = shuffled.select(range(n_hold, len(shuffled)))
    else:
        val_ds = _load_full(split_val)
        val_texts = _texts_from(val_ds.shuffle(seed=shuffle_seed), limit=eval_samples)
        corpus_ds = train_ds

    # Same-split companion to val_texts, so chars/token is reported on both bases.
    train_texts = _texts_from(
        corpus_ds.shuffle(seed=shuffle_seed).select(range(min(eval_samples, len(corpus_ds)))),
        limit=eval_samples,
    )

    corpus_docs = len(corpus_ds)

    def make_corpus_iter():
        # Fresh generator per tokenizer; reads lazily from memory-mapped Arrow.
        for batch in corpus_ds.iter(batch_size=1000):
            for text in batch[text_column]:
                if text and text.strip():
                    yield text

    approx_gb = corpus_ds.dataset_size / (1024 ** 3) if corpus_ds.dataset_size else None
    size_note = f" (~{approx_gb:.2f} GB on disk)" if approx_gb else ""
    print(f"{label}: corpus={corpus_docs:,} docs{size_note} | held-out eval={len(val_texts):,} | train eval={len(train_texts):,}")
    return make_corpus_iter, val_texts, train_texts, corpus_docs

def evaluate_on_prepared_corpus(make_corpus_iter, val_texts, train_texts, vocab_size, label="Wikitext",
                                corpus_docs=None, roundtrip_samples=200):
    print(f"Evaluating {label} with vocab size {vocab_size}...")
    try:
        configs = [
            # ("metaspace_split", False, None),
            # ("metaspace_split_functionalizer", True, "metaspace_split"),
            # ("metaspace_split_functionalizer_repeat", True, "metaspace_split"),
            # ("split", False, None),
            # ("split_functionalizer", True, "split"),
            # ("split_functionalizer_repeat", True, "split"),
            ("llama", False, None),
            ("llama_functionalizer", True, "llama"),
            # ("metaspace_whitespace", False, None),
            # ("metaspace_whitespace_functionalizer", True, "metaspace_whitespace"),
            # ("metaspace", False, None),
            # ("metaspace_functionalizer", True, "metaspace"),
        ]
        
        tokenizers = {}
        for name, _, _ in configs:
            print(f"  Training {name} on {corpus_docs:,} documents...")
            # Fresh iterator per tokenizer -- this is why the source must be repeatable.
            tokenizers[name] = train_tokenizer(name, make_corpus_iter(), vocab_size, length=corpus_docs)
            
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
            
        # Measure chars/token on BOTH the held-out split and the training split.
        # Reporting both makes the train-vs-val gap directly visible instead of leaving it as an
        # unexplained discrepancy against the training pipeline (which measures on train text).
        def measure_chars_per_token(texts):
            tot_toks = {name: 0 for name, _, _ in configs}
            tot_chars = 0
            for text in texts:
                if not text or not text.strip():
                    continue
                for name, _, _ in configs:
                    tot_toks[name] += len(tokenizers[name].encode(text).ids)
                tot_chars += len(text)
            return {
                name: (tot_chars / tot_toks[name] if tot_toks[name] > 0 else 0)
                for name, _, _ in configs
            }

        chars_per_token = measure_chars_per_token(val_texts)
        chars_per_token_train = measure_chars_per_token(train_texts)


        # Compute inflations, unconstrained vocab diffs, and density diffs
        inflations = {}
        vocab_diffs = {}
        density_diffs = {}
        for name, is_dec, baseline_name in configs:
            if is_dec and baseline_name:
                base_cpt = chars_per_token[baseline_name]
                cpt = chars_per_token[name]
                inflations[name] = ((cpt / base_cpt - 1) * 100) if base_cpt > 0 else 0
                
                base_vocab = vocab_analyses[baseline_name]["total"]
                dec_vocab = vocab_analyses[name]["total"]
                if base_vocab < vocab_size and dec_vocab < vocab_size:
                    vocab_diffs[name] = ((dec_vocab - base_vocab) / base_vocab) * 100
                else:
                    vocab_diffs[name] = None

                base_density = vocab_analyses[baseline_name]["density"]
                density = vocab_analyses[name]["density"]
                density_diffs[name] = ((density / base_density - 1) * 100) if base_density > 0 else 0
            else:
                inflations[name] = 0.0
                vocab_diffs[name] = None
                density_diffs[name] = None
                
        # Roundtrip (losslessness) verification.
        #
        # Two prior weaknesses are fixed here:
        #  1. Sequences >=256 chars skipped the assertion entirely -- i.e. most of Wikitext and
        #     nearly every CSN function went unchecked while still being counted as "verified".
        #     The length guard is gone; every sampled sequence is now compared in full.
        #  2. The comparison targeted the raw text, but every config applies NFC normalization,
        #     which is not invertible. Losslessness is guaranteed only up to NFC, so that is what
        #     we assert; comparing against raw text produced spurious mismatches on decomposed
        #     input. verified/bypassed counts are tracked separately and reported honestly.
        roundtrip_stats = {name: {"verified": 0, "failed": 0, "bypassed_oov": 0} for name, _, _ in configs}

        for text in val_texts[:roundtrip_samples]:
            if not text or not text.strip():
                continue

            for name, tokenizer in tokenizers.items():
                check_text = text
                if "whitespace" in name:
                    check_text = check_text.replace("\n", "").replace("\t", "")
                # The pipeline is lossless with respect to its NFC-normalized input.
                expected = unicodedata.normalize("NFC", check_text)

                encoded = tokenizer.encode(check_text)
                decoded = tokenizer.decode(encoded.ids)
                if decoded != expected:
                    missing_chars = set(expected) - set(decoded)
                    has_oov = any(tokenizer.token_to_id(c) is None for c in missing_chars if not c.isspace())
                    has_unk = "[UNK]" in encoded.tokens

                    if has_unk or has_oov:
                        roundtrip_stats[name]["bypassed_oov"] += 1
                        continue

                    roundtrip_stats[name]["failed"] += 1
                    print(f"\n--- Mismatch on {name} tokenizer ({label}, len={len(expected)}) ---")
                    print("Original (NFC): ", repr(expected[:512]))
                    print("Decoded:        ", repr(decoded[:512]))
                    print("Tokens:         ", encoded.tokens[:64])
                    continue

                roundtrip_stats[name]["verified"] += 1

        for name, _, _ in configs:
            st = roundtrip_stats[name]
            print(
                f"Losslessness [{label} / {name}]: verified={st['verified']} "
                f"failed={st['failed']} "
                f"bypassed(OOV/UNK)={st['bypassed_oov']}"
            )


        results_list = []
        for name, is_dec, baseline_name in configs:
            actual = vocab_analyses[name]["total"]
            results_list.append({
                "label": label,
                "target_vocab_size": vocab_size,
                "corpus_docs": corpus_docs,
                "tokenizer_type": name,
                "baseline_type": baseline_name,
                "actual_vocab_size": actual,
                # True only when BPE ran out of merges before hitting the target, i.e. the
                # unconstrained regime the vocab-reduction claim depends on.
                "exhausted": actual < vocab_size,
                "concepts": vocab_analyses[name]["concepts"],
                "density": vocab_analyses[name]["density"] * 100,
                "density_diff": density_diffs[name] if is_dec else None,
                "chars_per_token": chars_per_token[name],
                "chars_per_token_train": chars_per_token_train[name],
                "roundtrip_verified": roundtrip_stats[name]["verified"],
                "roundtrip_failed": roundtrip_stats[name]["failed"],
                "roundtrip_bypassed_oov": roundtrip_stats[name]["bypassed_oov"],
                "chars_per_token_diff": inflations[name] if is_dec else None,
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
        "Tokenizer", "Dataset", "Target", "Actual Voc", "Exhausted", "Concepts", "Density (%)",
        "Chars/Tok (val)", "Chars/Tok (train)", "Chars/Tok Δ (%)", "Density Δ (%)", "Vocab Diff (%)"
    ]
    col_widths = [len(h) for h in headers]

    rows = []
    for r in results:
        target_str = f"{r['target_vocab_size'] // 1000}k" if r['target_vocab_size'] >= 1000 else f"{r['target_vocab_size']}"
        diff = r.get("chars_per_token_diff")
        infl_str = f"{diff:+.2f}%" if diff is not None else "-"
        vocab_diff = r.get("vocab_diff")
        vocab_diff_str = f"{vocab_diff:+.2f}%" if vocab_diff is not None else "-"
        dens_diff = r.get("density_diff")
        dens_diff_str = f"{dens_diff:+.2f}%" if dens_diff is not None else "-"

        pretty_name = TOK_NAME_MAP.get(r['tokenizer_type'], r['tokenizer_type'])

        rows.append([
            pretty_name, r["label"], target_str,
            str(r["actual_vocab_size"]), "yes" if r.get("exhausted") else "NO",
            str(r["concepts"]), f"{r['density']:.2f}",
            f"{r['chars_per_token']:.4f}", f"{r.get('chars_per_token_train', 0):.4f}",
            infl_str, dens_diff_str, vocab_diff_str
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
    report += "* **Density Δ vs Baseline (%)** - for Functionalizer configs, the relative percentage change in concept density relative to the baseline tokenizer using the same pre-tokenization split.\n"
    report += "* **Exhausted** - whether BPE ran out of merges before reaching the target vocab size. The vocabulary-reduction claim is only meaningful when this is `yes` for BOTH configs; a `NO` means the run was capped by the target and Vocab Size Δ is suppressed.\n"
    report += "* **Chars / Token (val)** - average characters per token on the held-out sample; higher means better compression.\n"
    report += "* **Chars / Token (train)** - the same measure on a sample of the tokenizer's own training text. Reported alongside the held-out figure so any train/held-out gap is visible directly rather than surfacing later as an unexplained discrepancy against the training pipeline.\n"
    report += (
        "* **Chars/Token Δ vs Baseline (%)** - for Functionalizer configs, the relative change in character density "
        "relative to the non-Functionalizer tokenizer using the same pre-tokenization split (positive = compression gain, negative = token inflation).\n"
    )
    report += (
        "* **Vocab Size Δ vs Baseline (%)** - for Functionalizer configs, the change in actual vocab size relative to "
        "the baseline, only reported when both configs exhausted below the target vocab size budget.\n"
    )
    report += "* **Roundtrip** - counts of (verified / failed / bypassed) sequences. Verified decode matched the NFC-normalized input exactly, failed sequences had decoding mismatches without OOV/UNK, and bypassed sequences contained characters out-of-vocabulary or unknown to the tokenizer.\n\n"

    by_dataset = {}
    for r in results:
        by_dataset.setdefault(r["label"], {}).setdefault(r["target_vocab_size"], []).append(r)

    for label, by_vocab in by_dataset.items():
        report += f"## Dataset: {label}\n\n"
        for vocab_size, rows in by_vocab.items():
            vs_str = f"{vocab_size // 1000}k" if vocab_size >= 1000 else str(vocab_size)
            report += f"### Target Vocab Size: {vs_str}\n\n"
            docs = rows[0].get("corpus_docs")
            if docs:
                report += f"Tokenizer training corpus: **{docs:,} documents**.\n\n"
            if not all(r.get("exhausted") for r in rows):
                report += (
                    "> WARN: **Not all configs exhausted below the target vocab size.** This run is in the "
                    "constrained regime, so the vocabulary-reduction comparison does not apply here; "
                    "raise the target vocab size until both configs exhaust.\n\n"
                )
            report += "| Tokenizer Type | Actual Vocab | Exhausted | Unique Concepts | Concept Density (%) | Chars / Token (val) | Chars / Token (train) | Chars/Token Δ vs Baseline (%) | Density Δ vs Baseline (%) | Vocab Size Δ vs Baseline (%) | Roundtrip (verified / failed / bypassed) |\n"
            report += "|---|---|---|---|---|---|---|---|---|---|---|\n"
            for r in rows:
                pretty_name = TOK_NAME_MAP.get(r["tokenizer_type"], r["tokenizer_type"])
                diff = r.get("chars_per_token_diff")
                infl_str = f"{diff:+.2f}%" if diff is not None else "-"
                vdiff = r.get("vocab_diff")
                vdiff_str = f"{vdiff:+.2f}%" if vdiff is not None else "-"
                dens_diff = r.get("density_diff")
                dens_diff_str = f"{dens_diff:+.2f}%" if dens_diff is not None else "-"
                report += (
                    f"| **{pretty_name}** | {r['actual_vocab_size']} | {'yes' if r.get('exhausted') else '**NO**'} | "
                    f"{r['concepts']} | {r['density']:.2f}% | {r['chars_per_token']:.4f} | "
                    f"{r.get('chars_per_token_train', 0):.4f} | {infl_str} | {dens_diff_str} | {vdiff_str} | "
                    f"{r.get('roundtrip_verified', 0)} / {r.get('roundtrip_failed', 0)} / {r.get('roundtrip_bypassed_oov', 0)} |\n"
                )
            report += "\n"

    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_samples", type=int, default=100_000,
        help="Documents used to train each tokenizer. Default: 1,000,000 documents (streaming mode). "
             "Pass a different value or None for full corpus mode."
    )
    parser.add_argument(
        "--eval_samples", type=int, default=5_000,
        help="Shuffled documents used to measure chars/token on each of the held-out and train bases."
    )
    parser.add_argument(
        "--roundtrip_samples", type=int, default=200,
        help="Held-out documents to verify losslessly roundtrip (full length, no length cutoff)."
    )
    parser.add_argument(
        "--vocab_sizes", type=int, nargs="+", default=[4_096_000],
        help="Target vocab size(s). Raise this if a dataset fails to exhaust below the target."
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer", type=int, default=10000)
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Subset of dataset labels to run (e.g. --datasets TinyStories CSN-Go). Default: all."
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--run_tag", type=str, default="",
        help="Suffix for output files, so a new run does not overwrite existing results."
    )
    args = parser.parse_args()

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""

    if args.corpus_samples is None:
        print(
            "\n*** --corpus_samples not set: training each tokenizer on the FULL training split.\n"
            "    This is what makes 'corpus exhaustion' accurate, but it is substantially slower\n"
            "    and more memory-hungry than a capped run. Use --corpus_samples to cap it, and\n"
            "    consider calibrating on one dataset first via --datasets.\n"
        )

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
        # {
        #     "dataset_name": "roneneldan/TinyStories",
        #     "dataset_config": None,
        #     "split_train": "train",
        #     "split_val": "validation",
        #     "text_column": "text",
        #     "label": "TinyStories"
        # },
        # {
        #     "dataset_name": "code-search-net/code_search_net",
        #     "dataset_config": "python",
        #     "split_train": "train",
        #     "split_val": "validation",
        #     "text_column": "func_code_string",
        #     "label": "CSN-Python"
        # },
        # {
        #     "dataset_name": "code-search-net/code_search_net",
        #     "dataset_config": "java",
        #     "split_train": "train",
        #     "split_val": "validation",
        #     "text_column": "func_code_string",
        #     "label": "CSN-Java"
        # },
        # {
        #     "dataset_name": "code-search-net/code_search_net",
        #     "dataset_config": "go",
        #     "split_train": "train",
        #     "split_val": "validation",
        #     "text_column": "func_code_string",
        #     "label": "CSN-Go"
        # },
        {
            "dataset_name": "HuggingFaceFW/fineweb-edu",
            "dataset_config": "sample-10BT",
            "split_train": "train",
            "split_val": None,
            "text_column": "text",
            "label": "FineWeb-Edu"
        },
        {
            "dataset_name": "hasankursun/github-code-2025-language-split",
            "dataset_config": "python",
            "split_train": "train",
            "split_val": None,
            "text_column": "content",
            "label": "GitHub-Code-Python"
        },
        # {
        #     "dataset_name": "hasankursun/github-code-2025-language-split",
        #     "dataset_config": "go",
        #     "split_train": "train",
        #     "split_val": None,
        #     "text_column": "content",
        #     "label": "GitHub-Code-Go"
        # },
    ]

    if args.datasets:
        wanted = set(args.datasets)
        unknown = wanted - {d["label"] for d in datasets_to_run}
        if unknown:
            raise SystemExit(f"Unknown dataset label(s): {sorted(unknown)}")
        datasets_to_run = [d for d in datasets_to_run if d["label"] in wanted]

    vocab_sizes = args.vocab_sizes
    results = []

    for ds in datasets_to_run:
        try:
            make_corpus_iter, val_texts, train_texts, corpus_docs = prepare_corpus(
                dataset_name=ds["dataset_name"],
                dataset_config=ds["dataset_config"],
                split_train=ds["split_train"],
                split_val=ds["split_val"],
                text_column=ds["text_column"],
                label=ds["label"],
                corpus_samples=args.corpus_samples,
                eval_samples=args.eval_samples,
                shuffle_seed=args.shuffle_seed,
                shuffle_buffer=args.shuffle_buffer,
            )

            for vocab_size in vocab_sizes:
                metrics = evaluate_on_prepared_corpus(
                    make_corpus_iter=make_corpus_iter,
                    val_texts=val_texts,
                    train_texts=train_texts,
                    vocab_size=vocab_size,
                    label=ds["label"],
                    corpus_docs=corpus_docs,
                    roundtrip_samples=args.roundtrip_samples,
                )
                if metrics:
                    results.extend(metrics)
            # No temp corpus file to clean up any more.
        except Exception as e:
            print(f"Failed to prepare dataset {ds['label']}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        print_results_table(results)
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"vocab_results{run_suffix}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        report_path = os.path.join(args.output_dir, f"vocab_report{run_suffix}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(results))

        not_exhausted = [r for r in results if not r.get("exhausted")]
        if not_exhausted:
            print("\n*** WARNING: these configs hit the target vocab size instead of exhausting:")
            for r in not_exhausted:
                print(f"      {r['label']} / {r['tokenizer_type']} "
                      f"({r['actual_vocab_size']} == target {r['target_vocab_size']})")
            print("    Vocab Size Δ is suppressed for them. Re-run with a larger --vocab_sizes.")

        print(f"\nVocabulary analysis finished!")
        print(f"Results saved to {output_file}")
        print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
