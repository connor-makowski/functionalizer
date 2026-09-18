# The local `tokenizers` build is a source checkout of the functionalizer branch and reports a
# dev version (0.23.2.dev0) that falls outside the range transformers pins (>=0.22.0,<=0.23.0).
# transformers hard-fails at import on that check, so we report a compatible version instead.
# This suppresses ONLY the version-range assertion -- if the tokenizers API ever actually breaks,
# this mock will hide it, so re-verify after upgrading either package.
import importlib.metadata
orig_version = importlib.metadata.version
def mock_version(package_name):
    if package_name == "tokenizers":
        return "0.22.2"
    return orig_version(package_name)
importlib.metadata.version = mock_version

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset


from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Metaspace, Whitespace
from tokenizers.pre_tokenizers import Functionalizer as FunctionalizerPreTok
from tokenizers.decoders import Functionalizer as FunctionalizerDecoder, Sequence as SequenceDecoder, Metaspace as MetaspaceDecoder, Fuse
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFC
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup

# Seeding is set dynamically in main() based on the --seed flag

DEFAULT_SPLIT_PAT = r"\n+|\t+|\u2581+| +|[^\p{L}\p{N}\s _]+"
LLAMA_SPLIT_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def train_tokenizer(tokenizer_type, corpus_texts, vocab_size):
    """`corpus_texts` is any iterable of strings."""
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
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    # The texts are already in memory, so round-tripping them through a temp file bought nothing.
    tokenizer.train_from_iterator(corpus_texts, trainer, length=len(corpus_texts))
    return tokenizer



class TextDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = torch.as_tensor(chunks, dtype=torch.long)
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        # In causal language modeling, we predict the next token.
        # We can return the chunk, and split into input and target in the training loop
        return self.chunks[idx]

def load_dataset_samples(dataset_name, dataset_config, split, text_column, num_samples, data_dir=None, offset=0):
    print(f"Loading {num_samples} samples from {dataset_name} ({split}, data_dir={data_dir}, offset={offset})...")
    kwargs = {"streaming": True}
    if dataset_config is not None:
        kwargs["name"] = dataset_config
    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    has_only_train = dataset_name in ["HuggingFaceFW/fineweb-edu", "hasankursun/github-code-2025-language-split"]
    actual_split = "train" if (has_only_train and split == "validation") else split

    try:
        ds = load_dataset(dataset_name, split=actual_split, **kwargs)
    except Exception as e:
        if split == "validation":
            print(f"Split '{split}' not found for {dataset_name}, falling back to 'train' with offset {offset}...")
            actual_split = "train"
            ds = load_dataset(dataset_name, split="train", **kwargs)
        else:
            raise e

    # Apply offset only when pulling from the train split (e.g. holding out validation from train)
    if actual_split == "train" and offset > 0:
        ds = ds.skip(offset)
    samples = []
    for item in ds:
        text = item.get(text_column) if isinstance(item, dict) else None
        if text and text.strip():
            samples.append(text)
            if len(samples) >= num_samples:
                break
    return samples

def prepare_tokenized_data(samples, tokenizer, context_length, desc="",
                           max_example_tokens=None, max_example_chars=None,
                           batch_size=4096):
    """
    Tokenize using multi-threaded batch encoding, optionally drop over-long examples,
    then pack into fixed-length chunks.

    max_example_tokens drops any example whose own token count (excluding the appended [SEP])
    reaches the limit, so "less than N tokens" is strict.

    WARN: A token-count filter is TOKENIZER-DEPENDENT and therefore not arm-neutral: the same
    document can be 1,050 tokens under Split and 950 under Split+Functionalizer, so the two arms
    end up training on different document sets. The returned stats report exactly how many
    examples each arm dropped so the size of that asymmetry is visible. Use max_example_chars for
    a tokenizer-neutral filter that keeps both arms on an identical document set.
    """
    all_token_ids = []
    total_chars = 0
    sep_id = tokenizer.token_to_id("[SEP]")

    kept = 0
    dropped_by_chars = 0
    dropped_by_tokens = 0
    total_samples = len(samples)

    with tqdm(total=total_samples, desc=f"Tokenizing {desc}") as pbar:
        for i in range(0, total_samples, batch_size):
            batch_texts = samples[i:i + batch_size]

            # 1. Pre-filter by characters (tokenizer-neutral and avoids encoding dropped texts)
            if max_example_chars is not None:
                passed_texts = []
                for text in batch_texts:
                    if len(text) > max_example_chars:
                        dropped_by_chars += 1
                    else:
                        passed_texts.append(text)
            else:
                passed_texts = batch_texts

            if not passed_texts:
                pbar.update(len(batch_texts))
                continue

            # 2. Rust multi-threaded batch encoding across all CPU cores
            encodings = tokenizer.encode_batch(passed_texts)

            # 3. Post-filter by token count and collect IDs
            for text, enc in zip(passed_texts, encodings):
                ids = enc.ids
                if max_example_tokens is not None and len(ids) >= max_example_tokens:
                    dropped_by_tokens += 1
                    continue
                all_token_ids.extend(ids)
                all_token_ids.append(sep_id)
                # Count only visible characters. [SEP] is appended to the token stream but renders as
                # zero characters, so adding 1 here inflated chars/token for whichever arm packed more
                # documents per token -- an asymmetric bias between the two configurations.
                total_chars += len(text)
                kept += 1

            pbar.update(len(batch_texts))

    num_tokens = len(all_token_ids)
    num_chunks = num_tokens // context_length
    usable_tokens = num_chunks * context_length

    # Vectorized chunk packing into contiguous 2D tensor
    if usable_tokens > 0:
        arr = np.array(all_token_ids[:usable_tokens], dtype=np.int64)
        chunks = torch.from_numpy(arr.reshape(num_chunks, context_length))
    else:
        chunks = torch.empty((0, context_length), dtype=torch.long)

    total_seen = len(samples)
    stats = {
        "examples_seen": total_seen,
        "examples_kept": kept,
        "examples_dropped_by_tokens": dropped_by_tokens,
        "examples_dropped_by_chars": dropped_by_chars,
        "pct_examples_dropped": ((total_seen - kept) / total_seen * 100) if total_seen else 0.0,
    }
    if total_seen - kept:
        print(f"  [{desc}] kept {kept:,}/{total_seen:,} examples "
              f"({stats['pct_examples_dropped']:.2f}% dropped: "
              f"{dropped_by_tokens:,} over token limit, {dropped_by_chars:,} over char limit)")

    return chunks, num_tokens, total_chars, stats

def evaluate_loss(model, val_loader, device, max_batches=None):
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            if max_batches is not None and val_batches >= max_batches:
                break
            val_batch = val_batch.to(device)
            # Unshifted: the model shifts labels internally (see training loop).
            val_outputs = model(input_ids=val_batch, labels=val_batch)
            val_loss_sum += val_outputs.loss.item()
            val_batches += 1
    return (val_loss_sum / val_batches) if val_batches > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "tinystories", "csn-python", "csn-go",
        "fineweb-edu", "fineweb_edu", "fineweb",
        "github-code-python", "github_code_python",
        "github-code-go", "github_code_go",
    ])
    parser.add_argument("--tokenizer_type", type=str, required=True, choices=[
        "split", "split_functionalizer",
        "llama", "llama_functionalizer",
    ])
    parser.add_argument("--model_size", type=str, default="125M", choices=["25M", "125M", "2B"])
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=1000,
                        help="Maximum number of validation batches to evaluate during eval steps (default: 1000). Set to 0 or negative for unlimited.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument(
        "--max_example_tokens", type=int, default=None,
        help="Optional: drop training/validation examples whose own token count reaches this limit "
             "(strictly less than N is kept). Default None = train on all examples. NOTE: this "
             "filter is tokenizer-dependent, so enabling it can leave the two arms with different "
             "document sets; the results JSON records how many each dropped."
    )
    parser.add_argument(
        "--max_example_chars", type=int, default=None,
        help="Optional tokenizer-NEUTRAL length filter: drop inbound examples longer than this "
             "many characters. Default None = unlimited (train on all examples). Applied before "
             "tokenization, so both arms keep an identical document set. Pass 0 to disable."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_file", type=str, default="./results/results.json")
    # Suffix appended to checkpoint dirs so a re-run cannot resume or overwrite
    # checkpoints produced by an earlier version of the training objective.
    parser.add_argument("--run_tag", type=str, default="")
    args = parser.parse_args()

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""

    # `0` is the documented way to disable the length filters.
    if args.max_example_chars is not None and args.max_example_chars <= 0:
        args.max_example_chars = None
    if args.max_example_tokens is not None and args.max_example_tokens <= 0:
        args.max_example_tokens = None
    if args.max_val_batches is not None and args.max_val_batches <= 0:
        args.max_val_batches = None

    # Ensure reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set parameters dynamically based on model size class
    if args.model_size == "25M":
        max_steps = args.max_steps if args.max_steps is not None else 3000
        eval_steps = args.eval_steps if args.eval_steps is not None else 1000
        context_length = args.context_length if args.context_length is not None else 512
        lr = args.lr if args.lr is not None else 5e-4
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else 200
        grad_accum = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 1
        train_samples_count = args.train_samples if args.train_samples is not None else 50000
        val_samples_count = args.val_samples if args.val_samples is not None else 5000
        n_embd, n_layer, n_head = 512, 6, 8
    elif args.model_size == "125M":
        max_steps = args.max_steps if args.max_steps is not None else 50000
        eval_steps = args.eval_steps if args.eval_steps is not None else 5000
        context_length = args.context_length if args.context_length is not None else 512
        lr = args.lr if args.lr is not None else 4e-4
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else 1000
        grad_accum = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 2
        train_samples_count = args.train_samples if args.train_samples is not None else 500000
        val_samples_count = args.val_samples if args.val_samples is not None else 10000
        n_embd, n_layer, n_head = 768, 12, 12
    elif args.model_size == "2B":
        max_steps = args.max_steps if args.max_steps is not None else 100000
        eval_steps = args.eval_steps if args.eval_steps is not None else 5000
        context_length = args.context_length if args.context_length is not None else 1024
        lr = args.lr if args.lr is not None else 1.5e-4
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else 2000
        grad_accum = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 4
        train_samples_count = args.train_samples if args.train_samples is not None else 2000000
        val_samples_count = args.val_samples if args.val_samples is not None else 20000
        n_embd, n_layer, n_head = 2560, 24, 32
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    # Override parsed args with dynamically selected values to preserve downstream references
    args.max_steps = max_steps
    args.eval_steps = eval_steps
    args.context_length = context_length
    args.lr = lr
    args.warmup_steps = warmup_steps
    args.gradient_accumulation_steps = grad_accum

    # Set dataset specific settings
    if args.dataset in ["tinystories"]:
        dataset_name = "roneneldan/TinyStories"
        dataset_config = None
        data_dir = None
        text_column = "text"
    elif args.dataset in ["csn-python"]:
        dataset_name = "code-search-net/code_search_net"
        dataset_config = "python"
        data_dir = None
        text_column = "func_code_string"
    elif args.dataset in ["csn-go"]:
        dataset_name = "code-search-net/code_search_net"
        dataset_config = "go"
        data_dir = None
        text_column = "func_code_string"
    elif args.dataset in ["fineweb-edu", "fineweb_edu", "fineweb"]:
        dataset_name = "HuggingFaceFW/fineweb-edu"
        dataset_config = "sample-10BT"
        data_dir = None
        text_column = "text"
    elif args.dataset in ["github-code-python", "github_code_python"]:
        dataset_name = "hasankursun/github-code-2025-language-split"
        dataset_config = "python"
        data_dir = None
        text_column = "content"
    elif args.dataset in ["github-code-go", "github_code_go"]:
        dataset_name = "hasankursun/github-code-2025-language-split"
        dataset_config = "go"
        data_dir = None
        text_column = "content"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 1. Load raw text samples
    train_texts = load_dataset_samples(dataset_name, dataset_config, "train", text_column, train_samples_count, data_dir=data_dir, offset=0)
    val_texts = load_dataset_samples(dataset_name, dataset_config, "validation", text_column, val_samples_count, data_dir=data_dir, offset=train_samples_count)

    # 2. Train Tokenizer
    print("Training tokenizer...")
    # Use first 10,000 train texts to train tokenizer
    tokenizer = train_tokenizer(args.tokenizer_type, train_texts[:10000], args.vocab_size)


    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer trained. Actual vocab size: {actual_vocab_size}")

    # 3. Tokenize dataset
    train_chunks, train_tokens_count, train_chars_count, train_filter_stats = prepare_tokenized_data(
        train_texts, tokenizer, args.context_length, "train",
        max_example_tokens=args.max_example_tokens, max_example_chars=args.max_example_chars
    )
    val_chunks, val_tokens_count, val_chars_count, val_filter_stats = prepare_tokenized_data(
        val_texts, tokenizer, args.context_length, "val",
        max_example_tokens=args.max_example_tokens, max_example_chars=args.max_example_chars
    )

    print(f"Train: {len(train_chunks)} chunks, {train_tokens_count} tokens, {train_chars_count} characters.")
    print(f"Val: {len(val_chunks)} chunks, {val_tokens_count} tokens, {val_chars_count} characters.")
    
    char_to_token_ratio = train_chars_count / train_tokens_count if train_tokens_count > 0 else 0
    # Validation perplexity must be normalized by the VALIDATION chars/token ratio, not the
    # training one: the paper's Eq. (1)-(2) define R_char/token over the corpus being measured.
    val_char_to_token_ratio = val_chars_count / val_tokens_count if val_tokens_count > 0 else 0
    print(f"Characters per token (train): {char_to_token_ratio:.4f}")
    print(f"Characters per token (val):   {val_char_to_token_ratio:.4f}")

    if len(train_chunks) == 0:
        raise ValueError(f"Training dataset produced 0 chunks (from {len(train_texts)} raw texts). Check dataset configuration and filtering.")
    if len(val_chunks) == 0:
        raise ValueError(f"Validation dataset produced 0 chunks (from {len(val_texts)} raw texts). Check dataset configuration and filtering.")

    # 4. Prepare DataLoaders
    train_dataset = TextDataset(train_chunks)
    val_dataset = TextDataset(val_chunks)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 5. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = GPT2Config(
        vocab_size=actual_vocab_size,
        n_positions=args.context_length,
        n_ctx=args.context_length,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.token_to_id("[CLS]"),
        eos_token_id=tokenizer.token_to_id("[SEP]"),
        pad_token_id=tokenizer.token_to_id("[PAD]"),
    )
    model = GPT2LMHeadModel(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    save_dir = f"./checkpoints/{args.dataset}_{args.tokenizer_type}_seed{args.seed}_{args.model_size}{run_suffix}"
    resumed_step = 0

    # 6. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # Load existing checkpoint if available
    trainer_state_path = os.path.join(save_dir, "trainer_state.pt")
    step = 0
    epoch = 0
    total_tokens_processed = 0
    total_chars_processed = 0

    # Checkpoints written before the double-shift fix trained a two-token-ahead objective and
    # must never be silently resumed into a run that reports next-token numbers.
    OBJECTIVE = "next-token"

    if os.path.exists(trainer_state_path):
        print(f"\nResuming training from checkpoint: {trainer_state_path}")
        checkpoint_state = torch.load(trainer_state_path, map_location=device)
        ckpt_objective = checkpoint_state.get("objective")
        if ckpt_objective != OBJECTIVE:
            raise RuntimeError(
                f"Refusing to resume {trainer_state_path}: it was trained under objective "
                f"{ckpt_objective!r}, but this run expects {OBJECTIVE!r}. Checkpoints written "
                f"before the double-shift fix carry no objective stamp. Use a fresh --run_tag "
                f"(e.g. --run_tag fix2ahead) so this run writes to a new checkpoint directory."
            )
        model = type(model).from_pretrained(save_dir).to(device)
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])
        step = checkpoint_state["step"]
        epoch = checkpoint_state["epoch"]
        resumed_step = step
        total_tokens_processed = checkpoint_state.get("total_tokens_processed", 0)
        total_chars_processed = checkpoint_state.get("total_chars_processed", 0)
        print(f"Resumed at step {step}, epoch {epoch}\n")

    # 7. Training Loop
    model.train()
    
    # Logging variables
    train_losses = []
    step_times = []
    accum_steps = max(1, args.gradient_accumulation_steps)
    optimizer.zero_grad()
    
    # Reset peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # We will run steps up to max_steps
    pbar = tqdm(total=args.max_steps, desc="Training")
    if resumed_step > 0:
        pbar.update(resumed_step)
        
    global_batch_idx = 0
    micro_step = 0
    accumulated_loss = 0.0
    step_start = time.time()

    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if global_batch_idx < resumed_step * accum_steps:
                global_batch_idx += 1
                continue
            if step >= args.max_steps:
                break
            global_batch_idx += 1
                
            batch = batch.to(device)
            # GPT2LMHeadModel shifts labels internally: the loss is computed between
            # logits[..., :-1, :] and labels[..., 1:]. Pass the unshifted chunk as both
            # input_ids and labels. Pre-shifting here (input=batch[:, :-1],
            # labels=batch[:, 1:]) double-shifts and trains a two-token-ahead objective,
            # which does not match next-token greedy decoding at inference.

            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            accumulated_loss += loss.item()
            tokens_in_batch = batch.numel()
            total_tokens_processed += tokens_in_batch
            total_chars_processed += tokens_in_batch * char_to_token_ratio

            micro_step += 1

            if micro_step % accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                avg_accum_loss = accumulated_loss / accum_steps
                train_losses.append(avg_accum_loss)
                
                step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{avg_accum_loss:.4f}", "step_time": f"{step_time*1000:.1f}ms"})

                # Reset accumulation trackers for next optimizer step
                accumulated_loss = 0.0
                step_start = time.time()

                # Evaluation
                if step % args.eval_steps == 0 or step == args.max_steps:
                    val_loss = evaluate_loss(model, val_loader, device, max_batches=args.max_val_batches)
                    val_ppl_token = torch.exp(torch.tensor(val_loss)).item()
                    
                    # Per-character perplexity
                    val_ppl_char = torch.exp(torch.tensor(val_loss / val_char_to_token_ratio)).item() if val_char_to_token_ratio > 0 else float("inf")
                    
                    print(f"\nStep {step} | Val Loss: {val_loss:.4f} | PPL (Token): {val_ppl_token:.2f} | PPL (Char): {val_ppl_char:.4f}")
                    
                    # Save intermediate checkpoint
                    print(f"Saving intermediate checkpoint at step {step}...")
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
                    checkpoint_state = {
                        "objective": OBJECTIVE,
                        "step": step,
                        "epoch": epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "total_tokens_processed": total_tokens_processed,
                        "total_chars_processed": total_chars_processed,
                    }
                    torch.save(checkpoint_state, os.path.join(save_dir, "trainer_state.pt"))
                    print("Checkpoint saved successfully.\n")
                    
                    model.train()

    pbar.close()
    total_time = time.time() - start_time
    
    # 8. Final metrics
    peak_vram = torch.cuda.max_memory_allocated(device=device) if torch.cuda.is_available() else 0
    avg_step_time = (sum(step_times) / len(step_times)) if step_times else 0.0
    
    # Metrics
    tokens_per_sec = total_tokens_processed / total_time
    chars_per_sec = total_chars_processed / total_time
    
    # Final Validation Evaluation
    final_val_loss = evaluate_loss(model, val_loader, device, max_batches=args.max_val_batches)
    final_ppl_token = torch.exp(torch.tensor(final_val_loss)).item()
    final_ppl_char = torch.exp(torch.tensor(final_val_loss / val_char_to_token_ratio)).item() if val_char_to_token_ratio > 0 else float("inf")
    
    results = {
        "dataset": args.dataset,
        "tokenizer_type": args.tokenizer_type,
        "vocab_size": actual_vocab_size,
        "model_parameters": num_params,
        "total_train_time_sec": total_time,
        "avg_step_time_ms": avg_step_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "chars_per_sec": chars_per_sec,
        "peak_vram_gb": peak_vram / (1024 ** 3),
        "final_val_loss": final_val_loss,
        "final_ppl_token": final_ppl_token,
        "final_ppl_char": final_ppl_char,
        "context_length": args.context_length,
        "max_steps": args.max_steps,
        "max_val_batches": args.max_val_batches,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_example_tokens": args.max_example_tokens,
        "max_example_chars": args.max_example_chars,
        # How many examples the length filter removed. Compare across arms: a large gap means the
        # two configurations trained on materially different document sets.
        "train_filter_stats": train_filter_stats,
        "val_filter_stats": val_filter_stats,
        "char_to_token_ratio": char_to_token_ratio,
        "val_char_to_token_ratio": val_char_to_token_ratio,
        "train_tokens_count": train_tokens_count,
        "train_chars_count": train_chars_count,
        "val_tokens_count": val_tokens_count,
        "val_chars_count": val_chars_count,
        "objective": "next-token",
    }
    
    print("\nTraining complete! Results:")
    print(json.dumps(results, indent=2))
    
    # Save model and tokenizer
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f"Model and tokenizer saved to {save_dir}")
    
    # Preserve trainer state to allow extending training iterations later
    print("Trainer state file preserved for future resumption.")
    
    results_dir = os.path.dirname(args.results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {args.results_file}")
    
    # Force exit to prevent GIL/PyArrow finalization crashes
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
