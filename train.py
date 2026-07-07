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

DEFAULT_SPLIT_PAT = r"\n+|\t+|[ ]| |[^\p{L}\p{N}\s _]+"
LLAMA_SPLIT_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


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
    tokenizer.train([corpus_path], trainer)
    return tokenizer



class TextDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = torch.tensor(chunks, dtype=torch.long)
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        # In causal language modeling, we predict the next token.
        # We can return the chunk, and split into input and target in the training loop
        return self.chunks[idx]

def load_dataset_samples(dataset_name, dataset_config, split, text_column, num_samples):
    print(f"Loading {num_samples} samples from {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    samples = []
    for item in ds:
        text = item[text_column]
        if text and text.strip():
            samples.append(text)
            if len(samples) >= num_samples:
                break
    return samples

def prepare_tokenized_data(samples, tokenizer, context_length, desc=""):
    all_token_ids = []
    total_chars = 0
    sep_id = tokenizer.token_to_id("[SEP]")
    
    for text in tqdm(samples, desc=f"Tokenizing {desc}"):
        encoding = tokenizer.encode(text)
        ids = encoding.ids + [sep_id]
        all_token_ids.extend(ids)
        total_chars += len(text) + 1  # count character length
        
    chunks = []
    num_tokens = len(all_token_ids)
    for i in range(0, num_tokens - context_length, context_length):
        chunks.append(all_token_ids[i:i+context_length])
        
    return chunks, num_tokens, total_chars

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["tinystories", "csn-python", "csn-go"])
    parser.add_argument("--tokenizer_type", type=str, required=True, choices=[
        "metaspace", "metaspace_functionalizer",
        "metaspace_split", "metaspace_split_functionalizer", "metaspace_split_functionalizer_repeat",
        "split", "split_functionalizer", "split_functionalizer_repeat",
        "llama", "llama_functionalizer",
        "metaspace_whitespace", "metaspace_whitespace_functionalizer",
        "metaspace", "metaspace_functionalizer"
    ])
    parser.add_argument("--model_size", type=str, default="25M", choices=["25M", "125M"])
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_file", type=str, default="./results/results.json")
    args = parser.parse_args()

    # Ensure reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set parameters dynamically based on model size class
    if args.model_size == "25M":
        max_steps = args.max_steps if args.max_steps is not None else 5000
        eval_steps = args.eval_steps if args.eval_steps is not None else 1000
        context_length = args.context_length if args.context_length is not None else 256
        train_samples_count = args.train_samples if args.train_samples is not None else 50000
        val_samples_count = args.val_samples if args.val_samples is not None else 5000
        n_embd, n_layer, n_head = 512, 6, 8
    else:  # 125M
        max_steps = args.max_steps if args.max_steps is not None else 50000
        eval_steps = args.eval_steps if args.eval_steps is not None else 5000
        context_length = args.context_length if args.context_length is not None else 512
        train_samples_count = args.train_samples if args.train_samples is not None else 500000
        val_samples_count = args.val_samples if args.val_samples is not None else 10000
        n_embd, n_layer, n_head = 768, 12, 12

    # Override parsed args with dynamically selected values to preserve downstream references
    args.max_steps = max_steps
    args.eval_steps = eval_steps
    args.context_length = context_length

    # Set dataset specific settings
    if args.dataset == "tinystories":
        dataset_name = "roneneldan/TinyStories"
        dataset_config = None
        text_column = "text"
    elif args.dataset == "csn-python":
        dataset_name = "code-search-net/code_search_net"
        dataset_config = "python"
        text_column = "func_code_string"
    elif args.dataset == "csn-go":
        dataset_name = "code-search-net/code_search_net"
        dataset_config = "go"
        text_column = "func_code_string"

    # 1. Load raw text samples
    train_texts = load_dataset_samples(dataset_name, dataset_config, "train", text_column, train_samples_count)
    val_texts = load_dataset_samples(dataset_name, dataset_config, "validation", text_column, val_samples_count)

    # 2. Train Tokenizer
    print("Training tokenizer...")
    temp_corpus_path = f"temp_tokenizer_corpus_{args.dataset}_{args.tokenizer_type}.txt"
    # Use first 10,000 train texts to train tokenizer
    with open(temp_corpus_path, "w", encoding="utf-8") as f:
        for text in train_texts[:10000]:
            f.write(text + "\n")
            
    tokenizer = train_tokenizer(args.tokenizer_type, temp_corpus_path, args.vocab_size)

        
    if os.path.exists(temp_corpus_path):
        os.remove(temp_corpus_path)
        
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer trained. Actual vocab size: {actual_vocab_size}")

    # 3. Tokenize dataset
    train_chunks, train_tokens_count, train_chars_count = prepare_tokenized_data(
        train_texts, tokenizer, args.context_length, "train"
    )
    val_chunks, val_tokens_count, val_chars_count = prepare_tokenized_data(
        val_texts, tokenizer, args.context_length, "val"
    )

    print(f"Train: {len(train_chunks)} chunks, {train_tokens_count} tokens, {train_chars_count} characters.")
    print(f"Val: {len(val_chunks)} chunks, {val_tokens_count} tokens, {val_chars_count} characters.")
    
    char_to_token_ratio = train_chars_count / train_tokens_count if train_tokens_count > 0 else 0
    print(f"Characters per token (train): {char_to_token_ratio:.4f}")

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

    save_dir = f"./checkpoints/{args.dataset}_{args.tokenizer_type}_seed{args.seed}_{args.model_size}"
    resumed_step = 0

    # 6. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=200,
        num_training_steps=args.max_steps
    )

    # Load existing checkpoint if available
    trainer_state_path = os.path.join(save_dir, "trainer_state.pt")
    step = 0
    epoch = 0
    total_tokens_processed = 0
    total_chars_processed = 0

    if os.path.exists(trainer_state_path):
        print(f"\nResuming training from checkpoint: {trainer_state_path}")
        checkpoint_state = torch.load(trainer_state_path, map_location=device)
        model.load_state_dict(torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location=device))
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
    
    # Reset peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # We will run steps up to max_steps
    pbar = tqdm(total=args.max_steps, desc="Training")
    if resumed_step > 0:
        pbar.update(resumed_step)
        
    global_batch_idx = 0
    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if global_batch_idx < resumed_step:
                global_batch_idx += 1
                continue
            if step >= args.max_steps:
                break
            global_batch_idx += 1
                
            batch = batch.to(device)
            # Input: tokens 0 to L-2, Target: tokens 1 to L-1
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            step_start = time.time()
            
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            step_time = time.time() - step_start
            step_times.append(step_time)
            train_losses.append(loss.item())
            
            # Throughput calculation
            # Batch size * sequence length inputs
            tokens_in_step = inputs.numel()
            total_tokens_processed += tokens_in_step
            # Character throughput = tokens * characters per token ratio
            total_chars_processed += tokens_in_step * char_to_token_ratio
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step_time": f"{step_time*1000:.1f}ms"})

            # Evaluation
            if step % args.eval_steps == 0 or step == args.max_steps:
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = val_batch.to(device)
                        val_inputs = val_batch[:, :-1]
                        val_targets = val_batch[:, 1:]
                        val_outputs = model(input_ids=val_inputs, labels=val_targets)
                        val_loss_sum += val_outputs.loss.item()
                        val_batches += 1
                
                val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
                val_ppl_token = torch.exp(torch.tensor(val_loss)).item()
                
                # Per-character perplexity
                # NLL total = mean_loss * number of predicted tokens in val
                # predicted tokens in val = val_batches * batch_size * (context_length - 1)
                # But to make it precise, we can use the actual tokens in val_dataset:
                # predicted tokens = len(val_dataset) * (context_length - 1)
                # predicted chars = predicted tokens * character_per_token_ratio
                # PPL_char = exp(NLL_total / predicted_chars)
                #          = exp(mean_loss * predicted_tokens / (predicted_tokens * char_per_token_ratio))
                #          = exp(mean_loss / char_per_token_ratio)
                val_ppl_char = torch.exp(torch.tensor(val_loss / char_to_token_ratio)).item()
                
                print(f"\nStep {step} | Val Loss: {val_loss:.4f} | PPL (Token): {val_ppl_token:.2f} | PPL (Char): {val_ppl_char:.4f}")
                
                # Save intermediate checkpoint
                print(f"Saving intermediate checkpoint at step {step}...")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
                checkpoint_state = {
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
    avg_step_time = sum(step_times) / len(step_times)
    
    # Metrics
    tokens_per_sec = total_tokens_processed / total_time
    chars_per_sec = total_chars_processed / total_time
    
    # Final Validation Evaluation
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            val_inputs = val_batch[:, :-1]
            val_targets = val_batch[:, 1:]
            val_outputs = model(input_ids=val_inputs, labels=val_targets)
            val_loss_sum += val_outputs.loss.item()
            val_batches += 1
            
    final_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
    final_ppl_token = torch.exp(torch.tensor(final_val_loss)).item()
    final_ppl_char = torch.exp(torch.tensor(final_val_loss / char_to_token_ratio)).item()
    
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
        "char_to_token_ratio": char_to_token_ratio,
        "train_tokens_count": train_tokens_count,
        "train_chars_count": train_chars_count,
    }
    
    print("\nTraining complete! Results:")
    print(json.dumps(results, indent=2))
    
    # Save model and tokenizer
    save_dir = f"./checkpoints/{args.dataset}_{args.tokenizer_type}_seed{args.seed}_{args.model_size}"
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
