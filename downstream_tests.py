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
import ast
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

def load_validation_prompts(dataset_name, dataset_config, split, text_column, num_samples, is_code=False):
    print(f"Loading {num_samples} validation prompts from {dataset_name}...")
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    prompts = []
    targets = []
    
    for item in ds:
        text = item[text_column]
        if not text or not text.strip():
            continue
            
        if is_code:
            # Code prompt: Extract the function signature (usually the first line)
            lines = text.split("\n")
            prompt = lines[0]
            target = "\n".join(lines[1:])
            # Make sure it looks like a valid python definition or go definition
            is_python = prompt.strip().startswith("def ") and prompt.strip().endswith(":")
            is_go = prompt.strip().startswith("func ")
            if is_python or is_go:
                prompts.append(prompt)
                targets.append(target)
        else:
            # Natural language prompt: Take first 50 characters, split at last space to be clean
            if len(text) > 60:
                prompt_raw = text[:50]
                last_space = prompt_raw.rfind(" ")
                if last_space > 20:
                    prompt = prompt_raw[:last_space]
                else:
                    prompt = prompt_raw
                target = text[len(prompt):]
                prompts.append(prompt)
                targets.append(target)
                
        if len(prompts) >= num_samples:
            break
            
    return prompts, targets

def get_repetition_rate(text, n=2):
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    unique_ngrams = set(ngrams)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def get_tokens_before_collapse(token_ids, max_consecutive=3):
    if not token_ids:
        return 0
    current_tok = token_ids[0]
    consecutive_count = 1
    for idx, tok in enumerate(token_ids[1:]):
        if tok == current_tok:
            consecutive_count += 1
            if consecutive_count > max_consecutive:
                return max(0, idx + 1 - max_consecutive)
        else:
            current_tok = tok
            consecutive_count = 1
    return len(token_ids)

def is_non_empty_content(text):
    if not text:
        return False
    # Strip Unicode Private Use Area (PUA) characters U+E000 to U+F8FF
    cleaned = "".join([c for c in text if not (0xE000 <= ord(c) <= 0xF8FF)])
    # Strip metaspace characters
    cleaned = cleaned.replace("▁", "")
    # Strip all whitespaces, newlines, tabs
    cleaned = "".join([c for c in cleaned if not c.isspace()])
    return len(cleaned) > 0

def test_syntax_validity(prompt, completion, is_go=False):
    if is_go:
        # Check Go syntax using gofmt if available, otherwise fallback to bracket matching
        import subprocess
        import tempfile
        
        full_code = "package main\n\n" + prompt + " " + completion
        if not full_code.strip().endswith("}"):
            full_code += "\n}"
            
        try:
            # Check if gofmt is available
            subprocess.run(["gofmt", "-h"], capture_output=True)
            with tempfile.NamedTemporaryFile("w", suffix=".go", delete=False) as f:
                f.write(full_code)
                temp_name = f.name
            try:
                res_check = subprocess.run(["gofmt", "-e", temp_name], capture_output=True)
                return res_check.returncode == 0
            finally:
                os.remove(temp_name)
        except Exception:
            pass
            
        # Fallback: matching braces, brackets, parentheses, and quotes
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        in_string = False
        string_char = None
        escaped = False
        
        for i, char in enumerate(full_code):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if in_string:
                if char == string_char:
                    in_string = False
                continue
            else:
                if char in ['"', "'", '`']:
                    in_string = True
                    string_char = char
                    continue
                    
            if char in mapping.values():
                stack.append(char)
            elif char in mapping.keys():
                if not stack or stack[-1] != mapping[char]:
                    return False
                stack.pop()
                
        return not in_string and len(stack) == 0
    else:
        # Python syntax
        full_code = prompt + "\n" + completion
        try:
            ast.parse(full_code)
            return True
        except SyntaxError:
            return False

def run_evaluation(dataset, tokenizer_type, seed, prompts, is_code=False, is_go=False, model_size="25M"):
    checkpoint_dir = f"./checkpoints/{dataset}_{tokenizer_type}_seed{seed}_{model_size}"
        
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} not found. Skipping seed {seed} for {tokenizer_type}.")
        return None
        
    print(f"\nEvaluating downstream task for {dataset} ({tokenizer_type})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    
    sep_id = tokenizer.token_to_id("[SEP]")
    max_new_tokens = 64 if is_code else 50
    
    total_tokens_generated = 0
    total_tokens_before_collapse = 0
    total_chars_generated = 0
    total_generation_time = 0.0
    
    total_coherent_tokens = 0
    total_coherent_chars = 0
    total_coherent_time = 0.0
    
    repetition_rates = []
    syntax_success_count = 0
    non_empty_count = 0
    success_sample = None
    failure_sample = None
    
    for prompt in prompts:
        # Encode prompt
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
        
        # Greedy generation loop to measure exact tokens and characters time
        start_time = time.time()
        curr_input = input_ids
        generated_ids = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(curr_input)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
                
            token_val = next_token_id.item()
            generated_ids.append(token_val)
            
            if token_val == sep_id:
                break
                
            # Early stopping: if a token repeats consecutively > 3 times, stop generating
            if len(generated_ids) >= 4 and len(set(generated_ids[-4:])) == 1:
                break
                
            curr_input = torch.cat([curr_input, next_token_id.unsqueeze(0)], dim=-1)
            
        generation_time = time.time() - start_time
        
        # Decode completion
        try:
            completion = tokenizer.decode(generated_ids)
        except Exception as e:
            # Fallback to join raw token strings if functionalizer decoder crashes on malformed generated sequences
            raw_tokens = []
            for tid in generated_ids:
                tok = tokenizer.id_to_token(tid)
                if tok is not None:
                    raw_tokens.append(tok)
            completion = " ".join(raw_tokens)
        
        # Calculate coherent prefix metrics
        k = get_tokens_before_collapse(generated_ids)
        try:
            coherent_completion = tokenizer.decode(generated_ids[:k])
        except Exception as e:
            raw_tokens = []
            for tid in generated_ids[:k]:
                tok = tokenizer.id_to_token(tid)
                if tok is not None:
                    raw_tokens.append(tok)
            coherent_completion = " ".join(raw_tokens)
            
        coherent_time = generation_time * (k / len(generated_ids)) if len(generated_ids) > 0 else 0.0
        
        total_tokens_generated += len(generated_ids)
        total_tokens_before_collapse += k
        total_chars_generated += len(completion)
        total_generation_time += generation_time
        
        total_coherent_tokens += k
        total_coherent_chars += len(coherent_completion)
        total_coherent_time += coherent_time
        
        is_non_empty = is_non_empty_content(completion)
        if is_code:
            is_success = is_non_empty and test_syntax_validity(prompt, completion, is_go=is_go)
            if is_success:
                syntax_success_count += 1
        else:
            rep_rate = get_repetition_rate(completion, n=2) if is_non_empty else 1.0
            is_success = is_non_empty and rep_rate < 0.25
            repetition_rates.append(rep_rate)
                
        if not is_non_empty:
            # Empty completion counts as a failure, do not increment non_empty_count
            pass
        else:
            non_empty_count += 1
            
        # Collect success and failure samples
        sample_data = {
            "prompt": prompt,
            "completion": completion,
            "tokens": [tokenizer.id_to_token(tid) for tid in generated_ids[:20]] + (["..."] if len(generated_ids) > 20 else [])
        }
        if is_success and success_sample is None:
            success_sample = sample_data
        elif not is_success and failure_sample is None:
            failure_sample = sample_data
            
    pct_non_empty = (non_empty_count / len(prompts)) * 100
    avg_tokens = total_tokens_generated / len(prompts)
    avg_tokens_before_collapse = total_tokens_before_collapse / len(prompts)
    avg_chars = total_chars_generated / len(prompts)
    
    # Speed is computed based on coherent prefix to be fair and accurate
    tokens_per_sec = total_coherent_tokens / total_coherent_time if total_coherent_time > 0 else 0
    chars_per_sec = total_coherent_chars / total_coherent_time if total_coherent_time > 0 else 0
    
    success_rate = (syntax_success_count / len(prompts)) * 100 if is_code else (1.0 - (sum(repetition_rates)/len(repetition_rates) if repetition_rates else 0.0)) * 100
    success_metric_label = "Syntax Success Rate (%)" if is_code else "Coherence (1 - Repetition) (%)"
    
    results = {
        "dataset": dataset,
        "tokenizer_type": tokenizer_type,
        "avg_tokens_generated": avg_tokens,
        "avg_tokens_before_collapse": avg_tokens_before_collapse,
        "avg_chars_generated": avg_chars,
        "tokens_per_sec": tokens_per_sec,
        "chars_per_sec": chars_per_sec,
        "pct_non_empty": pct_non_empty,
        "pct_empty": 100.0 - pct_non_empty,
        "success_rate": success_rate,
        "success_metric_label": success_metric_label,
        "success_sample": success_sample,
        "failure_sample": failure_sample
    }
    
    print(json.dumps(results, indent=2))
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--model_size", type=str, default="25M", choices=["25M", "125M"])
    args = parser.parse_args()

    PREDEFINED_SEEDS = [10, 42, 100, 2026, 7, 999]
    active_seeds = PREDEFINED_SEEDS[:args.num_seeds]

    datasets = [
        {"name": "tinystories", "path": "roneneldan/TinyStories", "config": None, "text_column": "text", "is_code": False, "is_go": False},
        {"name": "csn-python", "path": "code-search-net/code_search_net", "config": "python", "text_column": "func_code_string", "is_code": True, "is_go": False}, 
        {"name": "csn-go", "path": "code-search-net/code_search_net", "config": "go", "text_column": "func_code_string", "is_code": True, "is_go": True}
    ]
    
    tokenizer_types = [
        # "metaspace", 
        #   "metaspace_functionalizer",
        # "metaspace_split", 
        #     "metaspace_split_functionalizer", 
        #     "metaspace_split_functionalizer_repeat",
        "split", 
            "split_functionalizer",
            "split_functionalizer_repeat",
        # "llama", 
        #     "llama_functionalizer",
        # "metaspace_whitespace", 
        #   "metaspace_whitespace_functionalizer"
    ]
    num_prompts = args.num_prompts
    
    all_results = {}
    
    for ds in datasets:
        prompts, _ = load_validation_prompts(
            ds["path"], ds["config"], "validation", ds["text_column"], num_prompts, ds["is_code"]
        )
        
        if not prompts:
            print(f"Failed to load prompts for dataset {ds['name']}. Skipping.")
            continue
            
        ds_results = {}
        for tok_type in tokenizer_types:
            ds_results[tok_type] = []
            for seed in active_seeds:
                res = run_evaluation(ds["name"], tok_type, seed, prompts, ds["is_code"], is_go=ds["is_go"], model_size=args.model_size)
                if res:
                    ds_results[tok_type].append(res)
                    
        all_results[ds["name"]] = ds_results
        
    # Generate downstream markdown report
    report = "# Downstream Tasks Evaluation Report\n\n"
    if args.model_size == "25M":
        report += "This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) "
        report += f"for the models trained under the different tokenization configurations across {args.num_seeds} seed(s).\n\n"
    else:
        report += "This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) "
        report += f"for the 125M models trained under the different tokenization configurations across {args.num_seeds} seed(s).\n\n"
    report += f"Predefined seed list used: {active_seeds}\n\n"
    
    for ds_name, tok_results_dict in all_results.items():
        has_any_results = any(len(lst) > 0 for lst in tok_results_dict.values())
        if not has_any_results:
            continue
            
        if ds_name == "csn-python":
            ds_label = "CSN-Python (Code Generation)"
        elif ds_name == "csn-go":
            ds_label = "CSN-Go (Code Generation)"
        else:
            ds_label = "TinyStories (Natural Language Generation)"
            
        metric_label = "Success Rate (%)"
        for tok_type, r_list in tok_results_dict.items():
            if r_list:
                metric_label = r_list[0]["success_metric_label"]
                break
                
        report += f"## Dataset: {ds_label}\n\n"
        report += f"| Tokenizer Type | Avg Tokens Generated | Avg Tokens Pre-Collapse | Avg Chars Generated | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | {metric_label} |\n"
        report += "|---|---|---|---|---|---|---|---|\n"
        
        def get_stat(r_list, key, fmt="{:.1f}"):
            values = [r[key] for r in r_list if r is not None and key in r]
            if not values:
                return "N/A"
            if len(values) == 1:
                return fmt.format(values[0])
            import math
            n = len(values)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = math.sqrt(variance)
            return f"{fmt.format(mean)} ± {fmt.format(std)}"
            
        tok_name_map = {
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
            "metaspace_functionalizer": "Metaspace + Functionalizer"
        }
        for tok_type in tokenizer_types:
            r_list = tok_results_dict.get(tok_type, [])
            pretty_name = tok_name_map.get(tok_type, tok_type)
            if not r_list:
                report += f"| **{pretty_name}** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
                continue
                
            report += (
                f"| **{pretty_name}** | "
                f"{get_stat(r_list, 'avg_tokens_generated', '{:.1f}')} | "
                f"{get_stat(r_list, 'avg_tokens_before_collapse', '{:.1f}')} | "
                f"{get_stat(r_list, 'avg_chars_generated', '{:.1f}')} | "
                f"{get_stat(r_list, 'tokens_per_sec', '{:.1f}')} | "
                f"{get_stat(r_list, 'chars_per_sec', '{:.1f}')} | "
                f"{get_stat(r_list, 'pct_empty', '{:.1f}')}% | "
                f"{get_stat(r_list, 'success_rate', '{:.2f}')}% |\n"
            )
        report += "\n"
        
    downstream_report_path = f"./results/downstream_report_{args.model_size}.md"
    os.makedirs(os.path.dirname(downstream_report_path), exist_ok=True)
    with open(downstream_report_path, "w") as f:
        f.write(report)
        
    # Generate downstream samples report
    samples_report = f"# Downstream Generation Samples Report ({args.model_size})\n\n"
    samples_report += "This report showcases a comparison of generated outputs between the different tokenizer configurations.\n\n"
    
    for ds_name, tok_results_dict in all_results.items():
        has_any_results = any(len(lst) > 0 for lst in tok_results_dict.values())
        if not has_any_results:
            continue
            
        if ds_name == "csn-python":
            ds_label = "CSN-Python (Code Generation)"
        elif ds_name == "csn-go":
            ds_label = "CSN-Go (Code Generation)"
        else:
            ds_label = "TinyStories (Natural Language Generation)"
            
        samples_report += f"## Dataset: {ds_label}\n\n"
        
        # We will collect one success and one failure sample for each available tokenizer type
        for tok_type in tokenizer_types:
            r_list = tok_results_dict.get(tok_type, [])
            if not r_list:
                continue
            first_run = r_list[0]
            if first_run.get("success_sample") is None and first_run.get("failure_sample") is None:
                continue
                
            pretty_name = tok_name_map.get(tok_type, tok_type)
            samples_report += f"### Configuration: {pretty_name}\n\n"
            
            if first_run.get("success_sample"):
                s = first_run["success_sample"]
                samples_report += "#### Success Sample\n"
                samples_report += f"* **Prompt**: `{repr(s['prompt'])}`\n"
                samples_report += f"* **Generated Tokens (first 20)**: `{s['tokens']}`\n"
                samples_report += f"* **Decoded Completion**: `{repr(s['completion'])}`\n"
                samples_report += f"* **Completion Length (chars)**: {len(s['completion'])}\n\n"
            else:
                samples_report += "#### Success Sample\n*No success sample found for this configuration.*\n\n"
                
            if first_run.get("failure_sample"):
                f_s = first_run["failure_sample"]
                samples_report += "#### Failure Sample\n"
                samples_report += f"* **Prompt**: `{repr(f_s['prompt'])}`\n"
                samples_report += f"* **Generated Tokens (first 20)**: `{f_s['tokens']}`\n"
                samples_report += f"* **Decoded Completion**: `{repr(f_s['completion'])}`\n"
                samples_report += f"* **Completion Length (chars)**: {len(f_s['completion'])}\n\n"
            else:
                samples_report += "#### Failure Sample\n*No failure sample found for this configuration.*\n\n"
                
            samples_report += "---\n\n"
            
    downstream_samples_path = f"./results/downstream_samples_{args.model_size}.md"
    with open(downstream_samples_path, "w") as f:
        f.write(samples_report)
        
    print(f"\nDownstream evaluation finished!")
    print(f"Report saved to {downstream_report_path}")
    print(f"Samples saved to {downstream_samples_path}")

if __name__ == "__main__":
    main()
