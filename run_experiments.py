import os
import sys
import json
import subprocess
import argparse

def run_cmd(cmd):
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="25M", choices=["25M", "125M"])
    parser.add_argument("--num_seeds", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--python_bin", type=str, default="./venv/bin/python")
    args = parser.parse_args()

    if args.model_size == "25M":
        experiments = [
            # {"dataset": "tinystories", "tokenizer_type": "metaspace_split", "file": "./results/25M/results_ts_metaspace_split.json"},
            # {"dataset": "tinystories", "tokenizer_type": "metaspace_split_functionalizer", "file": "./results/25M/results_ts_metaspace_split_functionalizer.json"},
            # {"dataset": "tinystories", "tokenizer_type": "metaspace_split_functionalizer_repeat", "file": "./results/25M/results_ts_metaspace_split_functionalizer_repeat.json"},
            {"dataset": "tinystories", "tokenizer_type": "split", "file": "./results/25M/results_ts_split.json"},
            {"dataset": "tinystories", "tokenizer_type": "split_functionalizer", "file": "./results/25M/results_ts_split_functionalizer.json"},
            {"dataset": "tinystories", "tokenizer_type": "split_functionalizer_repeat", "file": "./results/25M/results_ts_split_functionalizer_repeat.json"},
            {"dataset": "tinystories", "tokenizer_type": "llama", "file": "./results/25M/results_ts_llama.json"},
            {"dataset": "tinystories", "tokenizer_type": "llama_functionalizer", "file": "./results/25M/results_ts_llama_functionalizer.json"},

            # {"dataset": "csn-python", "tokenizer_type": "metaspace_split", "file": "./results/25M/results_csn_python_metaspace_split.json"},
            # {"dataset": "csn-python", "tokenizer_type": "metaspace_split_functionalizer", "file": "./results/25M/results_csn_python_metaspace_split_functionalizer.json"},
            # {"dataset": "csn-python", "tokenizer_type": "metaspace_split_functionalizer_repeat", "file": "./results/25M/results_csn_python_metaspace_split_functionalizer_repeat.json"},
            {"dataset": "csn-python", "tokenizer_type": "split", "file": "./results/25M/results_csn_python_split.json"},
            {"dataset": "csn-python", "tokenizer_type": "split_functionalizer", "file": "./results/25M/results_csn_python_split_functionalizer.json"},
            {"dataset": "csn-python", "tokenizer_type": "split_functionalizer_repeat", "file": "./results/25M/results_csn_python_split_functionalizer_repeat.json"},
            {"dataset": "csn-python", "tokenizer_type": "llama", "file": "./results/25M/results_csn_python_llama.json"},
            {"dataset": "csn-python", "tokenizer_type": "llama_functionalizer", "file": "./results/25M/results_csn_python_llama_functionalizer.json"},

            # {"dataset": "csn-go", "tokenizer_type": "metaspace_split", "file": "./results/25M/results_csn_go_metaspace_split.json"},
            # {"dataset": "csn-go", "tokenizer_type": "metaspace_split_functionalizer", "file": "./results/25M/results_csn_go_metaspace_split_functionalizer.json"},
            # {"dataset": "csn-go", "tokenizer_type": "metaspace_split_functionalizer_repeat", "file": "./results/25M/results_csn_go_metaspace_split_functionalizer_repeat.json"},
            {"dataset": "csn-go", "tokenizer_type": "split", "file": "./results/25M/results_csn_go_split.json"},
            {"dataset": "csn-go", "tokenizer_type": "split_functionalizer", "file": "./results/25M/results_csn_go_split_functionalizer.json"},
            {"dataset": "csn-go", "tokenizer_type": "split_functionalizer_repeat", "file": "./results/25M/results_csn_go_split_functionalizer_repeat.json"},
            {"dataset": "csn-go", "tokenizer_type": "llama", "file": "./results/25M/results_csn_go_llama.json"},
            {"dataset": "csn-go", "tokenizer_type": "llama_functionalizer", "file": "./results/25M/results_csn_go_llama_functionalizer.json"},
        ]
    else:
        experiments = [
            {"dataset": "csn-python", "tokenizer_type": "split", "file": "./results/125M/results_csn_python_split.json"},
            {"dataset": "csn-python", "tokenizer_type": "split_functionalizer", "file": "./results/125M/results_csn_python_split_functionalizer.json"},
        ]


    PREDEFINED_SEEDS = [10, 42, 100, 2026, 7]
    active_seeds = PREDEFINED_SEEDS[:args.num_seeds]

    for exp in experiments:
        for seed in active_seeds:
            # Append seed and model size to the results filename and checkpoint dir
            base_name, ext = os.path.splitext(exp["file"])
            seed_file = f"{base_name}_seed{seed}_{args.model_size}{ext}"
            checkpoint_dir = f"./checkpoints/{exp['dataset']}_{exp['tokenizer_type']}_seed{seed}_{args.model_size}"
            
            if os.path.exists(seed_file) and os.path.exists(checkpoint_dir):
                print(f"\n==========================================")
                print(f"SKIPPING EXPERIMENT: {exp['dataset']} ({exp['tokenizer_type']}) | Seed: {seed} (results and checkpoint already exist)")
                print(f"==========================================")
                continue
                
            print(f"\n==========================================")
            print(f"STARTING EXPERIMENT: {exp['dataset']} ({exp['tokenizer_type']}) | Seed: {seed}")
            print(f"==========================================")
            
            cmd = [
                args.python_bin,
                "./train.py",
                "--dataset", exp["dataset"],
                "--tokenizer_type", exp["tokenizer_type"],
                "--model_size", args.model_size,
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(seed),
                "--results_file", seed_file
            ]
            if args.max_steps is not None:
                cmd.extend(["--max_steps", str(args.max_steps)])
            if args.eval_steps is not None:
                cmd.extend(["--eval_steps", str(args.eval_steps)])
            if args.context_length is not None:
                cmd.extend(["--context_length", str(args.context_length)])
                
            run_cmd(cmd)

    # Read results and generate comparison
    results = {}
    for exp in experiments:
        key = f"{exp['dataset']}_{exp['tokenizer_type']}"
        results[key] = []
        for seed in active_seeds:
            base_name, ext = os.path.splitext(exp["file"])
            seed_file = f"{base_name}_seed{seed}_{args.model_size}{ext}"
            if os.path.exists(seed_file):
                with open(seed_file, "r") as f:
                    results[key].append(json.load(f))
            else:
                print(f"Warning: Result file {seed_file} not found!")

    # Check if we have results to generate comparison
    datasets = ["tinystories", "csn-python", "csn-go"]
    markdown_report = "# Tokenizer Functionalizer Comparison Report\n\n"
    if args.model_size == "25M":
        markdown_report += f"This report compares the performance and compute usage of a GPT-2 model (6 layers, 512 embedding dim, 8 attention heads, ~25M parameters) "
        markdown_report += f"trained over {args.num_seeds} seed(s) with different tokenizers:\n"
        markdown_report += "1. **Metaspace + Split**: Character-based BPE (using Custom Regex split).\n"
        markdown_report += "2. **Metaspace + Split + Functionalizer**: BPE with the Custom Regex split + Functionalizer pre-tokenizer and decoder (all parameters enabled).\n"
        markdown_report += "3. **Functionalizer (Repeat)**: BPE with the Custom Regex split + Functionalizer where only repeat=True is enabled (capitalize/serialize/operators disabled).\n"
        markdown_report += "4. **Split Only**: Split without Metaspace.\n"
        markdown_report += "5. **Split + Functionalizer**: Split + Functionalizer.\n"
        markdown_report += "6. **Llama Split Only**: Llama split pattern without Metaspace.\n"
        markdown_report += "7. **Llama Split + Functionalizer**: Llama split + Functionalizer.\n"
        markdown_report += "8. **Metaspace + Whitespace**: Metaspace + Whitespace.\n"
        markdown_report += "9. **Metaspace + Whitespace + Functionalizer**: Metaspace + Whitespace + Functionalizer.\n\n"
    else:
        markdown_report += f"This report compares the performance and compute usage of a GPT-2 model (12 layers, 768 embedding dim, 12 attention heads, ~125M parameters) "
        markdown_report += f"trained over {args.num_seeds} seed(s) with different tokenizers:\n"
        markdown_report += "1. **Metaspace Only**: Character-based BPE (using Metaspace pre-tokenizer).\n"
        markdown_report += "2. **Metaspace + Functionalizer**: BPE with Metaspace + Functionalizer pre-tokenizer and decoder (all parameters enabled).\n\n"
    markdown_report += f"Predefined seed list used: {active_seeds}\n\n"
    
    baseline_mapping = {
        "metaspace": "metaspace",
        "metaspace_functionalizer": "metaspace",
        "metaspace_split": "metaspace_split",
        "metaspace_split_functionalizer": "metaspace_split",
        "metaspace_split_functionalizer_repeat": "metaspace_split",
        "split": "split",
        "split_functionalizer": "split",
        "split_functionalizer_repeat": "split",
        "llama": "llama",
        "llama_functionalizer": "llama",
        "metaspace_whitespace": "metaspace_whitespace",
        "metaspace_whitespace_functionalizer": "metaspace_whitespace",
    }
    
    tokenizer_types = [
        # "metaspace", 
        #   "metaspace_functionalizer",
        # "metaspace_split", 
        #   "metaspace_split_functionalizer", 
        #   "metaspace_split_functionalizer_repeat",
        "split", 
            "split_functionalizer", 
            "split_functionalizer_repeat",
        # "llama", 
        #     "llama_functionalizer",
        # "metaspace_whitespace", "metaspace_whitespace_functionalizer"
    ]
    
    tokenizer_names = {
        "metaspace": "Metaspace",
        "metaspace_functionalizer": "Metaspace + Functionalizer",
        "metaspace_split": "Metaspace + Split",
        "metaspace_split_functionalizer": "Metaspace + Split + Functionalizer",
        "metaspace_split_functionalizer_repeat": "Metaspace + Split + Functionalizer (Repeat)",
        "split": "Split",
        "split_functionalizer": "Split + Functionalizer",
        "split_functionalizer_repeat": "Split + Functionalizer (Repeat)",
        "llama": "Llama Split",
        "llama_functionalizer": "Llama Split + Functionalizer",
        "metaspace_whitespace": "Metaspace + Whitespace",
        "metaspace_whitespace_functionalizer": "Metaspace + Whitespace + Functionalizer",
    }
    
    for ds in datasets:
        def get_val_stats(results_list, key, fmt="{:.2f}"):
            values = [r[key] for r in results_list if r is not None and key in r]
            if not values: return "N/A"
            if len(values) == 1:
                return fmt.format(values[0])
            import math
            n = len(values)
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = math.sqrt(variance)
            return f"{fmt.format(mean)} ± {fmt.format(std)}"
            
        def get_seq_infl_stats(results_list, base_list):
            inflations = []
            # Align by seed (paired comparison)
            for r, r_base in zip(results_list, base_list):
                if r is None or r_base is None: continue
                cpt = r.get("char_to_token_ratio", 0)
                base_cpt = r_base.get("char_to_token_ratio", 0)
                if cpt > 0 and base_cpt > 0:
                    inflations.append((base_cpt / cpt - 1) * 100)
            if not inflations: return "N/A"
            if len(inflations) == 1:
                return f"{inflations[0]:+.2f}%"
            import math
            n = len(inflations)
            mean = sum(inflations) / n
            variance = sum((x - mean) ** 2 for x in inflations) / n
            std = math.sqrt(variance)
            return f"{mean:+.2f}% ± {std:.2f}%"
            
        if ds == "tinystories":
            ds_name = "TinyStories"
        elif ds == "csn-python":
            ds_name = "CSN-Python"
        else:
            ds_name = "CSN-Go"
        
        markdown_report += f"## Dataset: {ds_name}\n\n"
        markdown_report += "| Tokenizer Type | Vocab Size | Chars/Token | Inflation | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |\n"
        markdown_report += "|---|---|---|---|---|---|---|---|---|\n"
        
        for tok_type in tokenizer_types:
            key = f"{ds}_{tok_type}"
            r_list = results.get(key, [])
            if not r_list:
                markdown_report += f"| **{tokenizer_names[tok_type]}** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n"
                continue
                
            base_key = f"{ds}_{baseline_mapping[tok_type]}"
            r_base_list = results.get(base_key, [])
            
            infl_val = "0.00%" if tok_type == baseline_mapping[tok_type] else get_seq_infl_stats(r_list, r_base_list)
            
            markdown_report += (
                f"| **{tokenizer_names[tok_type]}** | "
                f"{get_val_stats(r_list, 'vocab_size', '{:,}')} | "
                f"{get_val_stats(r_list, 'char_to_token_ratio', '{:.3f}')} | "
                f"{infl_val} | "
                f"{get_val_stats(r_list, 'tokens_per_sec', '{:.1f}')} | "
                f"{get_val_stats(r_list, 'chars_per_sec', '{:.1f}')} | "
                f"{get_val_stats(r_list, 'final_val_loss', '{:.4f}')} | "
                f"{get_val_stats(r_list, 'final_ppl_token', '{:.2f}')} | "
                f"{get_val_stats(r_list, 'final_ppl_char', '{:.4f}')} |\n"
            )
        markdown_report += "\n"

    report_path = f"./results/training_report_{args.model_size}.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(markdown_report)
        
    print(f"\n==========================================")
    print(f"EXPERIMENTS COMPLETE!")
    print(f"Report written to {report_path}")
    print(f"==========================================")
    print(markdown_report)

if __name__ == "__main__":
    main()

