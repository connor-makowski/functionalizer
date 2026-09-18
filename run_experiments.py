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
    parser.add_argument("--model_size", type=str, default="125M", choices=["25M", "125M", "2B"])
    parser.add_argument("--num_seeds", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None,
                        help="Passed through to train.py; max validation batches to evaluate.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--max_example_tokens", type=int, default=None,
                        help="Passed through to train.py; drops examples at/above this token count.")
    parser.add_argument("--max_example_chars", type=int, default=None,
                        help="Passed through to train.py; tokenizer-neutral length filter.")
    parser.add_argument("--python_bin", type=str, default="./venv/bin/python")
    # Suffix for checkpoint dirs, per-seed result files, and the report, so a re-run
    # cannot skip over or overwrite artifacts from an earlier training objective.
    parser.add_argument("--run_tag", type=str, default="")
    args = parser.parse_args()

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""

    default_experiments = [

        # Natural Language (TinyStories)
        # {"dataset": "tinystories", "tokenizer_type": "llama", "file": f"./results/{args.model_size}/results_ts_llama.json"},
        # {"dataset": "tinystories", "tokenizer_type": "llama_functionalizer", "file": f"./results/{args.model_size}/results_ts_llama_functionalizer.json"},

        # Code (Code Search Net) Python
        # {"dataset": "csn-python", "tokenizer_type": "llama", "file": f"./results/{args.model_size}/results_csn_python_llama.json"},
        # {"dataset": "csn-python", "tokenizer_type": "llama_functionalizer", "file": f"./results/{args.model_size}/results_csn_python_llama_functionalizer.json"},

        # Code (Code Search Net) Go
        # {"dataset": "csn-go", "tokenizer_type": "llama", "file": f"./results/{args.model_size}/results_csn_go_llama.json"},
        # {"dataset": "csn-go", "tokenizer_type": "llama_functionalizer", "file": f"./results/{args.model_size}/results_csn_go_llama_functionalizer.json"},

        # FineWeb-Edu (Natural Language Prose)
        {"dataset": "fineweb-edu", "tokenizer_type": "llama", "file": f"./results/{args.model_size}/results_fineweb_edu_llama.json"},
        {"dataset": "fineweb-edu", "tokenizer_type": "llama_functionalizer", "file": f"./results/{args.model_size}/results_fineweb_edu_llama_functionalizer.json"},

        # GitHub-Code-Python
        {"dataset": "github-code-python", "tokenizer_type": "llama", "file": f"./results/{args.model_size}/results_github_code_python_llama.json"},
        {"dataset": "github-code-python", "tokenizer_type": "llama_functionalizer", "file": f"./results/{args.model_size}/results_github_code_python_llama_functionalizer.json"},

        # GitHub-Code-Go
        # {"dataset": "github-code-go", "tokenizer_type": "llama", "file": f"./results/{args.model_size}/results_github_code_go_llama.json"},
        # {"dataset": "github-code-go", "tokenizer_type": "llama_functionalizer", "file": f"./results/{args.model_size}/results_github_code_go_llama_functionalizer.json"},
    ]

    if args.model_size in ["25M", "125M", "2B"]:
        experiments = default_experiments
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")


    PREDEFINED_SEEDS = [1, 2, 3, 4, 5]
    active_seeds = PREDEFINED_SEEDS[:args.num_seeds]

    for exp in experiments:
        for seed in active_seeds:
            # Append seed and model size to the results filename and checkpoint dir
            base_name, ext = os.path.splitext(exp["file"])
            seed_file = f"{base_name}_seed{seed}_{args.model_size}{run_suffix}{ext}"
            checkpoint_dir = f"./checkpoints/{exp['dataset']}_{exp['tokenizer_type']}_seed{seed}_{args.model_size}{run_suffix}"

            if os.path.exists(seed_file) and os.path.exists(checkpoint_dir):
                # Check whether the saved run reached the requested max_steps.
                # If max_steps was bumped since the last run, we must re-launch
                # train.py so it can resume from the checkpoint and keep going.
                skip = False
                requested_max_steps = args.max_steps  # None means train.py uses its own default
                try:
                    with open(seed_file, "r") as _f:
                        _saved = json.load(_f)
                    saved_max_steps = _saved.get("max_steps")  # None if old result file
                    # Only skip when the saved run used the same (or higher) max_steps target.
                    # Treat missing saved_max_steps as "unknown / incomplete" → do not skip.
                    if saved_max_steps is not None and saved_max_steps >= (requested_max_steps or saved_max_steps):
                        skip = True
                except Exception:
                    pass  # Unreadable result file → do not skip
                if skip:
                    print(f"\n==========================================")
                    print(f"SKIPPING EXPERIMENT: {exp['dataset']} ({exp['tokenizer_type']}) | Seed: {seed} (already trained to {saved_max_steps} steps)")
                    print(f"==========================================")
                    continue
                else:
                    print(f"\n==========================================")
                    print(f"RESUMING EXPERIMENT: {exp['dataset']} ({exp['tokenizer_type']}) | Seed: {seed} (checkpoint exists; extending to {requested_max_steps} steps)")
                    print(f"============================================")
                
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
                "--seed", str(seed),
                "--results_file", seed_file
            ]
            if args.gradient_accumulation_steps is not None:
                cmd.extend(["--gradient_accumulation_steps", str(args.gradient_accumulation_steps)])
            if args.lr is not None:
                cmd.extend(["--lr", str(args.lr)])
            if args.warmup_steps is not None:
                cmd.extend(["--warmup_steps", str(args.warmup_steps)])
            if args.run_tag:
                cmd.extend(["--run_tag", args.run_tag])
            if args.max_steps is not None:
                cmd.extend(["--max_steps", str(args.max_steps)])
            if args.eval_steps is not None:
                cmd.extend(["--eval_steps", str(args.eval_steps)])
            if args.max_val_batches is not None:
                cmd.extend(["--max_val_batches", str(args.max_val_batches)])
            if args.context_length is not None:
                cmd.extend(["--context_length", str(args.context_length)])
            if args.max_example_tokens is not None:
                cmd.extend(["--max_example_tokens", str(args.max_example_tokens)])
            if args.max_example_chars is not None:
                cmd.extend(["--max_example_chars", str(args.max_example_chars)])
                
            run_cmd(cmd)

    # Read results and generate comparison
    results = {}
    for exp in experiments:
        key = f"{exp['dataset']}_{exp['tokenizer_type']}"
        results[key] = []
        for seed in active_seeds:
            base_name, ext = os.path.splitext(exp["file"])
            seed_file = f"{base_name}_seed{seed}_{args.model_size}{run_suffix}{ext}"
            if os.path.exists(seed_file):
                with open(seed_file, "r") as f:
                    results[key].append(json.load(f))
            else:
                print(f"Warning: Result file {seed_file} not found!")

    # Check if we have results to generate comparison
    datasets = list(dict.fromkeys([exp["dataset"] for exp in experiments]))
    markdown_report = "# Tokenizer Functionalizer Comparison Report\n\n"
    if args.model_size == "25M":
        markdown_report += f"This report compares the performance and compute usage of a GPT-2 model (6 layers, 512 embedding dim, 8 attention heads, ~25M parameters) "
        markdown_report += f"trained over {args.num_seeds} seed(s) with different tokenizers:\n"
    elif args.model_size == "125M":
        markdown_report += f"This report compares the performance and compute usage of a GPT-2 model (12 layers, 768 embedding dim, 12 attention heads, ~125M parameters) "
        markdown_report += f"trained over {args.num_seeds} seed(s) with different tokenizers:\n"
    elif args.model_size == "2B":
        markdown_report += f"This report compares the performance and compute usage of a GPT-2 model (24 layers, 2560 embedding dim, 32 attention heads, ~2B parameters) "
        markdown_report += f"trained over {args.num_seeds} seed(s) with different tokenizers:\n"
    else:
        markdown_report += f"This report compares the performance and compute usage of a GPT-2 model ({args.model_size} parameters) "
        markdown_report += f"trained over {args.num_seeds} seed(s) with different tokenizers:\n"
    # Preamble is generated from the configs that actually ran; it previously hardcoded a list of
    # nine configurations regardless of which two were enabled.
    configs_note_placeholder = "%%CONFIG_LIST%%\n"
    markdown_report += configs_note_placeholder
    markdown_report += f"\nPredefined seed list used: {active_seeds}\n\n"
    markdown_report += (
        "Trained under a standard next-token objective. `± ` denotes the population standard "
        "deviation across seeds.\n\n"
    )

    baseline_mapping = {
        "split": "split",
        "split_functionalizer": "split",
        "llama": "llama",
        "llama_functionalizer": "llama",
    }
    
    tokenizer_names = {
        "split": "Split",
        "split_functionalizer": "Split + Functionalizer",
        "llama": "Llama Split",
        "llama_functionalizer": "Llama Split + Functionalizer",
    }

    tokenizer_types = list(sorted(list(set([exp["tokenizer_type"] for exp in experiments]))))
    
    ds_display_names = {
        "tinystories": "TinyStories",
        "csn-python": "CSN-Python",
        "csn-go": "CSN-Go",
        "fineweb-edu": "FineWeb-Edu",
        "github-code-python": "GitHub-Code-Python",
        "github-code-go": "GitHub-Code-Go",
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
                    inflations.append((cpt / base_cpt - 1) * 100)
            if not inflations: return "N/A"
            if len(inflations) == 1:
                return f"{inflations[0]:+.2f}%"
            import math
            n = len(inflations)
            mean = sum(inflations) / n
            variance = sum((x - mean) ** 2 for x in inflations) / n
            std = math.sqrt(variance)
            return f"{mean:+.2f}% ± {std:.2f}%"
            
        ds_name = ds_display_names.get(ds, ds)
        
        markdown_report += f"## Dataset: {ds_name}\n\n"
        # "Chars/Token Δ vs Baseline" replaces the old "Inflation" header, which named the same
        # quantity with the opposite polarity connotation (a +37.82% "Inflation" is compression).
        markdown_report += "| Tokenizer Type | Vocab Size | Chars/Token | Chars/Token Δ vs Baseline | Tokens/Sec | Chars/Sec | Final Loss | Token PPL | Char PPL |\n"
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

    config_lines = "".join(
        f"{i}. **{tokenizer_names[t]}**\n" for i, t in enumerate(tokenizer_types, start=1)
    )
    markdown_report = markdown_report.replace(configs_note_placeholder, config_lines)

    report_path = f"./results/training_report_{args.model_size}{run_suffix}.md"
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

