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
import ast
import hashlib
import shutil
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

# Collapse is judged on the TOKEN stream: a cycle of up to DEFAULT_COLLAPSE_CYCLE_LEN tokens that
# repeats more than DEFAULT_COLLAPSE_CYCLE_REPEATS times back-to-back is a degenerate loop.
DEFAULT_COLLAPSE_CYCLE_LEN = 20
DEFAULT_COLLAPSE_CYCLE_REPEATS = 4
DEFAULT_REPETITION_MAX_N = 4

# Unicode Private Use Area block used by the Functionalizer for opcodes and parameters.
# Params occupy U+E000-U+E0FF and operators U+E100-U+EFFF, so the whole block is U+E000-U+EFFF.
# Keep this identical to vocab.py -- the two files previously disagreed (0xF0FF vs 0xF8FF).
PUA_START = 0xE000
PUA_END = 0xEFFF

# Resolved once at import so every run records which Go checker was actually used.
GOFMT_PATH = shutil.which("gofmt")

def load_validation_prompts(dataset_name, dataset_config, split, text_column, num_samples, data_dir=None, is_code=False, is_go=False, offset=0):
    print(f"Loading {num_samples} validation prompts from {dataset_name} (split={split}, data_dir={data_dir})...")
    kwargs = {"streaming": True}
    if dataset_config is not None:
        kwargs["name"] = dataset_config
    if data_dir is not None:
        kwargs["data_dir"] = data_dir

    has_only_train = dataset_name in ["HuggingFaceFW/fineweb-edu", "hasankursun/github-code-2025-language-split"]
    actual_split = "train" if (has_only_train and split == "validation") else split

    try:
        ds = load_dataset(dataset_name, split=actual_split, **kwargs)
    except Exception:
        if split == "validation":
            print(f"Split '{split}' not found for {dataset_name}, streaming from 'train' with offset {offset}...")
            actual_split = "train"
            ds = load_dataset(dataset_name, split="train", **kwargs)
        else:
            raise
    if actual_split == "train" and offset > 0:
        ds = ds.skip(offset)
    prompts = []
    targets = []
    
    for item in ds:
        text = item[text_column] if isinstance(item, dict) else None
        if not text or not text.strip():
            continue
            
        if is_code:
            lines = text.split("\n")
            found = False
            for i, line in enumerate(lines[:100]):
                sline = line.strip()
                if is_go:
                    if sline.startswith("func ") and ("{" in sline or not sline.endswith(";")):
                        if line.strip():
                            prompts.append(line)
                            targets.append("\n".join(lines[i+1:]))
                            found = True
                            break
                else:
                    if (sline.startswith("def ") or sline.startswith("class ")) and sline.endswith(":"):
                        if line.strip():
                            prompts.append(line)
                            targets.append("\n".join(lines[i+1:]))
                            found = True
                            break
            if not found:
                continue
        else:
            # Natural language prompt: Take first 50 characters, split at last space to be clean
            if len(text) > 60:
                prompt_raw = text[:50]
                last_space = prompt_raw.rfind(" ")
                if last_space > 20:
                    prompt = prompt_raw[:last_space]
                else:
                    prompt = prompt_raw
                if prompt.strip():
                    target = text[len(prompt):]
                    prompts.append(prompt)
                    targets.append(target)
                
        if len(prompts) >= num_samples:
            break
            
    return prompts, targets


def compute_repetition(text, max_n=DEFAULT_REPETITION_MAX_N):
    """
    Percentage of n-gram content that is a repeat, for each n in 2..max_n.

    For each n, builds every overlapping word n-gram and reports the share that is a duplicate:

        repetition = (1 - unique_ngrams / total_ngrams) * 100

    Example -- "the the the cat went to the the the door" has 9 bigrams of which 6 are distinct,
    so rep-2 = 33.3%.

    This is deliberately complementary to collapse detection. A run like "the the the" repeats
    only three times, so it stays under the collapse rule's repeat threshold; this metric still
    scores it. Collapse catches hard cyclic loops, repetition catches soft degeneracy that
    survives inside otherwise-varied output.

    Splits on whitespace, so runs of indentation do not register as repeated "content" -- those
    are already covered by the collapse and %-empty metrics.

    Returns {n: pct or None}; None when the text is too short to contain two n-grams, so that
    unmeasurable completions are excluded from averages rather than counted as 0%.
    """
    words = text.split()
    out = {}
    for n in range(2, max_n + 1):
        if len(words) < n + 1:
            out[n] = None
            continue
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        out[n] = (1.0 - len(set(grams)) / len(grams)) * 100.0
    return out


class CollapseDetector:
    """
    Streaming detector for degenerate repetition in the generated TOKEN stream.

    Collapse is declared when a cycle of 1..`max_cycle_len` tokens has occurred `cycle_repeats`
    times back to back. With the defaults that is any cycle of up to 20 tokens repeating 4 times --
    a single token emitted 4 times in a row, `a b a b a b a b`, and so on.

    Judging on tokens rather than on the decoded characters is deliberate: the failure being
    detected is a decoding loop, which lives in the token stream. A character-length rule missed
    loops made of short or zero-width tokens entirely -- exactly the "single token loop that was
    never recorded as collapsed" case -- and it could not see multi-token cycles at all.

    The cost is that a token-count rule is not perfectly comparable across tokenizers, since the
    arms differ in chars/token by construction. That is accepted: a repeating cycle is degenerate
    no matter how many characters it prints, and `Avg Chars Pre-Collapse` still reports the
    per-arm character yield.

    Feed tokens one at a time with `push()`. As soon as the threshold is crossed it returns the
    cutoff index -- the END of the cycle's FIRST occurrence, so the one legitimate emission of the
    pattern is kept and only the repeats are discarded. `[1, 2, 3, 6, 6, 6, 6]` cuts off at 4.
    Returns None otherwise. Because it is pure streaming state, the generation loop and the
    post-hoc metric share one definition and cannot disagree.
    """

    def __init__(self, max_cycle_len=DEFAULT_COLLAPSE_CYCLE_LEN,
                 cycle_repeats=DEFAULT_COLLAPSE_CYCLE_REPEATS):
        self.max_cycle_len = max(1, max_cycle_len)
        self.cycle_repeats = cycle_repeats
        # Per period p: how many consecutive positions i satisfy ids[i] == ids[i - p]. A cycle of
        # length p occurring c times spans p * (c - 1) such matches, so "c occurrences" is exactly
        # match_len >= p * (c - 1).
        self._trip = {p: p * max(1, cycle_repeats - 1) for p in range(1, self.max_cycle_len + 1)}
        self._match = {p: 0 for p in range(1, self.max_cycle_len + 1)}
        self._buf = []  # last max_cycle_len tokens, oldest first
        self._n = 0

    def push(self, token_id):
        idx = self._n
        cutoff = None
        for p in range(1, self.max_cycle_len + 1):
            if idx >= p and self._buf[-p] == token_id:
                self._match[p] += 1
            else:
                self._match[p] = 0
            if self._match[p] >= self._trip[p]:
                # First occurrence of the cycle starts p tokens before the matched run; keep that
                # occurrence and cut everything after it.
                start = idx - self._match[p] - p + 1
                if cutoff is None or start + p < cutoff:
                    cutoff = start + p
        self._buf.append(token_id)
        if len(self._buf) > self.max_cycle_len:
            del self._buf[0]
        self._n += 1
        return cutoff


def get_tokens_before_collapse(token_ids, max_cycle_len=DEFAULT_COLLAPSE_CYCLE_LEN,
                               cycle_repeats=DEFAULT_COLLAPSE_CYCLE_REPEATS):
    """
    Number of coherent tokens emitted BEFORE the sequence falls into a repeating loop.

    When small language models finish the response but fail to emit an explicit end-of-sequence
    token ([SEP]), they degenerate into repeating loops. See `CollapseDetector` for the rule.

    Returns the cutoff index -- the end of the repeating cycle's first occurrence -- or
    len(token_ids) if it never collapsed. So `[1, 2, 3, 6, 6, 6, 6]` returns 4: the three coherent
    tokens plus the first `6`, with the three repeats discarded.

    This is the single definition of collapse: the generation loop stops early using the same
    detector, so running this over the truncated sequence reproduces the same k.
    """
    detector = CollapseDetector(max_cycle_len=max_cycle_len, cycle_repeats=cycle_repeats)
    for tok in token_ids:
        start = detector.push(tok)
        if start is not None:
            return start
    return len(token_ids)

def strip_pua(text):
    return "".join([c for c in text if not (PUA_START <= ord(c) <= PUA_END)])

def is_non_empty_content(text):
    if not text:
        return False
    # Strip Unicode Private Use Area (PUA) opcode/parameter characters
    cleaned = strip_pua(text)
    # Strip all whitespaces, newlines, tabs
    cleaned = "".join([c for c in cleaned if not c.isspace()])
    return len(cleaned) > 0

def get_syntax_checker_name(is_code, is_go):
    """Names the checker actually used, so the report never leaves it ambiguous."""
    if not is_code:
        return None
    if not is_go:
        return "python-ast"
    return "gofmt" if GOFMT_PATH else "bracket-fallback"

def python_body_is_nontrivial(full_code):
    """
    Stricter companion to the plain ast.parse check.

    A bare identifier or a lone docstring parses cleanly, so `ast.parse` alone answers only
    "did the model emit any indented token?". This requires the function body to contain at
    least one statement that is not `pass`, not a docstring, and not a bare name reference.
    """
    try:
        tree = ast.parse(full_code)
    except (SyntaxError, ValueError, RecursionError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body:
                if isinstance(stmt, ast.Pass):
                    continue
                # Docstring or a bare `foo` expression statement -- syntactically fine, empty.
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.Name)):
                    continue
                return True
    return False

def test_syntax_validity(prompt, completion, is_go=False):
    if is_go:
        # Check Go syntax using gofmt if available, otherwise fallback to bracket matching.
        import subprocess
        import tempfile

        # Joined with "\n" to match the Python path; these were previously inconsistent
        # (Go used " "), which made the two languages' success criteria differ subtly.
        full_code = "package main\n\n" + prompt + "\n" + completion
        if not full_code.strip().endswith("}"):
            full_code += "\n}"

        if GOFMT_PATH:
            with tempfile.NamedTemporaryFile("w", suffix=".go", delete=False) as f:
                f.write(full_code)
                temp_name = f.name
            try:
                res_check = subprocess.run([GOFMT_PATH, "-e", temp_name], capture_output=True)
                return res_check.returncode == 0
            finally:
                os.remove(temp_name)

        # Fallback only reached when gofmt is genuinely absent. NOTE: this is a far weaker
        # test (balanced brackets only; ignores // and /* */ comments and treats ' as a string
        # delimiter when Go uses it for runes). run_evaluation records which path ran.

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
        except (SyntaxError, ValueError, RecursionError):
            # ValueError covers source containing null bytes; RecursionError covers
            # pathologically nested generations. Both previously escaped and killed the run.
            return False

def checkpoint_dir_for(dataset, tokenizer_type, seed, model_size, run_tag=""):
    """Single definition of the checkpoint layout, shared by the evaluator and the sweep loop."""
    run_suffix = f"_{run_tag}" if run_tag else ""
    return f"./checkpoints/{dataset}_{tokenizer_type}_seed{seed}_{model_size}{run_suffix}"


def tokenizer_digest(tokenizer_path):
    """SHA-256 of a tokenizer.json, used to prove two checkpoints really share a tokenizer."""
    h = hashlib.sha256()
    with open(tokenizer_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TokenizerBundle:
    """
    A loaded tokenizer plus everything derived from it that does not depend on the model.

    Every seed of a given (dataset, tokenizer_type) is trained with the identical tokenizer -- the
    BPE is fit on the same first 10k texts and the seed only touches model init and data order --
    so `tokenizer.json` is byte-identical across the seed checkpoints. Loading it, looking up
    `[SEP]`, and encoding the evaluation prompts once per arm instead of once per seed removes
    (seeds - 1) x (one tokenizer build + one encode of every prompt) from a sweep.

    `digest` exists so the reuse is checked rather than assumed: `matches()` re-hashes the
    candidate checkpoint's tokenizer.json, and a mismatch means the bundle is not valid for that
    checkpoint and must be rebuilt.
    """

    def __init__(self, tokenizer, digest, prompt_ids):
        self.tokenizer = tokenizer
        self.digest = digest
        # Encoded once here; run_evaluation only slices these to the model's context window.
        self.prompt_ids = prompt_ids
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def matches(self, tokenizer_path):
        return self.digest == tokenizer_digest(tokenizer_path)


def load_tokenizer_bundle(checkpoint_dir, prompts):
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    prompt_ids = [tokenizer.encode(p).ids for p in prompts]
    return TokenizerBundle(tokenizer, tokenizer_digest(tokenizer_path), prompt_ids)


def run_evaluation(dataset, tokenizer_type, seed, prompts, is_code=False, is_go=False, model_size="125M", run_tag="", max_new_tokens=1000,
                   collapse_cycle_len=DEFAULT_COLLAPSE_CYCLE_LEN,
                   collapse_cycle_repeats=DEFAULT_COLLAPSE_CYCLE_REPEATS,
                   repetition_max_n=DEFAULT_REPETITION_MAX_N,
                   tokenizer_bundle=None):
    """
    Evaluate one checkpoint. `tokenizer_bundle` is an optional TokenizerBundle from a previous
    seed of the same arm; it is used only after its digest is verified against this checkpoint's
    tokenizer.json, and rebuilt otherwise. Passing nothing is always safe, just slower.
    """
    checkpoint_dir = checkpoint_dir_for(dataset, tokenizer_type, seed, model_size, run_tag)

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} not found. Skipping seed {seed} for {tokenizer_type}.")
        return None

    print(f"\nEvaluating downstream task for {dataset} ({tokenizer_type})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse the tokenizer across seeds when the checkpoints genuinely carry the same one. The
    # digest check is a file hash against a full tokenizer build plus one encode per prompt, so
    # verifying is far cheaper than rebuilding -- and a diverged tokenizer would silently corrupt
    # every metric, so it is never assumed.
    tokenizer_reused = False
    if tokenizer_bundle is not None and len(tokenizer_bundle.prompt_ids) == len(prompts):
        if tokenizer_bundle.matches(os.path.join(checkpoint_dir, "tokenizer.json")):
            tokenizer_reused = True
        else:
            print(f"  Tokenizer for seed {seed} differs from the cached one; reloading for this seed.")
    if not tokenizer_reused:
        tokenizer_bundle = load_tokenizer_bundle(checkpoint_dir, prompts)

    tokenizer = tokenizer_bundle.tokenizer
    sep_id = tokenizer_bundle.sep_id

    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    # Hard ceiling from the model's learned position embeddings. Feeding a longer sequence raises
    # "IndexError: index out of range in self", so once the running sequence reaches this length
    # we slide the window and keep the most recent tokens. Generation can therefore run past the
    # trained context, but tokens beyond it are conditioned on a truncated history.
    max_ctx = model.config.n_positions
    window_hit_count = 0
    collapsed_count = 0
    # Per-prompt repetition, kept per n so unmeasurably short completions can be excluded
    # rather than averaged in as 0%.
    repetition_samples = {n: [] for n in range(2, repetition_max_n + 1)}

    # Only the pre-collapse prefix is measured. Totals over the full stream are meaningless now
    # that generation is cut at the first repeat of the collapsing cycle.
    total_tokens_before_collapse = 0
    total_chars_before_collapse = 0
    total_generation_time = 0.0

    total_coherent_tokens = 0
    total_coherent_chars = 0
    total_coherent_time = 0.0

    syntax_success_count = 0
    collapsed_syntax_failure_count = 0
    syntax_nontrivial_count = 0
    non_empty_count = 0
    decode_fallback_count = 0
    success_sample = None
    failure_sample = None
    # Datasets with no pass/fail notion (prose) still need something to show, so we always keep
    # the first couple of generations as plain examples.
    example_samples = []
    MAX_EXAMPLE_SAMPLES = 2

    def safe_decode(ids):
        """
        Decode, falling back to raw token strings if the Functionalizer decoder rejects a
        malformed generated opcode stream.

        The fallback previously joined with " ", inventing separator characters that were
        never generated and inflating every char-based metric -- and it can only ever fire on
        the Functionalizer arm, since the baseline's Fuse() decoder cannot fail. It now joins
        with "" and strips PUA so the fallback cannot manufacture visible characters.
        Returns (text, used_fallback).
        """
        try:
            return tokenizer.decode(ids), False
        except Exception:
            raw_tokens = []
            for tid in ids:
                tok = tokenizer.id_to_token(tid)
                if tok is not None:
                    raw_tokens.append(tok)
            return strip_pua("".join(raw_tokens)), True


    for prompt, encoded_prompt_ids in zip(prompts, tokenizer_bundle.prompt_ids):
        # Encoding happened once when the bundle was built; only the model-dependent truncation
        # below is per-seed work.
        prompt_ids = encoded_prompt_ids
        if not prompt_ids:
            continue
        if len(prompt_ids) > max_ctx:
            # Keep the tail so the signature/sentence end nearest the continuation survives.
            prompt_ids = prompt_ids[-max_ctx:]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        hit_window = False
        
        # Greedy generation loop to measure exact tokens and characters time
        start_time = time.time()
        curr_input = input_ids
        generated_ids = []
        all_ids = list(prompt_ids)

        # KV cache: after the prompt, only the newest token is fed each step, so a step costs
        # O(1) attention against the cache instead of a full O(context) forward pass.
        past_key_values = None
        past_len = 0

        # Collapse tracking, using exactly the detector get_tokens_before_collapse runs so the
        # early stop and the post-hoc metric can never disagree.
        detector = CollapseDetector(max_cycle_len=collapse_cycle_len,
                                    cycle_repeats=collapse_cycle_repeats)
        collapsed = False

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                # Sliding window: the position embeddings only cover max_ctx positions, so when
                # the cache is full we drop it and re-prime from the most recent tokens. That
                # costs one full forward roughly every max_ctx steps rather than every step.
                if past_len + curr_input.shape[1] > max_ctx:
                    curr_input = torch.tensor([all_ids[-(max_ctx - 1):]], dtype=torch.long).to(device)
                    past_key_values = None
                    past_len = 0
                    hit_window = True

                outputs = model(curr_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                past_len += curr_input.shape[1]

                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)

                token_val = next_token_id.item()
                generated_ids.append(token_val)
                all_ids.append(token_val)

                if token_val == sep_id:
                    break

                if detector.push(token_val) is not None:
                    # Everything after this point is a known-degenerate loop that the
                    # pre-collapse metrics discard anyway, so stop paying to generate it.
                    collapsed = True
                    break

                curr_input = next_token_id.unsqueeze(0)

        if hit_window:
            window_hit_count += 1
        if collapsed:
            collapsed_count += 1
            
        generation_time = time.time() - start_time
        
        # Decode completion
        completion, used_fallback = safe_decode(generated_ids)

        # Calculate coherent prefix metrics. The loop above already stopped at collapse, so this
        # re-derives the same k from the truncated sequence via the single shared definition.
        k = get_tokens_before_collapse(generated_ids, max_cycle_len=collapse_cycle_len,
                                       cycle_repeats=collapse_cycle_repeats)
        coherent_completion, coherent_used_fallback = safe_decode(generated_ids[:k])
        if used_fallback or coherent_used_fallback:
            decode_fallback_count += 1


        coherent_time = generation_time * (k / len(generated_ids)) if len(generated_ids) > 0 else 0.0
        
        total_tokens_before_collapse += k
        total_chars_before_collapse += len(coherent_completion)
        total_generation_time += generation_time
        
        total_coherent_tokens += k
        total_coherent_chars += len(coherent_completion)
        total_coherent_time += coherent_time
        
        # Repetition over the full returned text for this prompt.
        for n, pct in compute_repetition(completion, max_n=repetition_max_n).items():
            if pct is not None:
                repetition_samples[n].append(pct)

        is_non_empty = is_non_empty_content(completion)
        is_success = False
        if is_code:
            # A collapsed generation is a failure by definition: the model stopped producing code
            # and started looping, so whatever it emitted before that is not a completed function.
            # Checked before the parser both because it is the honest verdict and because it skips
            # the gofmt subprocess for generations already known to have failed.
            if collapsed:
                collapsed_syntax_failure_count += 1
            is_success = (
                is_non_empty and not collapsed
                and test_syntax_validity(prompt, completion, is_go=is_go)
            )
            if is_success:
                syntax_success_count += 1
            # Stricter Python-only variant: body must contain a real statement (see P1-5).
            if is_success and not is_go and python_body_is_nontrivial(prompt + "\n" + completion):
                syntax_nontrivial_count += 1


        if not is_non_empty:
            # Empty completion counts as a failure, do not increment non_empty_count
            pass
        else:
            non_empty_count += 1
            
        # Collect success and failure samples
        sample_data = {
            "prompt": prompt,
            "completion": completion,
            # Recorded per sample so a failure example makes clear whether it failed on syntax or
            # was auto-failed for collapsing.
            "collapsed": collapsed,
            "tokens_pre_collapse": k,
            "tokens": [tokenizer.id_to_token(tid) for tid in generated_ids[:20]] + (["..."] if len(generated_ids) > 20 else [])
        }
        if len(example_samples) < MAX_EXAMPLE_SAMPLES:
            example_samples.append(sample_data)

        if is_code:
            if is_success and success_sample is None:
                success_sample = sample_data
            elif not is_success and failure_sample is None:
                failure_sample = sample_data
            
    pct_non_empty = (non_empty_count / len(prompts)) * 100
    avg_tokens_before_collapse = total_tokens_before_collapse / len(prompts)
    avg_chars_before_collapse = total_chars_before_collapse / len(prompts)
    
    # Speed is computed based on coherent prefix to be fair and accurate
    tokens_per_sec = total_coherent_tokens / total_coherent_time if total_coherent_time > 0 else 0
    chars_per_sec = total_coherent_chars / total_coherent_time if total_coherent_time > 0 else 0
    
    success_rate = (syntax_success_count / len(prompts)) * 100 if is_code else None
    success_metric_label = "Syntax Success Rate (%)" if is_code else None
    # Stricter Python-only rate; None for Go, where the gofmt check has no direct analogue.
    nontrivial_rate = (syntax_nontrivial_count / len(prompts)) * 100 if (is_code and not is_go) else None
    # Share of non-empty completions that parse -- computed per seed so the paper does not have
    # to divide one cross-seed mean by another (see P1-7).
    success_rate_of_non_empty = (
        (syntax_success_count / non_empty_count) * 100 if (is_code and non_empty_count > 0) else None
    )
    pct_decode_fallback = (decode_fallback_count / len(prompts)) * 100

    # Mean repetition per n across the prompts long enough to measure, plus a headline figure
    # averaging those means over n = 2..repetition_max_n.
    repetition_by_n = {
        n: (sum(vals) / len(vals) if vals else None)
        for n, vals in repetition_samples.items()
    }
    repetition_measurable = {n: len(vals) for n, vals in repetition_samples.items()}
    _rep_means = [v for v in repetition_by_n.values() if v is not None]
    repetition_pct = sum(_rep_means) / len(_rep_means) if _rep_means else None

    results = {
        "dataset": dataset,
        "tokenizer_type": tokenizer_type,
        "seed": seed,
        "num_prompts": len(prompts),
        "syntax_checker": get_syntax_checker_name(is_code, is_go),
        # False on the first seed of an arm (which builds the tokenizer) and on any seed whose
        # tokenizer.json failed the digest check and had to be rebuilt.
        "tokenizer_reused_from_cache": tokenizer_reused,
        "max_new_tokens": max_new_tokens,
        "model_context_window": max_ctx,
        "collapse_cycle_len": collapse_cycle_len,
        "collapse_cycle_repeats": collapse_cycle_repeats,
        "repetition_max_n": repetition_max_n,
        # Repetition over the full returned text. `repetition_pct` is the headline (mean over
        # n=2..max_n); `repetition_by_n` keeps the per-n breakdown, e.g. {"2": 33.3, ...}.
        "repetition_pct": repetition_pct,
        "repetition_2gram": repetition_by_n.get(2),
        "repetition_by_n": {str(n): v for n, v in repetition_by_n.items()},
        "repetition_measurable_prompts": {str(n): c for n, c in repetition_measurable.items()},
        # Share of prompts stopped early by collapse detection (rather than [SEP] or the budget).
        "pct_collapsed": (collapsed_count / len(prompts)) * 100,
        # Share of prompts whose generation ran past the model's trained context window.
        "pct_hit_context_window": (window_hit_count / len(prompts)) * 100,
        # Pre-collapse only: generation is cut at the first repeat of the collapsing cycle, so a
        # total over the full stream would just be this plus a fixed slice of loop.
        "avg_tokens_before_collapse": avg_tokens_before_collapse,
        "avg_chars_before_collapse": avg_chars_before_collapse,
        "tokens_per_sec": tokens_per_sec,
        "chars_per_sec": chars_per_sec,
        "pct_non_empty": pct_non_empty,
        "pct_empty": 100.0 - pct_non_empty,
        "pct_decode_fallback": pct_decode_fallback,
        "decode_fallback_count": decode_fallback_count,
        "success_rate": success_rate,
        # Share of prompts auto-failed for collapsing (code datasets only); these are counted as
        # failures in `success_rate` without ever reaching the syntax checker.
        "pct_collapsed_syntax_failure": (
            (collapsed_syntax_failure_count / len(prompts)) * 100 if is_code else None
        ),
        "success_rate_nontrivial": nontrivial_rate,
        "success_rate_of_non_empty": success_rate_of_non_empty,
        "success_metric_label": success_metric_label,
        "success_sample": success_sample,
        "failure_sample": failure_sample,
        "example_samples": example_samples
    }
    
    print(json.dumps(results, indent=2))
    return results


def load_checkpoint_results(checkpoint_path):
    """
    Load previously saved evaluation results from JSON if the file exists and is valid.
    Returns a dict {dataset_name: {tokenizer_type: [result_dict, ...]}}.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"Warning: Failed to load existing checkpoint from {checkpoint_path}: {e}")
    return {}


def save_checkpoint_results(checkpoint_path, all_results):
    """
    Atomically save all_results to JSON checkpoint file using a temporary file.
    """
    if not checkpoint_path:
        return
    dir_name = os.path.dirname(checkpoint_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    temp_path = f"{checkpoint_path}.tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(all_results, f, indent=2)
        os.replace(temp_path, checkpoint_path)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint to {checkpoint_path}: {e}")


def find_checkpointed_result(all_results, dataset_name, tokenizer_type, seed, args):
    """
    Find a matching evaluated test result in all_results for (dataset, tokenizer_type, seed).
    Checks that critical evaluation settings (max_new_tokens, collapse settings, num_prompts) match.
    """
    if not all_results or not isinstance(all_results, dict):
        return None
    ds_data = all_results.get(dataset_name)
    if not ds_data or not isinstance(ds_data, dict):
        return None
    tok_data = ds_data.get(tokenizer_type)
    if not tok_data or not isinstance(tok_data, list):
        return None
    for r in tok_data:
        if not isinstance(r, dict):
            continue
        if r.get("seed") == seed and r.get("dataset") == dataset_name and r.get("tokenizer_type") == tokenizer_type:
            # Verify compatibility with current execution parameters
            if r.get("max_new_tokens") is not None and r.get("max_new_tokens") != args.max_new_tokens:
                continue
            if r.get("collapse_cycle_len") is not None and r.get("collapse_cycle_len") != args.collapse_cycle_len:
                continue
            if r.get("collapse_cycle_repeats") is not None and r.get("collapse_cycle_repeats") != args.collapse_cycle_repeats:
                continue
            if r.get("num_prompts") is not None and r.get("num_prompts") != args.num_prompts:
                continue
            if r.get("repetition_max_n") is not None and r.get("repetition_max_n") != args.repetition_max_n:
                continue
            # Ensure essential metrics exist in the record
            if "avg_tokens_before_collapse" in r and "pct_collapsed" in r:
                return r
    return None


def generate_and_save_reports(all_results, args, active_seeds, tokenizer_types, downstream_report_path, downstream_json_path, downstream_samples_path, run_suffix):
    """
    Generate markdown reports and persist json results from all_results.
    """
    report = "# Downstream Tasks Evaluation Report\n\n"
    if args.model_size == "25M":
        report += "This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) "
        report += f"for the models trained under the different tokenization configurations across {args.num_seeds} seed(s).\n\n"
    elif args.model_size == "125M":
        report += "This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) "
        report += f"for the 125M models trained under the different tokenization configurations across {args.num_seeds} seed(s).\n\n"
    elif args.model_size == "2B":
        report += "This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) "
        report += f"for the 2B models trained under the different tokenization configurations across {args.num_seeds} seed(s).\n\n"
    else:
        report += "This report evaluates the inference-time performance (latency, token length efficiency, and generation success metrics) "
        report += f"for the {args.model_size} models trained under the different tokenization configurations across {args.num_seeds} seed(s).\n\n"
    report += f"Predefined seed list used: {active_seeds}\n\n"

    ctx_windows = sorted({
        r["model_context_window"]
        for d in all_results.values() if isinstance(d, dict)
        for lst in d.values() if isinstance(lst, list)
        for r in lst if isinstance(r, dict) and r.get("seed") in active_seeds and "model_context_window" in r
    })
    report += f"Greedy generation budget: **{args.max_new_tokens} new tokens** per prompt (stops early on `[SEP]`).\n\n"
    if ctx_windows and args.max_new_tokens > min(ctx_windows):
        report += (
            f"> ⚠️ The budget exceeds the model's trained context window ({', '.join(str(c) for c in ctx_windows)} "
            f"positions). Generation past that point slides the window and keeps the most recent tokens, so those "
            f"tokens are conditioned on a truncated history rather than the full generation. "
            f"**% Past Context** below reports how often this happened.\n\n"
        )
    
    report += "### Note on Token Collapse\n"
    report += "When small models finish generating content without emitting an explicit end token (`[SEP]`), they frequently fall into repetition loops. "
    report += f"Collapse is judged on the **generated token stream**: generation **stops early** once a cycle of up to **{args.collapse_cycle_len} tokens** has occurred **{args.collapse_cycle_repeats} times** back to back (a single token emitted {args.collapse_cycle_repeats} times in a row, `a b a b ...`, and so on). "
    report += "The rule is applied to tokens rather than to decoded characters on purpose: the failure being detected is a decoding loop, which lives in the token stream, and a character-length rule silently missed loops built from short or zero-width tokens as well as every multi-token cycle. The trade-off is that a token-count rule is not perfectly comparable across tokenizers, since the arms differ in chars/token by construction; a repeating cycle is degenerate regardless of how many characters it prints, and **Avg Chars Pre-Collapse** still reports the per-arm character yield. "
    report += "The cutoff is the **end of the cycle's first occurrence**, so the one legitimate emission of the pattern is kept and only the repeats are discarded: `[1, 2, 3, 6, 6, 6, 6]` counts as **4 tokens pre-collapse**. "
    report += "**Avg Tokens Pre-Collapse** and **Avg Chars Pre-Collapse** measure that prefix, and generation throughput (`Tokens/Sec`, `Chars/Sec`) is computed exclusively over it. "
    report += "Totals over the full generated stream are no longer reported: since generation is killed at the collapse point, a total would be the pre-collapse figure plus a fixed slice of loop, which measures the cutoff rule rather than the model.\n\n"
    report += "**% Collapsed** reports how often the early stop fired, as opposed to `[SEP]` or the token budget. "
    report += "**On code datasets a collapsed generation is scored as a syntax failure automatically**, without being handed to the syntax checker: the model stopped emitting code and started looping, so the fragment before the loop is not a completed function. "
    report += "**% Auto-Failed (Collapse)** reports the share of prompts failed this way; it is a component of the failures behind the syntax success rate, not an additional penalty on top of it.\n\n"
    report += "Generation uses a KV cache, so per-token decode cost is O(1) against the cache rather than a full forward pass over the context. "
    report += "`Tokens/Sec` is therefore **not comparable to earlier runs** that recomputed the whole sequence each step.\n\n"

    report += "### Note on Syntax Checking\n"
    checkers = sorted({
        r["syntax_checker"]
        for d in all_results.values() if isinstance(d, dict)
        for lst in d.values() if isinstance(lst, list)
        for r in lst if isinstance(r, dict) and r.get("seed") in active_seeds and r.get("syntax_checker")
    })
    report += f"Syntax checkers used in this run: `{', '.join(checkers) if checkers else 'none'}`. "
    report += "`gofmt` means real Go parsing; `bracket-fallback` means balanced-bracket matching only, which is far more permissive and NOT comparable to the Python results. "
    report += "**Non-Trivial Syntax** (Python only) additionally requires the parsed function body to contain a statement that is not `pass`, a docstring, or a bare identifier.\n\n"

    report += "### Note on Repetition\n"
    report += (
        f"**Repetition (%)** measures how much of the returned text is repeated content. For each n in "
        f"2..{args.repetition_max_n}, every overlapping word n-gram is built and the share that is a duplicate is reported "
        f"(`1 - unique/total`); the headline figure averages those across n, and **Repetition 2-gram (%)** shows n=2 alone. "
        f"For example `the the the cat went to the the the door` has 9 bigrams of which 6 are distinct, so rep-2 = 33.3%.\n\n"
    )
    report += (
        "This is complementary to collapse detection rather than a restatement of it: `the the the` repeats only three "
        f"times, well under the {args.collapse_cycle_repeats}-repeat collapse rule, yet it is clearly degenerate. Collapse catches hard "
        "short-cycle loops; repetition catches soft degeneracy inside otherwise-varied output. "
        "Text is split on whitespace, so runs of indentation do not count as repeated content (those are already covered "
        "by **% Collapsed** and **% Empty**). Completions too short to contain two n-grams are excluded from the average "
        "rather than scored as 0%.\n\n"
    )

    report += "### Note on Decode Fallback\n"
    report += "**% Decode Fallback** is the share of prompts where the tokenizer's decoder rejected the generated stream and raw token strings were used instead. "
    report += "This can only occur for Functionalizer configurations (malformed opcode streams); a non-zero value means the corresponding character metrics are approximate.\n\n"
    
    for ds_name, tok_results_dict in all_results.items():
        if not isinstance(tok_results_dict, dict):
            continue
        active_tok_results_dict = {
            t: [r for r in lst if isinstance(r, dict) and r.get("seed") in active_seeds]
            for t, lst in tok_results_dict.items() if isinstance(lst, list)
        }
        has_any_results = any(len(lst) > 0 for lst in active_tok_results_dict.values())
        if not has_any_results:
            continue
            
        if ds_name in ["csn-python", "github-code-python"]:
            ds_label = f"{ds_name} (Code Generation - Python)"
        elif ds_name in ["csn-go", "github-code-go"]:
            ds_label = f"{ds_name} (Code Generation - Go)"
        elif ds_name in ["fineweb-edu", "fineweb"]:
            ds_label = "FineWeb-Edu (Natural Language Generation)"
        else:
            ds_label = f"{ds_name} (Natural Language Generation)"
            
        has_success_metric = False
        metric_label = None
        for tok_type, r_list in active_tok_results_dict.items():
            if r_list and r_list[0].get("success_metric_label"):
                has_success_metric = True
                metric_label = r_list[0]["success_metric_label"]
                break
                
        report += f"## Dataset: {ds_label}\n\n"
        if has_success_metric:
            report += f"| Tokenizer Type | Avg Tokens Pre-Collapse | Avg Chars Pre-Collapse | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | % Decode Fallback | % Past Context | % Collapsed | % Auto-Failed (Collapse) | Repetition (%) | Repetition 2-gram (%) | {metric_label} | Min {metric_label} | Max {metric_label} | Non-Trivial Syntax (%) | Syntax (% of Non-Empty) |\n"
            report += "|---|" + "---|" * 16 + "\n"
        else:
            report += f"| Tokenizer Type | Avg Tokens Pre-Collapse | Avg Chars Pre-Collapse | Tokens / Sec | Chars / Sec (Text Speed) | % Empty | % Decode Fallback | % Past Context | % Collapsed | Repetition (%) | Repetition 2-gram (%) |\n"
            report += "|---|" + "---|" * 10 + "\n"
        
        def get_stat(r_list, key, fmt="{:.1f}"):
            # `± ` is the POPULATION standard deviation across seeds (divides by n, not n-1).
            values = [r[key] for r in r_list if r is not None and r.get(key) is not None]
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

        def get_min_stat(r_list, key, fmt="{:.1f}"):
            values = [r[key] for r in r_list if r is not None and r.get(key) is not None]
            if not values:
                return "N/A"
            return fmt.format(min(values))

        def get_max_stat(r_list, key, fmt="{:.1f}"):
            values = [r[key] for r in r_list if r is not None and r.get(key) is not None]
            if not values:
                return "N/A"
            return fmt.format(max(values))
            
        tok_name_map = {
            "split": "Split Only",
            "split_functionalizer": "Split + Functionalizer",
            "llama": "Llama Split Only",
            "llama_functionalizer": "Llama Split + Functionalizer",
        }
        tok_order = [t for t in ["split", "split_functionalizer", "llama", "llama_functionalizer"] if t in tokenizer_types or t in active_tok_results_dict]
        for t in active_tok_results_dict:
            if t not in tok_order:
                tok_order.append(t)

        for tok_type in tok_order:
            r_list = active_tok_results_dict.get(tok_type, [])
            pretty_name = tok_name_map.get(tok_type, tok_type)
            if not r_list:
                if has_success_metric:
                    report += f"| **{pretty_name}** |" + " N/A |" * 16 + "\n"
                else:
                    report += f"| **{pretty_name}** |" + " N/A |" * 10 + "\n"
                continue
                
            if has_success_metric:
                report += (
                    f"| **{pretty_name}** | "
                    f"{get_stat(r_list, 'avg_tokens_before_collapse', '{:.1f}')} | "
                    f"{get_stat(r_list, 'avg_chars_before_collapse', '{:.1f}')} | "
                    f"{get_stat(r_list, 'tokens_per_sec', '{:.1f}')} | "
                    f"{get_stat(r_list, 'chars_per_sec', '{:.1f}')} | "
                    f"{get_stat(r_list, 'pct_empty', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_decode_fallback', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_hit_context_window', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_collapsed', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_collapsed_syntax_failure', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'repetition_pct', '{:.1f}')} | "
                    f"{get_stat(r_list, 'repetition_2gram', '{:.1f}')} | "
                    f"{get_stat(r_list, 'success_rate', '{:.2f}')}% | "
                    f"{get_min_stat(r_list, 'success_rate', '{:.2f}')}% | "
                    f"{get_max_stat(r_list, 'success_rate', '{:.2f}')}% | "
                    f"{get_stat(r_list, 'success_rate_nontrivial', '{:.2f}')} | "
                    f"{get_stat(r_list, 'success_rate_of_non_empty', '{:.2f}')} |\n"
                )
            else:
                report += (
                    f"| **{pretty_name}** | "
                    f"{get_stat(r_list, 'avg_tokens_before_collapse', '{:.1f}')} | "
                    f"{get_stat(r_list, 'avg_chars_before_collapse', '{:.1f}')} | "
                    f"{get_stat(r_list, 'tokens_per_sec', '{:.1f}')} | "
                    f"{get_stat(r_list, 'chars_per_sec', '{:.1f}')} | "
                    f"{get_stat(r_list, 'pct_empty', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_decode_fallback', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_hit_context_window', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'pct_collapsed', '{:.1f}')}% | "
                    f"{get_stat(r_list, 'repetition_pct', '{:.1f}')} | "
                    f"{get_stat(r_list, 'repetition_2gram', '{:.1f}')} |\n"
                )
        report += "\n"
        
    os.makedirs(os.path.dirname(downstream_report_path), exist_ok=True)
    with open(downstream_report_path, "w") as f:
        f.write(report)

    # Persist raw per-seed results so cross-seed statistics can be recomputed without re-running
    # generation (previously only the aggregated markdown survived).
    save_checkpoint_results(downstream_json_path, all_results)
        
    # Generate downstream samples report
    samples_report = f"# Downstream Generation Samples Report ({args.model_size}{run_suffix})\n\n"
    samples_report += "This report showcases a comparison of generated outputs between the different tokenizer configurations.\n\n"
    samples_report += (
        "For code datasets, samples are labelled by whether the generation passed the syntax check; a collapsed "
        "generation is a failure regardless of what the checker would have said, so each sample also reports "
        "whether it collapsed. "
        "Prose generation has no pass/fail criterion, so two representative generations are shown instead.\n\n"
    )

    def render_sample(heading, s):
        block = f"#### {heading}\n"
        block += f"* **Prompt**: `{repr(s['prompt'])}`\n"
        block += f"* **Generated Tokens (first 20)**: `{s['tokens']}`\n"
        block += f"* **Decoded Completion**: `{repr(s['completion'])}`\n"
        block += f"* **Completion Length (chars)**: {len(s['completion'])}\n"
        # Makes clear whether a code failure was a syntax failure or a collapse auto-failure.
        block += f"* **Collapsed**: {'yes' if s.get('collapsed') else 'no'}"
        if s.get("tokens_pre_collapse") is not None:
            block += f" (tokens pre-collapse: {s['tokens_pre_collapse']})"
        block += "\n\n"
        return block
    
    for ds_name, tok_results_dict in all_results.items():
        if not isinstance(tok_results_dict, dict):
            continue
        active_tok_results_dict = {
            t: [r for r in lst if isinstance(r, dict) and r.get("seed") in active_seeds]
            for t, lst in tok_results_dict.items() if isinstance(lst, list)
        }
        has_any_results = any(len(lst) > 0 for lst in active_tok_results_dict.values())
        if not has_any_results:
            continue
            
        if ds_name in ["csn-python", "github-code-python"]:
            ds_label = f"{ds_name} (Code Generation - Python)"
        elif ds_name in ["csn-go", "github-code-go"]:
            ds_label = f"{ds_name} (Code Generation - Go)"
        elif ds_name in ["fineweb-edu", "fineweb"]:
            ds_label = "FineWeb-Edu (Natural Language Generation)"
        else:
            ds_label = f"{ds_name} (Natural Language Generation)"
            
        samples_report += f"## Dataset: {ds_label}\n\n"
        
        tok_order = [t for t in ["split", "split_functionalizer", "llama", "llama_functionalizer"] if t in tokenizer_types or t in active_tok_results_dict]
        for t in active_tok_results_dict:
            if t not in tok_order:
                tok_order.append(t)

        for tok_type in tok_order:
            r_list = active_tok_results_dict.get(tok_type, [])
            if not r_list:
                continue
            first_run = r_list[0]
            examples = first_run.get("example_samples") or []
            is_code_ds = first_run.get("success_metric_label") is not None
            if not is_code_ds and not examples:
                continue

            pretty_name = tok_name_map.get(tok_type, tok_type)
            samples_report += f"### Configuration: {pretty_name}\n\n"

            if is_code_ds:
                # Code: label by syntax-check outcome.
                if first_run.get("success_sample"):
                    samples_report += render_sample("Success Sample", first_run["success_sample"])
                else:
                    samples_report += "#### Success Sample\n*No success sample found for this configuration.*\n\n"

                if first_run.get("failure_sample"):
                    samples_report += render_sample("Failure Sample", first_run["failure_sample"])
                else:
                    samples_report += "#### Failure Sample\n*No failure sample found for this configuration.*\n\n"
            else:
                # Prose: no pass/fail criterion exists, so show representative generations.
                for i, s in enumerate(examples, start=1):
                    samples_report += render_sample(f"Example Generation {i}", s)

            samples_report += "---\n\n"
            
    os.makedirs(os.path.dirname(downstream_samples_path), exist_ok=True)
    with open(downstream_samples_path, "w") as f:
        f.write(samples_report)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--num_prompts", type=int, default=1000)
    parser.add_argument("--model_size", type=str, default="125M", choices=["25M", "125M", "2B"])
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Greedy generation budget per prompt. Generation stops early on [SEP] or at collapse. "
             "This budget may EXCEED the token training window, so any generation running past it "
             "slides the window and conditions on a truncated history; the %% Past Context column "
             "reports how often that actually happens."
    )
    parser.add_argument(
        "--collapse_cycle_len", type=int, default=DEFAULT_COLLAPSE_CYCLE_LEN,
        help=f"Longest token cycle treated as a collapse loop (default {DEFAULT_COLLAPSE_CYCLE_LEN}). "
             "Covers single-token loops plus multi-token cycles such as `a b a b ...`."
    )
    parser.add_argument(
        "--collapse_cycle_repeats", type=int, default=DEFAULT_COLLAPSE_CYCLE_REPEATS,
        help="Generation stops once a cycle of up to --collapse_cycle_len tokens has occurred this "
             f"many times back to back (default {DEFAULT_COLLAPSE_CYCLE_REPEATS}). Measured on "
             "generated tokens, not characters. Only the cycle's first occurrence is kept: "
             "[1, 2, 3, 6, 6, 6, 6] counts as 4 tokens pre-collapse. On code datasets a collapsed "
             "generation is scored as a syntax failure automatically."
    )
    parser.add_argument(
        "--repetition_max_n", type=int, default=DEFAULT_REPETITION_MAX_N,
        help="Largest n for the n-gram repetition metric; measured for n = 2..this value over the "
             "full returned text."
    )
    # Must match the --run_tag used for training so the correct checkpoints are loaded.
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument(
        "--no_checkpoint", action="store_true",
        help="Ignore existing checkpoint results and re-run all evaluations from scratch."
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default=None,
        help="Path to JSON file for loading/saving intermediate checkpoints. Defaults to ./results/downstream_results_{model_size}{run_suffix}.json"
    )
    args = parser.parse_args()

    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    downstream_json_path = args.checkpoint_file or f"./results/downstream_results_{args.model_size}{run_suffix}.json"
    downstream_report_path = f"./results/downstream_report_{args.model_size}{run_suffix}.md"
    downstream_samples_path = f"./results/downstream_samples_{args.model_size}{run_suffix}.md"

    # Must stay identical to run_experiments.PREDEFINED_SEEDS or checkpoints will not be found.
    PREDEFINED_SEEDS = [1, 2, 3, 4, 5]
    active_seeds = PREDEFINED_SEEDS[:args.num_seeds]

    datasets = [
        # {"name": "tinystories", "path": "roneneldan/TinyStories", "config": None, "text_column": "text", "is_code": False, "is_go": False, "offset": 0},
        # {"name": "csn-python", "path": "code-search-net/code_search_net", "config": "python", "text_column": "func_code_string", "is_code": True, "is_go": False, "offset": 0}, 
        # {"name": "csn-go", "path": "code-search-net/code_search_net", "config": "go", "text_column": "func_code_string", "is_code": True, "is_go": True, "offset": 0}
        {"name": "fineweb-edu", "path": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "text_column": "text", "is_code": False, "is_go": False, "offset": 500000},
        {"name": "github-code-python", "path": "hasankursun/github-code-2025-language-split", "config": "python", "text_column": "content", "is_code": True, "is_go": False, "offset": 500000}, 
        # {"name": "github-code-go", "path": "hasankursun/github-code-2025-language-split", "config": "go", "text_column": "content", "is_code": True, "is_go": True, "offset": 500000}
    ]
    
    tokenizer_types = [
        # "split", 
        #   "split_functionalizer",
        "llama", 
          "llama_functionalizer",
    ]
    num_prompts = args.num_prompts
    
    all_results = {}
    if not args.no_checkpoint:
        loaded = load_checkpoint_results(downstream_json_path)
        if loaded:
            print(f"Loaded existing checkpoint from {downstream_json_path}")
            all_results = loaded
    
    for ds in datasets:
        ds_name = ds["name"]
        if ds_name not in all_results:
            all_results[ds_name] = {}

        needed_seeds = {}
        for tok_type in tokenizer_types:
            if tok_type not in all_results[ds_name]:
                all_results[ds_name][tok_type] = []
            needed_seeds[tok_type] = []
            for seed in active_seeds:
                if args.no_checkpoint:
                    needed_seeds[tok_type].append(seed)
                else:
                    cached_res = find_checkpointed_result(all_results, ds_name, tok_type, seed, args)
                    if cached_res is None:
                        needed_seeds[tok_type].append(seed)
                    else:
                        print(f"SKIPPING: {ds_name} ({tok_type}) | Seed: {seed} (reusing checkpointed result)")

        total_needed = sum(len(seeds) for seeds in needed_seeds.values())
        if total_needed == 0:
            print(f"\n==========================================")
            print(f"ALL TESTS COMPLETED IN CHECKPOINT: {ds_name}")
            print(f"==========================================")
            continue

        prompts, _ = load_validation_prompts(
            ds["path"], ds["config"], "validation", ds["text_column"], num_prompts,
            data_dir=ds.get("data_dir"), is_code=ds["is_code"], is_go=ds["is_go"],
            offset=ds.get("offset", 0)
        )
        
        if not prompts:
            print(f"Failed to load prompts for dataset {ds['name']}. Skipping.")
            continue
            
        for tok_type in tokenizer_types:
            bundle = None
            for seed in active_seeds:
                if seed not in needed_seeds[tok_type]:
                    continue

                checkpoint_dir = checkpoint_dir_for(ds_name, tok_type, seed, args.model_size, args.run_tag)
                if not os.path.exists(checkpoint_dir):
                    print(f"Checkpoint directory {checkpoint_dir} not found. Skipping seed {seed} for {tok_type}.")
                    continue

                if bundle is None:
                    bundle = load_tokenizer_bundle(checkpoint_dir, prompts)

                print(f"\n==========================================")
                print(f"STARTING EVALUATION: {ds_name} ({tok_type}) | Seed: {seed}")
                print(f"==========================================")

                res = run_evaluation(
                    ds_name, tok_type, seed, prompts, ds["is_code"], is_go=ds["is_go"],
                    model_size=args.model_size, run_tag=args.run_tag, max_new_tokens=args.max_new_tokens,
                    collapse_cycle_len=args.collapse_cycle_len,
                    collapse_cycle_repeats=args.collapse_cycle_repeats,
                    repetition_max_n=args.repetition_max_n,
                    tokenizer_bundle=bundle
                )
                if res:
                    existing_list = all_results[ds_name][tok_type]
                    updated = False
                    for idx, item in enumerate(existing_list):
                        if isinstance(item, dict) and item.get("seed") == seed:
                            existing_list[idx] = res
                            updated = True
                            break
                    if not updated:
                        existing_list.append(res)
                    existing_list.sort(key=lambda x: x.get("seed", 0) if isinstance(x, dict) else 0)

                    # Save checkpoint immediately so progress is preserved across restarts
                    save_checkpoint_results(downstream_json_path, all_results)
                    generate_and_save_reports(
                        all_results, args, active_seeds, tokenizer_types,
                        downstream_report_path, downstream_json_path, downstream_samples_path, run_suffix
                    )

    # Final report and checkpoint generation
    generate_and_save_reports(
        all_results, args, active_seeds, tokenizer_types,
        downstream_report_path, downstream_json_path, downstream_samples_path, run_suffix
    )
        
    print(f"\nDownstream evaluation finished!")
    print(f"Go syntax checker used: {'gofmt (' + GOFMT_PATH + ')' if GOFMT_PATH else 'BRACKET FALLBACK -- gofmt not found on PATH!'}")
    print(f"Report saved to {downstream_report_path}")
    print(f"Raw per-seed results saved to {downstream_json_path}")
    print(f"Samples saved to {downstream_samples_path}")

if __name__ == "__main__":
    main()

