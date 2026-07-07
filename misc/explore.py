import re
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Metaspace, Whitespace

# Custom split pattern matching newlines, tabs, single ASCII spaces, single Metaspaces, and grouped punctuation (excluding underscores)
DEFAULT_SPLIT_PATH = r"\n+|\t+|[ ]| |[^\p{L}\p{N}\s _]+"

def analyze_splits(raw_text, splits):
    notes = []
    
    # 1. Check if newlines/tabs were discarded
    raw_has_nl_tab = "\n" in raw_text or "\t" in raw_text
    splits_have_nl_tab = any("\n" in val or "\t" in val for val, _ in splits)
    if raw_has_nl_tab:
        if not splits_have_nl_tab:
            notes.append("❌ Newlines/tabs discarded")
        else:
            notes.append("✅ Newlines/tabs preserved")
            
    # 2. Check if punctuation runs (like ==) were grouped or isolated
    punc_runs = re.findall(r'[^\w\s]{2,}', raw_text)
    if punc_runs:
        split_vals = [val for val, _ in splits]
        if any(run in split_vals for run in punc_runs):
            notes.append("✨ Punctuation runs (e.g. '==') grouped")
        else:
            notes.append("⚠️ Punctuation runs split into individual chars")
            
    # 3. Check if spaces were preserved (lossless)
    raw_has_spaces = " " in raw_text
    splits_have_spaces = any(" " in val or " " in val for val, _ in splits)
    if raw_has_spaces:
        if splits_have_spaces:
            notes.append("✅ Spaces preserved (lossless)")
        else:
            notes.append("❌ Spaces discarded")
            
    return notes

def main():
    # Test Inputs
    inputs = [
        # Example 1: Standard text with punctuation and operators
        "Hello, world! 123 + 456 == 579.",
        
        # Example 2: Python code snippet with tabs and leading spaces
        'def add_one(x):\n\t# Add 1 to x\n\treturn x + 1',
        
        # Example 3: Triple-quoted multiline Python code block
        '''"""
def compute_all(data):
    if not data:
        return None
    return [x * 2 for x in data if x != 0]
"""''',
        # Example 4: Words with underscores (identifiers)
        "hi_there, this_is_a_variable_name.",
    ]

    # Pre-tokenizer configurations to test
    configs = [
        (
            "0. Custom Split Only (No Metaspace)",
            Split(pattern=Regex(DEFAULT_SPLIT_PATH), behavior="isolated")
        ),
        (
            "1. Metaspace Only",
            Metaspace(prepend_scheme="never")
        ),
        (
            "2. Metaspace + Whitespace",
            Sequence([
                Metaspace(prepend_scheme="never"),
                Whitespace()
            ])
        ),
        (
            "3. Metaspace + Custom Split (Punctuation-grouping, keeps \\n and \\t)",
            Sequence([
                Metaspace(prepend_scheme="never"),
                Split(pattern=Regex(DEFAULT_SPLIT_PATH), behavior="isolated")
            ])
        ),
        (
            "4. Whitespace Only (No Metaspace)",
            Whitespace()
        ),
    ]

    for idx, text in enumerate(inputs, 1):
        print("=" * 85)
        print(f" INPUT EXAMPLE {idx}")
        print("=" * 85)
        print("Raw multiline view:")
        print(text)
        print("-" * 85)
        print(f"Raw string representation: {repr(text)}")
        print("=" * 85)
        print("Pre-tokenization Results comparing each configuration:")
        print("-" * 85)
        
        for title, pre_tok in configs:
            tokenizer = Tokenizer(BPE())
            tokenizer.pre_tokenizer = pre_tok
            
            try:
                splits = tokenizer.pre_tokenizer.pre_tokenize_str(text)
                formatted_splits = [f"({repr(val)}, {span})" for val, span in splits]
                notes = analyze_splits(text, splits)
                
                print(f"➤ {title}:")
                print("  Tokens: " + ", ".join(formatted_splits))
                if notes:
                    print("  Status: " + " | ".join(notes))
            except Exception as e:
                print(f"➤ {title}:\n  Error: {e}")
            print("-" * 85)
        print("\n\n")

if __name__ == "__main__":
    main()
