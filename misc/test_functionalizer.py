from typing import Union, List
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Metaspace, Functionalizer as FunctionalizerPreTok, Whitespace
from tokenizers.decoders import Functionalizer as FunctionalizerDecoder, Metaspace as MetaspaceDecoder, Sequence as SequenceDecoder

# Custom split pattern matching newlines, tabs, single ASCII spaces, single Metaspaces, and grouped punctuation
DEFAULT_SPLIT_PAT = r"\n+|\t+|[ ]| |[^\p{L}\p{N}\s _]+"

def decode_functionalizer(tokens: list, split_operators: bool = True) -> str:
    """
    Decodes a list of token strings using the FunctionalizerDecoder and MetaspaceDecoder.
    """
    decoder = SequenceDecoder([
        FunctionalizerDecoder(capitalize=True, repeat=True, serialize=True, split_operators=split_operators),
        MetaspaceDecoder(prepend_scheme="never")
    ])
    return decoder.decode(tokens)


def print_functionalizer(text: Union[str, List[str]], split_operators: bool = True, decode: bool = False) -> None:
    """
    Runs pre-tokenization on the input text and prints the formatted tokens,
    optionally decoding them to show lossless verification.
    """
    if isinstance(text, list):
        for item in text:
            print_functionalizer(item, split_operators=split_operators, decode=decode)
        return

    # 1. Metaspace only
    metaspace_pretok = Metaspace(prepend_scheme="never")
    splits_metaspace = metaspace_pretok.pre_tokenize_str(text)
    tokens_metaspace = [val for val, _ in splits_metaspace]

    # 2. Metaspace + Split
    metaspace_split_pretok = Sequence([
        Metaspace(prepend_scheme="never"),
        Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated")
    ])
    splits_metaspace_split = metaspace_split_pretok.pre_tokenize_str(text)
    tokens_metaspace_split = [val for val, _ in splits_metaspace_split]

    # 3. Metaspace + Whitespace
    metaspace_whitespace_pretok = Sequence([
        Metaspace(prepend_scheme="never"),
        Whitespace()
    ])
    splits_metaspace_whitespace = metaspace_whitespace_pretok.pre_tokenize_str(text)
    tokens_metaspace_whitespace = [val for val, _ in splits_metaspace_whitespace]

    # 4. Metaspace + Whitespace + Functionalizer
    metaspace_whitespace_func_pretok = Sequence([
        Metaspace(prepend_scheme="never"),
        Whitespace(),
        FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=split_operators)
    ])
    splits_metaspace_whitespace_func = metaspace_whitespace_func_pretok.pre_tokenize_str(text)
    tokens_metaspace_whitespace_func = [val for val, _ in splits_metaspace_whitespace_func]

    # 5. All three stacked (Metaspace + Split + Functionalizer)
    full_pretok = Sequence([
        Metaspace(prepend_scheme="never"),
        Split(pattern=Regex(DEFAULT_SPLIT_PAT), behavior="isolated"),
        FunctionalizerPreTok(capitalize=True, repeat=True, serialize=True, split_operators=split_operators)
    ])
    splits_full = full_pretok.pre_tokenize_str(text)
    tokens_full = [val for val, _ in splits_full]

    print(f"Input: {repr(text)}")
    print("  - Metaspace:              ", tokens_metaspace)
    print("  - Metaspace + Split:      ", tokens_metaspace_split)
    print("  - Metaspace + Whitespace: ", tokens_metaspace_whitespace)
    print("  - M+W+Func:               ", tokens_metaspace_whitespace_func)
    print("  - M+S+Func:               ", tokens_full)
        
    if decode:
        decoded_text = decode_functionalizer(tokens_full, split_operators=split_operators)
        is_lossless = (decoded_text == text)
        status = "✅ (lossless)" if is_lossless else f"❌ (mismatch: {repr(decoded_text)})"
        print(f"  - Decoded:                          {status}")



print("=== Test Case 1: Basic Casing and Diacritics ===")
print_functionalizer("Hello, world! Café naïve résumé.", decode=False)
print_functionalizer(["    def my_function():\n","        ","return 'Hello, World!'\n"], decode=False)
print_functionalizer(["____Hello, Hello, Hello!"], decode=False)
