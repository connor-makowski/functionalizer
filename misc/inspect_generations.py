import importlib.metadata
orig_version = importlib.metadata.version
def mock_version(package_name):
    if package_name == "tokenizers":
        return "0.22.2"
    return orig_version(package_name)
importlib.metadata.version = mock_version

import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel
from downstream_tests import load_validation_prompts

def inspect_model(dataset_name, tokenizer_type, seed, prompts, num_inspect=5):
    checkpoint_dir = f"./checkpoints/{dataset_name}_{tokenizer_type}_seed{seed}_25M"
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} does not exist.")
        return
        
    print(f"\n==========================================")
    print(f"INSPECTING: {tokenizer_type} (Seed {seed})")
    print(f"==========================================")
    
    tokenizer = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    
    sep_id = tokenizer.token_to_id("[SEP]")
    
    for idx, prompt in enumerate(prompts[:num_inspect]):
        encoding = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
        
        curr_input = input_ids
        generated_ids = []
        
        for _ in range(50):
            with torch.no_grad():
                outputs = model(curr_input)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            token_val = next_token_id.item()
            generated_ids.append(token_val)
            if token_val == sep_id:
                break
            curr_input = torch.cat([curr_input, next_token_id.unsqueeze(0)], dim=-1)
            
        print(f"\n--- Sample {idx+1} ---")
        print(f"Prompt: {repr(prompt)}")
        print(f"Prompt tokens: {encoding.tokens}")
        print(f"Generated token IDs ({len(generated_ids)}): {generated_ids}")
        
        # Check tokens themselves
        token_strings = [tokenizer.id_to_token(tid) for tid in generated_ids]
        print(f"Generated token strings: {token_strings}")
        
        # Test decoding
        decode_exception = None
        try:
            completion = tokenizer.decode(generated_ids)
        except Exception as e:
            decode_exception = str(e)
            raw_tokens = []
            for tid in generated_ids:
                tok = tokenizer.id_to_token(tid)
                if tok is not None:
                    raw_tokens.append(tok)
            completion = " ".join(raw_tokens)
            
        print(f"Decode Exception: {decode_exception}")
        print(f"Decoded completion: {repr(completion)}")
        print(f"Completion length (chars): {len(completion)}")
        print(f"Chars/token: {len(completion)/len(generated_ids):.4f}")

if __name__ == "__main__":
    prompts, _ = load_validation_prompts(
        "roneneldan/TinyStories", None, "validation", "text", 2, is_code=False
    )
    
    inspect_model("tinystories", "metaspace_split", 100, prompts)
    inspect_model("tinystories", "metaspace_split_functionalizer", 100, prompts)
