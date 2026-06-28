"""Test tokenizer output for "Hello world."."""
import json

# Load tokenizer.json directly
with open(r"D:\voxcpm-cpp\pretrained_models\VoxCPM2\tokenizer.json", "r", encoding="utf-8") as f:
    tok_data = json.load(f)

vocab = tok_data["model"]["vocab"]  # token -> id
merges = tok_data["model"]["merges"]  # list of "left right" strings

def bpe_encode(text):
    """Simple BPE following HuggingFace tokenizers."""
    # 1. Character-level split
    words = [text[i] for i in range(len(text))]
    
    # 2. Apply merges greedily
    def get_rank(pair):
        s = pair[0] + " " + pair[1]
        if s in merges:
            return merges.index(s)
        return len(merges)
    
    while len(words) > 1:
        best_rank = len(merges)
        best_idx = -1
        for i in range(len(words) - 1):
            rank = get_rank((words[i], words[i+1]))
            if rank < best_rank:
                best_rank = rank
                best_idx = i
        if best_idx < 0:
            break
        words[best_idx] = words[best_idx] + words[best_idx + 1]
        del words[best_idx + 1]
    
    # Convert to IDs
    ids = []
    for w in words:
        if w in vocab:
            ids.append(vocab[w])
        else:
            # Byte encoding fallback
            for b in w.encode("utf-8"):
                byte_token = f"<0x{b:02X}>"
                if byte_token in vocab:
                    ids.append(vocab[byte_token])
    return ids

# Test cases
for text in ["Hello world.", "Hello", " world.", "▁Hello world.", "▁Hello", "▁world."]:
    ids = bpe_encode(text)
    print(f"{repr(text):30s} -> {ids}")

print()

# Now simulate the HuggingFace pre-tokenizer
# GPT-2 regex pre-tokenizer: splits on whitespace but keeps it
import re
gpt2_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

text = "Hello world."
print(f"Input: {repr(text)}")
tokens = gpt2_pattern.findall(text)
print(f"GPT-2 pre-tokenize: {tokens}")

# Add prefix space marker
normalized = ["\u2581" + t if i == 0 or t.startswith(" ") else t for i, t in enumerate(tokens)]
print(f"With ▁ prefix: {normalized}")

# BPE encode each
for nt in normalized:
    ids = bpe_encode(nt)
    print(f"  {repr(nt)} -> {ids}")
