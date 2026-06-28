"""Test tokenizer: compare C tokenizer output vs HuggingFace expected."""
import json
import re
import sys
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

with open(r"D:\voxcpm-cpp\pretrained_models\VoxCPM2\tokenizer.json", "r", encoding="utf-8") as f:
    tok_data = json.load(f)

vocab = tok_data["model"]["vocab"]
merges = tok_data["model"]["merges"]
MERGES_SET = {m: i for i, m in enumerate(merges)}

def bpe_encode(text):
    """BPE encode a single word/text."""
    words = list(text)
    while len(words) > 1:
        best_rank = len(merges)
        best_idx = -1
        for i in range(len(words) - 1):
            s = words[i] + " " + words[i+1]
            rank = MERGES_SET.get(s, len(merges))
            if rank < best_rank:
                best_rank = rank
                best_idx = i
        if best_idx < 0:
            break
        words[best_idx] = words[best_idx] + words[best_idx + 1]
        del words[best_idx + 1]
    ids = []
    for w in words:
        if w in vocab:
            ids.append(vocab[w])
        else:
            for b in w.encode("utf-8"):
                bt = f"<0x{b:02X}>"
                if bt in vocab:
                    ids.append(vocab[bt])
    return ids

print("=== C tokenizer path (normalize_voxcpm_text + split_initial_symbols) ===")
text = "Hello world."
normalized = "\u2581" + text  # what normalize_voxcpm_text does
print(f"normalized: {repr(normalized)}")

# C's split_initial_symbols: split into UTF-8 characters
# C's split_initial_symbols: split into UTF-8 characters
bdata = normalized.encode("utf-8")
symbols = []
p = 0
while p < len(bdata):
    c = bdata[p]
    if (c & 0x80) == 0:
        clen = 1
    elif (c & 0xE0) == 0xC0:
        clen = 2
    elif (c & 0xF0) == 0xE0:
        clen = 3
    else:
        clen = 4
    symbols.append(bdata[p:p+clen].decode("utf-8"))
    p += clen
print(f"split: {symbols}")

# BPE merge
words = list(symbols)
while len(words) > 1:
    best_rank = len(merges)
    best_idx = -1
    for i in range(len(words) - 1):
        s = words[i] + " " + words[i+1]
        rank = MERGES_SET.get(s, len(merges))
        if rank < best_rank:
            best_rank = rank
            best_idx = i
    if best_idx < 0:
        break
    words[best_idx] = words[best_idx] + words[best_idx + 1]
    del words[best_idx + 1]
print(f"BPE result: {words}")

ids = []
for w in words:
    if w in vocab:
        ids.append(vocab[w])
print(f"IDs: {ids}")

print()
print("=== GPT-2 pre-tokenizer (LlamaTokenizerFast) ===")
# The GPT-2/LLaMA regex pattern (without \p{L} which isn't in Python re)
# Split on: contractions, optional-space+letters, optional-space+digits, 
# optional-space+punctuation, trailing whitespace
gpt2_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z\x80-\xff]+| ?[0-9]+| ?[^\sA-Za-z0-9\x80-\xff]+|\s+(?!\S)|\s+""")
tokens = gpt2_pattern.findall(text)
print(f"Pre-tokenized: {tokens}")
normalized = ["\u2581" + t for t in tokens]
print(f"With U+2581: {normalized}")

all_ids = []
for nt in normalized:
    ids = bpe_encode(nt)
    print(f"  {repr(nt):30s} -> {ids}")
    all_ids.extend(ids)
print(f"Total IDs: {all_ids}")

print()
# Try HuggingFace
try:
    from transformers import AutoTokenizer
    import os
    os.chdir(r"D:\voxcpm-cpp")
    tok_hf = AutoTokenizer.from_pretrained(r"D:\voxcpm-cpp\pretrained_models\VoxCPM2", trust_remote_code=True)
    hf_ids = tok_hf.encode("Hello world.")
    print(f"HF tokenizer encode: {hf_ids}")
    for i, tid in enumerate(hf_ids):
        tok_str = tok_hf.convert_ids_to_tokens(tid)
        print(f"  [{i}] = {tid} {repr(tok_str)}")
except ImportError:
    print("transformers not available")
except Exception as e:
    print(f"HF error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== Try Llama HF tokenizer if available ===")
try:
    from transformers import AutoTokenizer
    tok_hf = AutoTokenizer.from_pretrained(r"D:\voxcpm-cpp\pretrained_models\VoxCPM2", trust_remote_code=True)
    hf_ids = tok_hf.encode("Hello world.")
    print(f"HF tokenizer encode: {hf_ids}")
    print(f"HF tokenizer decode: {tok_hf.decode(hf_ids)}")
except ImportError:
    print("transformers not available")
except Exception as e:
    print(f"Error: {e}")
