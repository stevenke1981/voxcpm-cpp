"""Dump Python attention intermediates for C comparison."""
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent / "fixtures"

def load_bin(name):
    return np.fromfile(HERE / name, dtype=np.float32)

# Parameters
hidden, n_heads, n_kv_heads, head_dim, n_tokens = 16, 4, 2, 4, 3
q_per_kv = n_heads // n_kv_heads
kv_hidden = n_kv_heads * head_dim

# Load inputs
x = load_bin("attention_input.bin").reshape(n_tokens, hidden)
q_w = load_bin("attention_q_weight.bin").reshape(hidden, hidden)
k_w = load_bin("attention_k_weight.bin").reshape(hidden, kv_hidden)
v_w = load_bin("attention_v_weight.bin").reshape(hidden, kv_hidden)
o_w = load_bin("attention_o_weight.bin").reshape(hidden, hidden)

# QKV projections (simple identity, so q ≈ x, k ≈ x[:, :kv_hidden])
q_proj = x @ q_w  # [3, 16]
k_proj = x @ k_w  # [3, 8]
v_proj = x @ v_w  # [3, 8]

print("=== Q projection (first 8 dims) ===")
print(q_proj[:, :8])
print("\n=== K projection ===")
print(k_proj)
print("\n=== V projection ===")
print(v_proj)

# Reshape to multi-head
q_mh = q_proj.reshape(n_tokens, n_heads, head_dim).transpose(1, 0, 2)
k_mh = k_proj.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)
v_mh = v_proj.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)

print(f"\nQ multi-head shape: {q_mh.shape}")  # [4, 3, 4]
print(f"Q[0, :, :]:\n{q_mh[0, :, :]}")  # head 0, all tokens
print(f"Q[1, :, :]:\n{q_mh[1, :, :]}")  # head 1, all tokens

# Per-token attention
out_tokens = []
for ti in range(n_tokens):
    kv_len = ti + 1
    group_outs = []
    for g in range(n_kv_heads):
        q_g = q_mh[g * q_per_kv:(g + 1) * q_per_kv, ti:ti+1, :]  # [q_per_kv, 1, head_dim]
        k_g = k_mh[g, :kv_len, :]    # [kv_len, head_dim]
        v_g = v_mh[g, :kv_len, :]    # [kv_len, head_dim]

        scores = k_g @ q_g.squeeze(1).T / np.sqrt(head_dim)  # [kv_len, q_per_kv]
        scores_max = scores.max(axis=0, keepdims=True)
        scores_exp = np.exp(scores.astype(np.float64) - scores_max)
        scores_sm = (scores_exp / scores_exp.sum(axis=0, keepdims=True)).astype(np.float32)

        out_g = v_g.T @ scores_sm  # [head_dim, q_per_kv]
        group_outs.append(out_g)

    group_out = np.concatenate(group_outs, axis=1).reshape(-1, 1)
    out_t = o_w.T @ group_out
    out_tokens.append(out_t)

out = np.concatenate(out_tokens, axis=1)
print(f"\nFinal output:\n{out}")

# Save as flat binary for C comparison
out.astype(np.float32).tofile(HERE / "attention_debug_output.bin")
print(f"\nSaved to {HERE / 'attention_debug_output.bin'}")
