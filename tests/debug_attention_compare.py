"""Compare C attention output against golden, per-token."""
import numpy as np
from pathlib import Path

HERE = Path("tests/fixtures")

# Parameters
hidden, n_heads, n_kv_heads, head_dim, n_tokens = 16, 4, 2, 4, 3

# Load fixture golden
golden = np.fromfile(HERE / "attention_output.bin", dtype=np.float32).reshape(hidden, n_tokens)

# Load C output (from debug_attention_dump, but using tensor_from_2d)
# Actually let me recompute Python reference
x = np.fromfile(HERE / "attention_input.bin", dtype=np.float32).reshape(n_tokens, hidden)
q_w = np.fromfile(HERE / "attention_q_weight.bin", dtype=np.float32).reshape(hidden, hidden)
k_w = np.fromfile(HERE / "attention_k_weight.bin", dtype=np.float32).reshape(hidden, n_kv_heads * head_dim)
v_w = np.fromfile(HERE / "attention_v_weight.bin", dtype=np.float32).reshape(hidden, n_kv_heads * head_dim)
o_w = np.fromfile(HERE / "attention_o_weight.bin", dtype=np.float32).reshape(hidden, hidden)

q_per_kv = n_heads // n_kv_heads
kv_hidden = n_kv_heads * head_dim

# QKV projections
q = x @ q_w  # [3, 16]
k = x @ k_w  # [3, 8]
v = x @ v_w  # [3, 8]

# Reshape to multi-head
q_mh = q.reshape(n_tokens, n_heads, head_dim).transpose(1, 0, 2)
k_mh = k.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)
v_mh = v.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)

# Compute per-token
out_tokens = []
for ti in range(n_tokens):
    kv_len = ti + 1
    group_outs = []
    for g in range(n_kv_heads):
        q_g = q_mh[g * q_per_kv:(g + 1) * q_per_kv, ti:ti+1, :]
        k_g = k_mh[g, :kv_len, :]
        v_g = v_mh[g, :kv_len, :]

        scores = k_g @ q_g.squeeze(1).T / np.sqrt(head_dim)
        scores_max = scores.max(axis=0, keepdims=True)
        scores_exp = np.exp(scores.astype(np.float64) - scores_max)
        scores_sm = (scores_exp / scores_exp.sum(axis=0, keepdims=True)).astype(np.float32)

        out_g = v_g.T @ scores_sm
        group_outs.append(out_g)

    group_out = np.concatenate(group_outs, axis=1).reshape(-1, 1)
    out_t = o_w.T @ group_out
    out_tokens.append(out_t)

out_python = np.concatenate(out_tokens, axis=1)

print("=== Python (recomputed) vs Golden ===")
for t in range(n_tokens):
    diff = out_python[:, t] - golden[:, t]
    print(f"Token {t}: max_diff={np.max(np.abs(diff)):.6f}, "
          f"cos={np.dot(out_python[:,t], golden[:,t]) / (np.linalg.norm(out_python[:,t]) * np.linalg.norm(golden[:,t]) + 1e-30):.6f}")

# Now let's try to figure out what C is computing
# The C code uses vcpm_attention with no_rope=1, no_causal=0
# Let me trace through manually

# First, Q/K/V projections:
# In C: q = ggml_mul_mat(q_w, x) = q_w^T @ x
# q_w is [16,16], x is [16,3]
# q_w stored as eye in row-major, loaded as column-major [16,16]
# After fix with tensor_from_2d_rowmajor: q_w[i,j] = eye[i,j]
# q_w^T = eye too. q = x. 
# q in C [16,3], flat: x[t*16 + d]
# q in Python [3,16], flat: x[t*16 + d] (since q = x @ eye = x)

print("\n=== Q projections ===")
print("Python q[:, :8]:")
print(q[:, :8])
print("\nPython q flat first 24:", q.flatten()[:24])

# In C, q after mul_mat: shape [16,3], ne0=16, ne1=3
# Column-major: q_c[d, t] at offset d + t*16 = q_data_flat[d + t*16]
# Python row-major: q_p[t, d] = q[t, d]
# q_p[t, d] should = q_c[d, t]

print("\nC-style flat layout of q (token-major):", end=" ")
for t in range(n_tokens):
    for d in range(min(8, hidden)):
        print(f"{q[t,d]:.0f}", end=" ")
print()

# Now reshape to 3D in C: q_reshaped = reshape_3d(q, head_dim=4, n_heads=4, n_tokens=3)
# C column-major: q_reshaped[d, h, t] = q[h*4+d, t] = x[t, h*4+d]
print("\n=== Q reshaped [d, h, t] first 8 elements ===")
for d in range(4):
    for h in range(2):
        for t in range(n_tokens):
            val = q[t, h*4 + d]
            print(f"  q_reshaped[d={d},h={h},t={t}] = {val:.0f}", end=" | ")
        print()

# Now check what Python uses
print("\n=== Python Q groups ===")
for g in range(2):
    q_g_py = q_mh[g*q_per_kv:(g+1)*q_per_kv, :, :]  # [q_per_kv, n_tokens, head_dim]
    print(f"\nGroup {g} Python Q heads (shape {q_g_py.shape}):")
    print(f"  Head {g*2}: {q_g_py[0, :, :]}")
    print(f"  Head {g*2+1}: {q_g_py[1, :, :]}")

# Now K/V cache and per-token processing
# For ti=1, group 0:
ti = 1
g = 0
kv_len = ti + 1

print(f"\n=== Token {ti}, Group {g} ===")
# Python Q for this group and token
q_g_py = q_mh[g*q_per_kv:(g+1)*q_per_kv, ti:ti+1, :]  # [q_per_kv, 1, head_dim] -> [2, 1, 4]
q_g_py_2d = q_g_py.squeeze(1)  # [2, 4]
print(f"Python q_g shape {q_g_py.shape} -> {q_g_py_2d.shape}")
print(f"Python q_g:\n{q_g_py_2d}")

# Python K for this group (causal: kv_len = ti+1)
k_g_py = k_mh[g, :kv_len, :]  # [kv_len, head_dim]
print(f"Python k_g shape {k_g_py.shape}")
print(f"Python k_g:\n{k_g_py}")

# Python V
v_g_py = v_mh[g, :kv_len, :]
print(f"Python v_g shape {v_g_py.shape}")
print(f"Python v_g:\n{v_g_py}")

# Scores
scores_py = k_g_py @ q_g_py_2d.T / np.sqrt(head_dim)
print(f"\nScores:\n{scores_py}")
scores_max = scores_py.max(axis=0, keepdims=True)
scores_exp = np.exp(scores_py.astype(np.float64) - scores_max)
scores_sm = (scores_exp / scores_exp.sum(axis=0, keepdims=True)).astype(np.float32)
print(f"Softmax:\n{scores_sm}")

out_g_py = v_g_py.T @ scores_sm
print(f"Out group shape: {out_g_py.shape}")
print(f"Out group:\n{out_g_py}")

# Check if fixture's k_w is stored correctly
print(f"\n=== k_w matrix check ===")
print(f"k_w shape: {k_w.shape}")
print(f"k_w[:8, :4]:\n{k_w[:8, :4]}")
print(f"k_w row 0: {k_w[0, :]}")
print(f"k_w row 4: {k_w[4, :]}")
print(f"k_w row 8: {k_w[8, :]}")
