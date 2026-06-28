#!/usr/bin/env python3
"""Generate golden reference data for C CUDA op isolation tests.

Each test case produces:
  <name>_input.bin   — synthetic input (float32 array)
  <name>_output.bin  — expected output from Python (float32 array)
  <name>_meta.txt    — shape/dim metadata

The C test loads these .bin files and compares CUDA output against them.

Usage:
  cd tests/fixtures && python gen_op_fixtures.py
"""

import numpy as np
from pathlib import Path
import struct

HERE = Path(__file__).parent


def write_bin(path: Path, arr: np.ndarray):
    """Write float32 array as raw binary."""
    path.write_bytes(arr.astype(np.float32).tobytes())
    print(f"  wrote {path.name}  {arr.shape}  [{arr.nbytes / 1024:.1f} KB]")


def write_meta(path: Path, text: str):
    path.write_text(text.strip() + "\n")


# ============================================================
# 1. RMS Norm (no weight) — plain ggml_rms_norm
# ============================================================
def gen_rms_norm():
    print("\n--- RMS Norm (no weight) ---")
    hidden, n_tokens = 8, 4
    rs = np.random.RandomState(42)
    x = np.zeros((n_tokens, hidden), dtype=np.float32)
    for t in range(n_tokens):
        for h in range(hidden):
            x[t, h] = float((t + 1) * 10 + h + 1)

    # RMS norm
    eps = 1e-6
    x_f64 = x.astype(np.float64)
    variance = (x_f64 ** 2).mean(axis=-1, keepdims=True)
    y = (x_f64 / np.sqrt(variance + eps)).astype(np.float32)

    prefix = "rms_norm_plain"
    write_bin(HERE / f"{prefix}_input.bin", x.reshape(-1))
    write_bin(HERE / f"{prefix}_output.bin", y.reshape(-1))
    write_meta(HERE / f"{prefix}_meta.txt",
               f"hidden_size={hidden}\nn_tokens={n_tokens}\neps={eps}")


# ============================================================
# 2. RMS Norm (with weight) — vcpm_rms_norm
# ============================================================
def gen_rms_norm_weighted():
    print("\n--- RMS Norm (with weight) ---")
    hidden, n_tokens = 8, 4
    rs = np.random.RandomState(42)
    x = np.zeros((n_tokens, hidden), dtype=np.float32)
    for t in range(n_tokens):
        for h in range(hidden):
            x[t, h] = float((t + 1) * 10 + h + 1)

    weight = np.array([1.0 + 0.1 * i for i in range(hidden)], dtype=np.float32)

    # RMS norm + weight
    eps = 1e-6
    x_f64 = x.astype(np.float64)
    w_f64 = weight.astype(np.float64)
    variance = (x_f64 ** 2).mean(axis=-1, keepdims=True)
    y = (x_f64 / np.sqrt(variance + eps)).astype(np.float32)
    y = y * weight  # F32 multiply

    prefix = "rms_norm_weighted"
    write_bin(HERE / f"{prefix}_input.bin", x.reshape(-1))
    write_bin(HERE / f"{prefix}_weight.bin", weight)
    write_bin(HERE / f"{prefix}_output.bin", y.reshape(-1))
    write_meta(HERE / f"{prefix}_meta.txt",
               f"hidden_size={hidden}\nn_tokens={n_tokens}\neps={eps}")


# ============================================================
# 3. MLP (SwiGLU)
# ============================================================
def gen_mlp():
    print("\n--- MLP (SwiGLU) ---")
    hidden, inter, n_tokens = 8, 16, 4

    x = np.zeros((n_tokens, hidden), dtype=np.float32)
    for t in range(n_tokens):
        for h in range(hidden):
            x[t, h] = float((t + 1) * 10 + h + 1)

    # Identity-like weights (diagonal 1.0)
    gate_w = np.zeros((hidden, inter), dtype=np.float32)
    up_w   = np.zeros((hidden, inter), dtype=np.float32)
    down_w = np.zeros((inter, hidden), dtype=np.float32)
    for i in range(min(hidden, inter)):
        gate_w[i, i] = 1.0
        up_w[i, i] = 1.0
    for i in range(min(inter, hidden)):
        down_w[i, i] = 1.0

    # SwiGLU: output = down_w^T @ (silu(x @ gate_w) * (x @ up_w))
    # In PyTorch Linear: out = x @ W^T
    # So gate = x @ gate_w.T  (since gate_w is weight matrix for Linear)
    # But we use raw matmul convention matching C:
    # C: gate = ggml_mul_mat(gate_w, x) = gate_w^T @ x  where gate_w is [hidden, inter]
    #   => output shape [inter, n_tokens], each col = x^T * gate_w_col = gate_w_col^T @ x
    # In numpy: gate = gate_w.T @ x.T  (but we want [n_tokens, inter])
    # Let's use: gate = x @ gate_w  gives [n_tokens, inter]
    # Or: gate = (gate_w.T @ x.T).T = x @ gate_w

    gate = x @ gate_w   # [n_tokens, inter]
    up   = x @ up_w     # [n_tokens, inter]

    # SiLU = sigmoid(x) * x
    def silu(z):
        return 1.0 / (1.0 + np.exp(-z.astype(np.float64))) * z

    gate_silu = silu(gate).astype(np.float32)
    product = gate_silu * up
    out = product @ down_w  # [n_tokens, hidden]

    prefix = "mlp"
    write_bin(HERE / f"{prefix}_input.bin", x.reshape(-1))
    write_bin(HERE / f"{prefix}_gate_weight.bin", gate_w.T.reshape(-1))
    write_bin(HERE / f"{prefix}_up_weight.bin", up_w.T.reshape(-1))
    write_bin(HERE / f"{prefix}_down_weight.bin", down_w.T.reshape(-1))
    write_bin(HERE / f"{prefix}_output.bin", out.reshape(-1))
    write_meta(HERE / f"{prefix}_meta.txt",
               f"hidden_size={hidden}\nintermediate_size={inter}\nn_tokens={n_tokens}\n"
               f"activation=SiLU")


# ============================================================
# 4. Attention (synthetic, no RoPE)
# ============================================================
def gen_attention():
    print("\n--- Attention (no RoPE) ---")
    hidden, n_heads, n_kv_heads, head_dim = 16, 4, 2, 4
    n_tokens = 3
    q_per_kv = n_heads // n_kv_heads

    x = np.zeros((n_tokens, hidden), dtype=np.float32)
    for t in range(n_tokens):
        for h in range(hidden):
            x[t, h] = float((t + 1) * 10 + h + 1)

    # QKV weights (identity-like)
    q_w = np.eye(hidden, hidden, dtype=np.float32)
    k_w = np.eye(hidden, n_kv_heads * head_dim, dtype=np.float32)
    v_w = np.eye(hidden, n_kv_heads * head_dim, dtype=np.float32) * 0.5
    o_w = np.eye(hidden, hidden, dtype=np.float32)

    # Q projection: q = x @ q_w  [n_tokens, hidden]
    q = x @ q_w

    # K/V projections: k = x @ k_w  [n_tokens, kv_hidden]
    kv_hidden = n_kv_heads * head_dim
    k = x @ k_w
    v = x @ v_w

    # Reshape to multi-head format
    # Q: [n_tokens, n_heads, head_dim]
    q = q.reshape(n_tokens, n_heads, head_dim).transpose(1, 0, 2)  # [n_heads, n_tokens, head_dim]
    # K: [n_tokens, n_kv_heads, head_dim]
    k = k.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)  # [n_kv_heads, n_tokens, head_dim]
    # V: [n_tokens, n_kv_heads, head_dim]
    v = v.reshape(n_tokens, n_kv_heads, head_dim).transpose(1, 0, 2)  # [n_kv_heads, n_tokens, head_dim]

    # Manual GQA attention (matching C implementation, no RoPE, causal mask)
    # Per-token: causal self-attention
    out_tokens = []
    for ti in range(n_tokens):
        # k_len = ti + 1 (causal)
        kv_len = ti + 1
        # Build head-major output matching C implementation:
        # gqa_group_attn returns [head_dim * q_per_kv, 1] = head0_dims, head1_dims
        # groups are concatenated along dim 0 -> [head_dim * n_heads, 1]
        # Result: head0 all dims, head1 all dims, ..., headN all dims
        group_out_list = []
        for h in range(n_heads):
            g = h // q_per_kv  # which KV head group
            h_q = h % q_per_kv  # which Q head within group

            # This Q head: [head_dim]
            q_h = q[h, ti, :]

            # Shared KV for this group: [kv_len, head_dim]
            k_g = k[g, :kv_len, :]
            v_g = v[g, :kv_len, :]

            # scores: [kv_len]
            scores = k_g @ q_h / np.sqrt(head_dim)

            # softmax
            scores_max = scores.max()
            scores_exp = np.exp(scores.astype(np.float64) - scores_max)
            sm = scores_exp / scores_exp.sum()

            # output: [head_dim]
            out_h = v_g.T @ sm
            group_out_list.append(out_h.reshape(-1, 1))

        # Concat all heads along dim 0: [head_dim * n_heads, 1]
        group_out = np.concatenate(group_out_list, axis=0)
        # O projection
        out_t = o_w.T @ group_out  # [hidden, 1]
        out_tokens.append(out_t)

    # Concat tokens: [hidden, n_tokens]
    out = np.concatenate(out_tokens, axis=1)

    # Write output transposed to match ggml column-major layout
    # ggml stores [hidden, n_tokens] as: token0_all, token1_all, token2_all, ...
    # np.tobytes() on [16,3] gives row-by-row = interleaved across tokens.
    # .T gives [n_tokens, hidden], then flatten gives token-major order.
    out_ggml_order = out.T.reshape(-1)

    prefix = "attention"
    write_bin(HERE / f"{prefix}_input.bin", x.reshape(-1))
    write_bin(HERE / f"{prefix}_q_weight.bin", q_w.reshape(-1))
    write_bin(HERE / f"{prefix}_k_weight.bin", k_w.reshape(-1))
    write_bin(HERE / f"{prefix}_v_weight.bin", v_w.reshape(-1))
    write_bin(HERE / f"{prefix}_o_weight.bin", o_w.reshape(-1))
    write_bin(HERE / f"{prefix}_output.bin", out_ggml_order)
    write_meta(HERE / f"{prefix}_meta.txt",
               f"hidden_size={hidden}\nn_heads={n_heads}\nn_kv_heads={n_kv_heads}\n"
               f"head_dim={head_dim}\nn_tokens={n_tokens}")


# ============================================================
# Main
# ============================================================
def main():
    print("=== Generating operation fixtures ===")
    print(f"Output dir: {HERE}")
    gen_rms_norm()
    gen_rms_norm_weighted()
    gen_mlp()
    gen_attention()
    print("\nDone.")


if __name__ == "__main__":
    main()
