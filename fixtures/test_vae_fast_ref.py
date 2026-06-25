#!/usr/bin/env python3
"""Quick VAE decoder sanity check using numpy with im2col for speed."""
import gguf
import numpy as np
import sys
import time

def conv1d_im2col(weight, bias, input_data, pad=3, causal=False):
    """Conv1d via im2col + matmul. weight[K, IC, OC], input[N, IC], output[N_out, OC]."""
    K, IC, OC = weight.shape
    N = input_data.shape[0]
    if causal:
        padded = np.pad(input_data, ((pad, 0), (0, 0)), mode='constant')
    else:
        padded = np.pad(input_data, ((pad, pad), (0, 0)), mode='constant')
    N_out = padded.shape[0] - K + 1
    
    # im2col: [K*IC, N_out]
    cols = np.zeros((K * IC, N_out), dtype=np.float32)
    for i in range(N_out):
        cols[:, i] = padded[i:i+K, :].ravel()
    
    # matmul: [OC, K*IC] @ [K*IC, N_out] = [OC, N_out] → [N_out, OC]
    out = (weight.reshape(K*IC, OC).T @ cols).T
    if bias is not None:
        out += bias.reshape(1, OC)
    return out

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else r"E:\voxcpm-cpp\voxcpm2_v2_full.gguf"
    reader = gguf.GGUFReader(gguf_path)
    
    def get(name):
        for t in reader.tensors:
            if t.name == name:
                d = t.data
                if d.dtype == np.float16:
                    d = d.astype(np.float32)
                return d
        return None
    
    print("Loading model.9 output conv weights...")
    w9 = get("audio_vae.decoder.model.9.weight.weight")  # [OC=1, IC=32, K=7]
    b9 = get("audio_vae.decoder.model.9.bias")
    
    # Convert from stored [OC, IC, K] to ggml-conv1d [K, IC, OC]
    w9_ggml = np.transpose(w9, (2, 1, 0))  # → [K=7, IC=32, OC=1]
    print(f"weight ggml shape: {w9_ggml.shape}")
    print(f"  stats: min={w9_ggml.min():.6f}, max={w9_ggml.max():.6f}, rms={w9_ggml.std():.6f}")
    print(f"  weight sum: {w9_ggml.sum():.6f}")
    print(f"  mean: {w9_ggml.mean():.6f}")
    
    # Test with small input first
    n_test = 100
    c = 32
    
    # Input with RMS=0.043 (matching C block.7 output)
    np.random.seed(42)
    h = np.random.randn(n_test, c).astype(np.float32) * 0.043
    
    t0 = time.time()
    out1 = conv1d_im2col(w9_ggml, b9, h, pad=3)
    t1 = time.time()
    print(f"\nSmall test (n={n_test}): {t1-t0:.4f}s")
    print(f"  range=[{out1.min():.6f}, {out1.max():.6f}], rms={out1.std():.6f}")
    print(f"  peak={np.max(np.abs(out1)):.6f}")
    
    # Full size test (15360 samples)
    n_full = 15360
    h_full = np.random.randn(n_full, c).astype(np.float32) * 0.043
    
    t0 = time.time()
    out_full = conv1d_im2col(w9_ggml, b9, h_full, pad=3)
    t1 = time.time()
    print(f"\nFull test (n={n_full}): {t1-t0:.4f}s")
    print(f"  range=[{out_full.min():.6f}, {out_full.max():.6f}], rms={out_full.std():.6f}")
    print(f"  peak={np.max(np.abs(out_full)):.6f}")
    
    # After tanh
    out_tanh = np.tanh(out_full)
    print(f"  After tanh: peak={np.max(np.abs(out_tanh)):.6f}")
    
    # What RMS input would give peak ~0.3?
    # With weight rms ~0.001 and 224 terms:
    # output_rms = input_rms * weight_rms * sqrt(K*IC)
    expected_output_rms = np.abs(out_full).mean() / 0.043  # output_rms per unit input_rms
    needed_rms = 0.3 / expected_output_rms
    print(f"\nExpected output RMS per unit input RMS: {expected_output_rms:.6f}")
    print(f"Needed input RMS for output peak ~0.3: {needed_rms:.1f}")
    
    # Check if maybe the layernorm or scaling is done differently
    # Let's also check model.8 alpha to see if snake is active
    a8 = get("audio_vae.decoder.model.8.alpha")
    if a8 is not None:
        print(f"\nmodel.8 alpha: shape={a8.shape}, stats: min={a8.min():.4f}, max={a8.max():.4f}, mean={a8.mean():.4f}")

if __name__ == '__main__':
    main()
