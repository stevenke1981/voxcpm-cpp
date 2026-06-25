#!/usr/bin/env python3
"""Quick VAE decoder sanity check using numpy.
Simplified: only tests model.9 output conv with pre-computed block.7 output.
"""
import gguf
import numpy as np
import sys

def ggml_conv1d_simple(weight, bias, input_data, pad=3):
    """Simple conv1d: weight [K, IC, OC], input [N, IC], output [N_out, OC]."""
    K, IC, OC = weight.shape
    N = input_data.shape[0]
    
    # Causal pad
    padded = np.pad(input_data, ((pad, 0), (0, 0)), mode='constant')
    
    # Direct convolution (not im2col)
    N_out = padded.shape[0] - K + 1
    out = np.zeros((N_out, OC), dtype=np.float32)
    for t in range(N_out):
        for oc in range(OC):
            total = 0.0
            for k in range(K):
                for ic in range(IC):
                    total += weight[k, ic, oc] * padded[t + k, ic]
            out[t, oc] = total
    
    if bias is not None:
        out += bias.reshape(1, OC)
    return out

def depthwise_conv1d_simple(weight, input_data, pad=3):
    """Depthwise conv1d: weight [K, 1, OC], input [N, OC], output [N_out, OC]."""
    K, _, OC = weight.shape
    N = input_data.shape[0]
    left_pad = pad * 2
    padded = np.pad(input_data, ((left_pad, 0), (0, 0)), mode='constant')
    
    N_out = padded.shape[0] - K + 1
    out = np.zeros((N_out, OC), dtype=np.float32)
    for t in range(N_out):
        for oc in range(OC):
            total = 0.0
            for k in range(K):
                total += weight[k, 0, oc] * padded[t + k, oc]
            out[t, oc] = total
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
    
    # Load model.9 weight and bias
    w9 = get("audio_vae.decoder.model.9.weight.weight")  # [OC=1, IC=32, K=7] in stored layout
    b9 = get("audio_vae.decoder.model.9.bias")
    
    print(f"model.9 weight stored shape: {w9.shape}")
    # Convert from stored [OC, IC, K] to ggml [K, IC, OC]
    w9_ggml = np.transpose(w9, (2, 1, 0))  # [K, IC, OC]
    print(f"model.9 weight ggml shape: {w9_ggml.shape}")
    print(f"  Weight stats: min={w9_ggml.min():.6f}, max={w9_ggml.max():.6f}, mean={w9_ggml.mean():.6f}, rms={w9_ggml.std():.6f}")
    
    # Create synthetic block.7 output [15360, 32]
    # Based on C output: RMS ≈ 0.043, range ≈ [-0.025, 0.074]
    np.random.seed(42)
    n = 15360
    c = 32
    
    # Option 1: Random normal scaled to match expected RMS
    h = np.random.randn(n, c).astype(np.float32) * 0.043
    print(f"\nInput to model.9: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    
    # Run model.9 conv
    out1 = ggml_conv1d_simple(w9_ggml, b9, h, pad=3)
    print(f"Output (random input): shape={out1.shape}, range=[{out1.min():.6f}, {out1.max():.6f}], rms={out1.std():.6f}")
    print(f"  Peak: {np.max(np.abs(out1)):.6f}")
    
    # Option 2: All ones
    h2 = np.ones((n, c), dtype=np.float32) * 0.043
    out2 = ggml_conv1d_simple(w9_ggml, b9, h2, pad=3)
    print(f"\nOutput (all 0.043): range=[{out2.min():.6f}, {out2.max():.6f}], rms={out2.std():.6f}")
    print(f"  First 10: {out2[:10, 0].tolist()}")
    
    # Option 3: All 1.0 (to see max possible output)
    h3 = np.ones((n, c), dtype=np.float32)
    out3 = ggml_conv1d_simple(w9_ggml, b9, h3, pad=3)
    print(f"\nOutput (all 1.0): range=[{out3.min():.6f}, {out3.max():.6f}], rms={out3.std():.6f}")
    print(f"  First 10: {out3[:10, 0].tolist()}")
    
    # Option 4: Use C output intermediate from block.7
    # Let's see what the conv weight sum is
    print(f"\nWeight sum per output channel:")
    for oc in range(1):
        total = np.sum(w9_ggml[:, :, oc])
        print(f"  OC {oc}: sum={total:.6f}")
    
    # What values would we NEED as input to model.9 to get peak ~0.3 output?
    # If output RMS is ~0.3 (normal speech), and weight RMS is 0.001:
    # input_rms = output_rms / (sqrt(K*IC) * weight_rms) = 0.3 / (15 * 0.001) = 20
    needed_rms = 0.3 / (np.sqrt(7*32) * w9_ggml.std())
    print(f"\nNeeded input RMS for output peak ~0.3: {needed_rms:.1f}")
    
    h4 = np.random.randn(n, c).astype(np.float32) * needed_rms
    out4 = ggml_conv1d_simple(w9_ggml, b9, h4, pad=3)
    out4_tanh = np.tanh(out4)
    print(f"Output (scaled input, rms={needed_rms:.1f}): peak={np.max(np.abs(out4_tanh)):.4f}")
    
    # Check each block's output - load block.7 intermediate
    # Actually, let's just check if depthwise conv works for model.0
    w0 = get("audio_vae.decoder.model.0.weight.weight")
    if w0 is not None:
        print(f"\nmodel.0 weight stored shape: {w0.shape}")
        w0_ggml = np.transpose(w0, (2, 1, 0))
        print(f"model.0 weight ggml [K, IC, OC]: {w0_ggml.shape}")
        print(f"  Weight stats: min={w0_ggml.min():.4f}, max={w0_ggml.max():.4f}, mean={w0_ggml.mean():.4f}, rms={w0_ggml.std():.4f}")

if __name__ == '__main__':
    main()
