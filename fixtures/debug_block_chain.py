"""Compare C decoder block outputs with Python reference.
Reads GGUF weights, builds a single conv transpose + residual block in numpy,
and compares intermediate tensor shapes/RMS with expected behavior.
"""

import gguf
import numpy as np
import sys
import struct
import math

def load_gguf_weight(reader, name_substring):
    """Find tensor by name substring and return (raw_dims, numpy_data)."""
    for t in reader.tensors:
        if name_substring in t.name:
            return np.array(t.shape), t.data
    return None, None

def conv_transpose1d_numpy(weight, input_2d, stride):
    """Simulate ggml_conv_transpose_1d with [K, OC, IC] weight layout.
    
    weight: numpy array shape (IC, OC, K) — this is the reversed GGUF dims layout
    input_2d: numpy array shape (L, IC) — seq_len x in_channels
    
    Returns: output (OW, OC)
    """
    IC, OC, K = weight.shape
    L, IC_in = input_2d.shape
    assert IC == IC_in, f"IC mismatch: {IC} vs {IC_in}"
    
    # Raw output length: (L-1)*stride + K
    OW = (L - 1) * stride + K
    
    # Build output via im2col
    output = np.zeros((OW, OC), dtype=np.float64)
    
    for oc in range(OC):
        for ic in range(IC):
            w_k = weight[ic, oc, :]  # kernel for this (oc, ic) pair
            for l in range(L):
                for k in range(K):
                    out_pos = l * stride + k
                    output[out_pos, oc] += input_2d[l, ic] * w_k[k]
    
    return output

def depthwise_conv1d_numpy(weight, input_2d, stride, pad, dilate):
    """Depthwise conv1d: weight shape (IC, 1, K), input (L, IC)."""
    IC_out, ch_per_group, K = weight.shape
    L, IC_in = input_2d.shape
    assert ch_per_group == 1, f"Expected depthwise, got {ch_per_group}"
    assert IC_out == IC_in or IC_out == 1, f"IC mismatch: {IC_out} vs {IC_in}"
    
    # Effective kernel with dilation
    K_eff = (K - 1) * dilate + 1
    
    # Output length
    OW = (L + 2 * pad - K_eff) // stride + 1
    
    output = np.zeros((OW, IC_out), dtype=np.float64)
    
    for ic in range(IC_out):
        w_k = weight[ic, 0, :]  # kernel for this channel
        for l in range(OW):
            acc = 0.0
            for k in range(K):
                in_pos = l * stride - pad + k * dilate
                if 0 <= in_pos < L:
                    acc += input_2d[in_pos, ic] * w_k[k]
            output[l, ic] = acc
    
    return output

def conv1d_numpy(weight, input_2d, stride, pad, dilate):
    """Regular conv1d: weight shape (OC, IC, K), input (L, IC)."""
    if weight.ndim == 3 and weight.shape[1] == 1 and weight.shape[0] > 1:
        # This might be a depthwise weight — check shape
        OC, _, K = weight.shape
        L, IC_in = input_2d.shape
        if OC == IC_in:
            # depthwise: use depthwise function
            return depthwise_conv1d_numpy(weight, input_2d, stride, pad, dilate)
    
    OC, IC, K = weight.shape
    L, IC_in = input_2d.shape
    assert IC == IC_in, f"IC mismatch: {IC} vs {IC_in}"
    
    K_eff = (K - 1) * dilate + 1
    OW = (L + 2 * pad - K_eff) // stride + 1
    
    output = np.zeros((OW, OC), dtype=np.float64)
    
    for oc in range(OC):
        for ic in range(IC):
            w_k = weight[ic if OC == 1 and IC == IC_in else oc, ...] if weight.ndim == 2 else weight[oc, ic, :]
            
            # Actually just index properly
            if weight.ndim == 2 and weight.shape[1] == K:
                # 2D weight: shape (IC*OC, K)
                w_k = weight[oc * IC + ic, :]
            elif weight.ndim == 3:
                w_k = weight[oc, ic, :]
            
            for l in range(OW):
                acc = 0.0
                for k in range(K):
                    in_pos = l * stride - pad + k * dilate
                    if 0 <= in_pos < L:
                        acc += input_2d[in_pos, ic] * w_k[k]
                output[l, oc] += acc
    
    return output

def snake_activation(x, alpha):
    """Snake activation: x + (1/alpha) * sin(alpha * x)^2."""
    return x + (1.0 / alpha) * np.sin(alpha * x) ** 2

def load_weight_np(reader, name_substring):
    """Load GGUF tensor as numpy array in (..., K, OC, IC) layout? 
    Returns numpy array with shape matching the input data layout.
    """
    for t in reader.tensors:
        if name_substring in t.name:
            # t.data has shape = reversed(raw_dims) in C-order
            raw_dims = np.array(t.shape)
            data = np.array(t.data)  # copy to regular numpy
            # raw_dims: [ne0, ne1, ne2] = [K, OC, IC] for conv_transpose
            # data shape: reversed = (IC, OC, K)
            return raw_dims, data
    return None, None

def compute_conv_transpose_output_shape(L, stride, K):
    """Raw output length before causal trim."""
    return (L - 1) * stride + K

def main():
    reader = gguf.GGUFReader('voxcpm2_v2_full.gguf')
    
    # ---- Config from model ----
    decoder_rates = [8, 6, 5, 2, 2, 2]
    
    # ---- Step 1: Create a test input similar to what C uses ----
    # The input to decoder is processed by model.0 and model.1 first.
    # We'll start with raw latent and trace through each step.
    
    # First, let's create a random latent similar to the C test
    np.random.seed(42)
    latent_dim = 64
    n_patches = 24  # from generate.c
    n_codebooks = 4  # fsq reduction factor
    seq_len = n_patches * n_codebooks  # 96
    
    # Random latent input
    latent = np.random.randn(seq_len, latent_dim).astype(np.float64)
    latent_rms = np.sqrt(np.mean(latent ** 2))
    print(f"Latent input shape: {latent.shape}, RMS: {latent_rms:.4f}")
    
    # ---- Step 2: model.0 (depthwise conv, in=64→out=64, k=7, groups=64) ----
    _, w0 = load_weight_np(reader, 'audio_vae.decoder.model.0.weight.weight')
    _, b0 = load_weight_np(reader, 'audio_vae.decoder.model.0.bias')
    
    if w0 is not None and b0 is not None:
        # weight shape (64, 1, 7) = (channels, groups=1, kernel)
        print(f"\nmodel.0 weight: shape={w0.shape}")
        h = depthwise_conv1d_numpy(w0, latent, stride=1, pad=3, dilate=1)
        h = h + b0[np.newaxis, :]
    else:
        h = latent.copy()
    
    print(f"model.0 output: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
    
    # ---- Step 3: model.1 (pointwise conv, in=64→out=2048, k=1) ----
    _, w1 = load_weight_np(reader, 'audio_vae.decoder.model.1.weight.weight')
    _, b1 = load_weight_np(reader, 'audio_vae.decoder.model.1.bias')
    
    if w1 is not None and b1 is not None:
        # weight shape (2048, 64, 1) for Conv1d(64, 2048, 1)
        print(f"\nmodel.1 weight: shape={w1.shape}")
        h = conv1d_numpy(w1, h, stride=1, pad=0, dilate=1)
        h = h + b1[np.newaxis, :]
    
    print(f"model.1 output: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
    
    # ---- Step 4: Decoder blocks 2-7 ----
    print("\n=== Decoder Blocks ===")
    
    for bi, stride in enumerate(decoder_rates):
        block_idx = bi + 2  # model.2 to model.7
        prev_rms = np.sqrt(np.mean(h**2))
        prev_shape = h.shape
        
        # ---- 4a: Upconv transpose ----
        _, up_w = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.1.weight.weight')
        if up_w is None:
            print(f"Block.{block_idx}: MISSING upconv weight")
            break
        
        # up_w.data shape = (IC, OC, K) 
        # IC = h.shape[1], OC = IC // 2 (halving), K = 2*stride
        print(f"\nBlock.{block_idx} (stride={stride}):")
        print(f"  Input: shape={h.shape}, RMS={prev_rms:.4f}")
        
        IC_w, OC_w, K_w = up_w.shape
        print(f"  Upconv weight: shape={up_w.shape} (IC={IC_w}, OC={OC_w}, K={K_w})")
        print(f"  Expected: in={prev_shape[1]}, out={prev_shape[1]//2}, K={2*stride}")
        
        # Check weight stats
        print(f"  Upconv weight RMS: {np.sqrt(np.mean(up_w**2)):.6f}")
        print(f"  Upconv weight mean abs: {np.mean(np.abs(up_w)):.6f}")
        
        # Conv transpose (no bias yet — add after trim)
        h_up = conv_transpose1d_numpy(up_w, h, stride)
        
        # Causal trim: remove last `stride` steps
        h_up = h_up[:h_up.shape[0] - stride, :]
        
        # Add bias
        _, up_b = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.1.bias')
        if up_b is not None:
            h_up = h_up + up_b[np.newaxis, :]
        
        h = h_up
        print(f"  After upconv: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
        
        # ---- 4b: 3 residual units ----
        for ri in range(3):
            res_idx = 2 + ri
            
            _, ra_w = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.{res_idx}.block.1.weight.weight')
            if ra_w is None:
                print(f"  No residual unit {res_idx}")
                continue
            
            # This is depthwise conv1d: weight shape = (OC=1024/512/..., 1, K=7)
            _, rb_w = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.{res_idx}.block.2.weight.weight')
            
            _, rb_b = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.{res_idx}.block.2.bias')
            
            # block.3.weight.weight = pointwise conv
            _, rc_w = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.{res_idx}.block.3.weight.weight')
            
            _, rc_b = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.{res_idx}.block.3.bias')
            
            # Alpha for snake activation
            ra_alpha_shape, ra_alpha = load_weight_np(reader, f'audio_vae.decoder.model.{block_idx}.block.{res_idx}.block.0.alpha')
            
            if ra_w is None or rb_w is None or rc_w is None:
                print(f"  Residual {res_idx}: missing weights, skipping")
                continue
            
            rms_before = np.sqrt(np.mean(h**2))
            
            # depthwise conv1d
            res_h = depthwise_conv1d_numpy(ra_w, h, stride=1, pad=3, dilate=1)
            
            # snake
            alpha_val = 1.0
            if ra_alpha is not None:
                alpha_val = float(np.mean(ra_alpha))
                res_h = snake_activation(res_h, alpha_val)
            
            # pointwise conv1d
            res_h = conv1d_numpy(rb_w, res_h, stride=1, pad=0, dilate=1)
            
            if rb_b is not None:
                res_h = res_h + rb_b[np.newaxis, :]
            
            # snake
            if ra_alpha is not None:
                res_h = snake_activation(res_h, alpha_val)
            
            # residual add
            # rc_w is pointwise: (OC=out_ch, IC=out_ch, K=1)
            # For the residual unit, the 3rd conv is also pointwise but different channels?
            # Actually res_h shape should be (L, IC) -> after pointwise (L, OC)
            # But for residual, h should be same shape as res_h
            
            res_h3 = conv1d_numpy(rc_w, res_h, stride=1, pad=0, dilate=1)
            if rc_b is not None:
                res_h3 = res_h3 + rc_b[np.newaxis, :]
            
            # Residual add
            h = res_h3 + h
            
            rms_after = np.sqrt(np.mean(h**2))
            print(f"  Residual {res_idx}: RMS {rms_before:.4f} -> {rms_after:.4f} (ratio={rms_after/max(rms_before,1e-10):.4f})")
        
        print(f"  Block.{block_idx} final: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
        
        if h.shape[1] <= 32:
            break
    
    print(f"\n=== After all decoder blocks: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
    
    # ---- Step 5: model.8 (pre-output conv) ----
    _, w8 = load_weight_np(reader, 'audio_vae.decoder.model.8.weight.weight')
    _, b8 = load_weight_np(reader, 'audio_vae.decoder.model.8.bias')
    
    if w8 is not None:
        print(f"\nmodel.8: shape={w8.shape}")
        h = depthwise_conv1d_numpy(w8, h, stride=1, pad=3, dilate=1)
        if b8 is not None:
            h = h + b8[np.newaxis, :]
        print(f"  After: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
    
    # ---- Step 6: model.9 (output conv, k=7, in=32→out=1) ----
    _, w9 = load_weight_np(reader, 'audio_vae.decoder.model.9.weight.weight')
    _, b9 = load_weight_np(reader, 'audio_vae.decoder.model.9.bias')
    
    if w9 is not None:
        print(f"\nmodel.9: shape={w9.shape}")
        print(f"  Weight RMS: {np.sqrt(np.mean(w9**2)):.6f}")
        h = conv1d_numpy(w9, h, stride=1, pad=3, dilate=1)
        if b9 is not None:
            h = h + b9[np.newaxis, :]
        print(f"  After: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f}")
        print(f"  Peak: {np.max(np.abs(h)):.6f}")
    
    print(f"\n=== Final output: shape={h.shape}, RMS={np.sqrt(np.mean(h**2)):.4f} ===")
    
    return h

if __name__ == '__main__':
    result = main()
