#!/usr/bin/env python3
"""
Python reference: Load VAE weights from GGUF, run decoder with test latents,
compare to C output.
"""
import gguf
import numpy as np
import sys
import os

def load_gguf_tensor(reader, name):
    """Load a tensor from GGUF, return as numpy float32 array in ggml layout."""
    for t in reader.tensors:
        if t.name == name:
            data = t.data
            if data.dtype == np.float16:
                data = data.astype(np.float32)
            return data
    return None

def ggml_conv1d_np(weight, bias, input_data, stride=1, pad=3, dilate=1):
    """
    Simulate ggml_conv_1d: conv1d with weight in ggml layout [K, IC, OC].
    Input: [N, IC] (time, channels)
    Weight: [K, IC, OC] (kernel, in_ch, out_ch)
    Output: [N_out, OC]
    
    Uses im2col + matmul (same as ggml_conv_1d).
    """
    K, IC, OC = weight.shape
    N = input_data.shape[0]
    
    # Causal padding
    left_pad = pad
    padded = np.pad(input_data, ((left_pad, 0), (0, 0)), mode='constant')
    
    # im2col: extract sliding windows
    N_out = (padded.shape[0] - K) // stride + 1
    if N_out <= 0:
        return np.zeros((0, OC))
    
    cols = np.zeros((K * IC, N_out))
    for i in range(N_out):
        col = padded[i * stride : i * stride + K, :].ravel()  # K*IC
        cols[:, i] = col
    
    # matmul: weight reshaped to [OC, K*IC]
    w_mat = weight.reshape(K * IC, OC).T  # [OC, K*IC]
    out = w_mat @ cols  # [OC, N_out]
    out = out.T  # [N_out, OC]
    
    if bias is not None:
        out += bias.reshape(1, OC)
    
    return out

def depthwise_conv1d_np(weight, bias, input_data, stride=1, pad=3, dilate=1):
    """
    Depthwise conv1d with diagonal weight expansion.
    Weight: [K, 1, OC] (kernel=7, IC_per_group=1, OC=groups)
    Input: [N, OC] (time, channels)
    """
    K, IC_g, OC = weight.shape
    assert IC_g == 1, f"Expected IC=1 per group for depthwise, got {IC_g}"
    N = input_data.shape[0]
    C = OC  # groups = OC
    
    # Causal padding for depthwise
    left_pad = pad * 2  
    padded = np.pad(input_data, ((left_pad, 0), (0, 0)), mode='constant')
    
    # Expand weight to diagonal [K, C, C]
    w_exp = np.zeros((K, C, C), dtype=np.float32)
    for ch in range(C):
        for kp in range(K):
            w_exp[kp, ch, ch] = weight[kp, 0, ch]
    
    # conv1d with expanded weight and pad=0 (already padded)
    return ggml_conv1d_np(w_exp, bias, padded, stride, 0, dilate)

def snake_activation_np(x, alpha):
    """snake(x, a) = x + sin(a*x)^2 / a"""
    if alpha is None:
        return x
    a = alpha.reshape(1, -1)
    return x + np.sin(a * x)**2 / (a + 1e-8)

def conv1d_layer_np(weight, bias, input_data, stride=1, pad=3, dilate=1):
    """Dispatch to regular or depthwise conv1d."""
    if weight.shape[1] == 1 and weight.shape[2] > 1:
        # Depthwise
        return depthwise_conv1d_np(weight, bias, input_data, stride, pad, dilate)
    else:
        return ggml_conv1d_np(weight, bias, input_data, stride, pad, dilate)

def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else r"E:\voxcpm-cpp\voxcpm2_v2_full.gguf"
    
    reader = gguf.GGUFReader(gguf_path)
    weights = {}
    for t in reader.tensors:
        name = t.name
        data = t.data
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        # Store in ggml layout (no transpose)
        # The GGUF stores data in the same layout as ggml expects
        weights[name] = data
    
    # Create test latent [8, 64] (same as test_vae_only)
    latent_dim = 64
    n_patches = 8
    latent = np.zeros((n_patches, latent_dim), dtype=np.float32)
    for p in range(n_patches):
        for d in range(latent_dim):
            if d == 0:
                latent[p, d] = 0.5
            elif d == 1:
                latent[p, d] = 0.2 * p / n_patches
            elif d < 16:
                latent[p, d] = 0.1 * np.sin(d * n_patches + p)
            else:
                latent[p, d] = 0.0
    
    print(f"Test latent [{n_patches}, {latent_dim}]:")
    print(f"  Range: [{latent.min():.4f}, {latent.max():.4f}]")
    print(f"  Mean: {latent.mean():.6f}, RMS: {latent.std():.6f}")
    
    # === Run decoder ===
    h = latent
    
    # model.0: Conv1d (k=7, 1->64) depthwise
    w0 = weights.get("audio_vae.decoder.model.0.weight.weight")
    b0 = weights.get("audio_vae.decoder.model.0.bias")
    if w0 is not None:
        print(f"\nmodel.0 weight: shape={w0.shape}")
        w0_ggml = np.transpose(w0, (2, 1, 0))  # [OC, IC, K] -> [K, IC, OC]
        h = conv1d_layer_np(w0_ggml, b0, h, stride=1, pad=3, dilate=1)
        print(f"  Output: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    
    # model.1: Pointwise Conv1d (k=1, 64->2048)
    w1 = weights.get("audio_vae.decoder.model.1.weight.weight")
    b1 = weights.get("audio_vae.decoder.model.1.bias")
    if w1 is not None:
        print(f"\nmodel.1 weight: shape={w1.shape}")
        w1_ggml = np.transpose(w1, (2, 1, 0))  # [OC, IC, K] -> [K, IC, OC]
        h = conv1d_layer_np(w1_ggml, b1, h, stride=1, pad=0, dilate=1)
        print(f"  Output: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    
    # model.2-7: Decoder blocks with sr_cond
    rates = [8, 6, 5, 2, 2, 2]
    
    for bi in range(2, 8):
        idx = bi - 2
        rate = rates[idx]
        
        # sr_cond FiLM
        sname = f"audio_vae.decoder.sr_cond_model.{bi}.scale_embed.weight"
        bname = f"audio_vae.decoder.sr_cond_model.{bi}.bias_embed.weight"
        
        scale_emb = weights.get(sname)
        bias_emb = weights.get(bname)
        
        if scale_emb is not None and bias_emb is not None:
            sr_cond_idx = 3  # 48000 Hz
            # scale_emb shape [num_buckets, C] in ggml layout
            # Extract: dst[j] = src[idx + j * num_buckets]
            C = scale_emb.shape[1]
            scale_vec = np.array([scale_emb[sr_cond_idx, j] for j in range(C)], dtype=np.float32)
            bias_vec = np.array([bias_emb[sr_cond_idx, j] for j in range(C)], dtype=np.float32)
            
            h = h * scale_vec.reshape(1, -1) + bias_vec.reshape(1, -1)
        
        # Decoder block
        # Upconv
        up_name = f"audio_vae.decoder.model.{bi}.block.1.weight.weight"
        up_b_name = f"audio_vae.decoder.model.{bi}.block.1.bias"
        up_w = weights.get(up_name)
        up_b = weights.get(up_b_name)
        
        if up_w is not None:
            # upconv_weight shape in ggml: [K, IC, OC] where K=2*rate
            # Python: ConvTranspose1d(IC, OC, K=2*rate, stride=rate)
            # ggml_conv_transpose_1d expects weight in [K, IC, OC] layout
            up_w_ggml = np.transpose(up_w, (2, 0, 1))  # [OC, IC, K] -> [K, IC, OC]
            
            # Simple conv_transpose 1D numpy
            K_up = up_w_ggml.shape[0]
            stride_up = rate
            # Transposed conv: N_out = N_in * stride
            N_in = h.shape[0]
            N_out = N_in * stride_up
            C_in = h.shape[1]
            C_out = up_w_ggml.shape[2]
            
            out = np.zeros((N_out, C_out), dtype=np.float32)
            for t in range(N_in):
                for k in range(K_up):
                    out_idx = t * stride_up + k
                    if out_idx < N_out:
                        for oc in range(C_out):
                            val = 0.0
                            for ic in range(C_in):
                                val += up_w_ggml[k, ic, oc] * h[t, ic]
                            out[out_idx, oc] += val
            
            h = out
            if up_b is not None:
                h += up_b.reshape(1, -1)
            
            print(f"  Block {bi} after upconv: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
        
        # 3 residual units with dilations 1, 3, 9
        for ru_idx in range(3):
            dilation = [1, 3, 9][ru_idx]
            
            # alpha0
            a0_name = f"audio_vae.decoder.model.{bi}.block.{2 + ru_idx}.block.0.alpha"
            a0 = weights.get(a0_name)
            
            # conv1 (depthwise)
            c1_name = f"audio_vae.decoder.model.{bi}.block.{2 + ru_idx}.block.1.weight.weight"
            c1_b_name = f"audio_vae.decoder.model.{bi}.block.{2 + ru_idx}.block.1.bias"
            c1_w = weights.get(c1_name)
            c1_b = weights.get(c1_b_name)
            
            # alpha2
            a2_name = f"audio_vae.decoder.model.{bi}.block.{2 + ru_idx}.block.2.alpha"
            a2 = weights.get(a2_name)
            
            # conv2 (pointwise)
            c2_name = f"audio_vae.decoder.model.{bi}.block.{2 + ru_idx}.block.3.weight.weight"
            c2_b_name = f"audio_vae.decoder.model.{bi}.block.{2 + ru_idx}.block.3.bias"
            c2_w = weights.get(c2_name)
            c2_b = weights.get(c2_b_name)
            
            residual = h.copy()
            
            if a0 is not None:
                a0_np = a0.ravel()
                h = snake_activation_np(h, a0_np)
            
            if c1_w is not None:
                c1_ggml = np.transpose(c1_w, (2, 1, 0))
                h = conv1d_layer_np(c1_ggml, c1_b, h, stride=1, pad=3*dilation, dilate=dilation)
            
            if a2 is not None:
                a2_np = a2.ravel()
                h = snake_activation_np(h, a2_np)
            
            if c2_w is not None:
                c2_ggml = np.transpose(c2_w, (2, 1, 0))
                h = conv1d_layer_np(c2_ggml, c2_b, h, stride=1, pad=0, dilate=1)
            
            if h is not None:
                h = residual + h
        
        print(f"  Block {bi} output: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    
    # model.8: Snake activation
    a8_name = "audio_vae.decoder.model.8.alpha"
    a8 = weights.get(a8_name)
    if a8 is not None:
        h = snake_activation_np(h, a8.ravel())
        print(f"\nmodel.8 snake: range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    
    # model.9: Output Conv1d (k=7, 32->1)
    w9 = weights.get("audio_vae.decoder.model.9.weight.weight")
    b9 = weights.get("audio_vae.decoder.model.9.bias")
    if w9 is not None:
        print(f"\nmodel.9 weight: shape={w9.shape}")
        w9_ggml = np.transpose(w9, (2, 1, 0))  # [OC, IC, K] -> [K, IC, OC]
        h = conv1d_layer_np(w9_ggml, b9, h, stride=1, pad=3, dilate=1)
        print(f"  Output: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    
    # model.10: Tanh
    h = np.tanh(h)
    print(f"\nFinal (tanh): shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}], rms={h.std():.6f}")
    print(f"  Peak: {np.max(np.abs(h)):.6f}")
    print(f"  Samples: {h.shape[0]}")

if __name__ == '__main__':
    main()
