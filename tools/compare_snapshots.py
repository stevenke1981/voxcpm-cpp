#!/usr/bin/env python3
"""Compare C snapshot intermediates against PyTorch reference for VAE decoder."""

import gguf, os, sys
import numpy as np
import torch
import torch.nn.functional as F

GGUF_PATH = "E:/voxcpm-cpp/voxcpm2_v2_full.gguf"

def load_gguf_tensor(name):
    """Load a GGUF tensor and convert from ggml layout to numpy.

    GGML stores tensors with ne[0] as fastest-changing dimension.
    For a conv weight stored as [K, IC, OC]:
      ggml element (k, ic, oc) at offset = k + ic*K + oc*K*IC
    This matches numpy C-order shape (OC, IC, K):
      arr[oc, ic, k] at offset = oc*IC*K + ic*K + k = k + ic*K + oc*K*IC
    """
    r = gguf.GGUFReader(GGUF_PATH)
    for t in r.tensors:
        if t.name == name:
            d = np.frombuffer(t.data, dtype=np.float32 if t.tensor_type==0 else np.float16)
            if len(t.shape) == 1:
                return d.astype(np.float32)
            elif len(t.shape) == 2:
                # For 2D tensors (e.g., some scale/bias), ggml ne=[C, 1]
                # numpy (1, C) or (C, 1)? We return flat for simplicity
                return d.reshape(t.shape).astype(np.float32)
            elif len(t.shape) == 3:
                K, IC, OC = t.shape
                return d.reshape((int(OC), int(IC), int(K))).astype(np.float32)
            return d.reshape(t.shape).astype(np.float32)
    return None

def load_snapshot(idx):
    path = f"E:/voxcpm-cpp/snap_{idx:02d}.bin"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        N = np.frombuffer(f.read(8), dtype=np.int64)[0]
        C = np.frombuffer(f.read(8), dtype=np.int64)[0]
        return np.frombuffer(f.read(), dtype=np.float32).reshape((int(N), int(C)))

def rms(x):
    return float(np.sqrt(np.mean(x**2)))

def compare(c_np, py_np, name):
    if c_np.shape != py_np.shape:
        print(f"  {name}: SHAPE MISMATCH C={c_np.shape} Py={py_np.shape}")
        return False
    diff = c_np - py_np
    drms = rms(diff)
    py_rms = rms(py_np)
    rel = drms / max(py_rms, 1e-10)
    print(f"  {name}: C rms={rms(c_np):.6f} Py rms={py_rms:.6f} diff rms={drms:.10f} rel={rel:.8f}")
    if rel > 0.01:
        print(f"    *** LARGE DISCREPANCY *** max_abs_diff={np.abs(diff).max():.8f}")
        return False
    return True

def do_conv1d(x, w_np, bias=None, pad=0, stride=1, dilate=1, groups=1):
    """
    Perform conv1d matching C's conv1d_f32.
    x: [L, C] in C format (L=time, C=channels)
    w_np: numpy array shape [OC, IC, K] (PyTorch format, from load_gguf_tensor)
    Returns: [L_out, OC] in C format
    """
    OC, IC, K = w_np.shape
    w = torch.from_numpy(w_np).float()
    
    # Input: [L, IC] → PyTorch [1, IC, L]
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float()
    
    # Conv1d
    out = F.conv1d(x_t, w, bias=None, stride=stride, padding=pad, dilation=dilate, groups=groups)
    
    if bias is not None:
        b_t = torch.from_numpy(bias).float()
        out = out + b_t.view(1, -1, 1)
    
    # Back to [L_out, OC]
    return out.squeeze(0).permute(1, 0).detach().numpy()

def manual_conv1d(x_np, w_np, bias_np=None, stride=1):
    """
    Direct nested-loop implementation of conv1d_f32 from C.
    x_np: [N, IC] numpy
    w_np: [OC, IC, K] numpy (PyTorch/load_gguf_tensor format)
    
    The C conv1d_f32 does:
    1. im2col: extracts sliding windows [K*IC, OW]
    2. matmul: im2col^T * w_2d → [OW, OC]
    
    Let's replicate directly.
    """
    OC, IC, K = w_np.shape
    N = x_np.shape[0]
    OW = (N - K) // stride + 1  # without padding
    
    # Direct conv: for each output position and channel
    out = np.zeros((OW, OC), dtype=np.float32)
    for j in range(OW):
        for oc in range(OC):
            s = 0.0
            for ic in range(IC):
                for k in range(K):
                    s += x_np[j*stride + k, ic] * w_np[oc, ic, k]
            out[j, oc] = s
            if bias_np is not None:
                out[j, oc] += bias_np[oc]
    return out

def do_causal_conv1d(x, weight_gguf, bias=None, pad=0, stride=1, dilate=1, groups=1):
    """
    Conv1d with CAUSAL (left-only) padding matching C's depthwise/conv1d_f32.
    C code: left_pad = pad * 2 (for depthwise) or pad (for regular conv1d_f32 via im2col).
    The ggml im2col replicates the input pad * dilate times on the left.
    """
    # Actually, let me look at conv1d_f32 more carefully.
    # conv1d_f32 calls ggml_im2col which pads left side only.
    # For depthwise, left_pad = pad * 2 = 6 (for pad=3).
    # For regular conv, the im2col pads with `pad` zeros on the left.
    K, OC, IC = weight_gguf.shape
    w = torch.from_numpy(weight_gguf).permute(1, 2, 0).contiguous().float()  # [OC, IC, K]
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float()  # [1, IC, L]
    
    # Apply causal (left-only) padding
    if pad > 0:
        left_pad = pad * 2 if groups > 1 else pad  # depthwise uses pad*2, regular uses pad
        x_t = F.pad(x_t, (left_pad, 0), "constant", 0)
    
    out = F.conv1d(x_t, w, bias=None, stride=stride, padding=0, dilation=dilate, groups=groups)
    if bias is not None:
        b_t = torch.from_numpy(bias).float()
        out = out + b_t.view(1, -1, 1)
    return out.squeeze(0).permute(1, 0).detach().numpy()

def do_depthwise_causal_conv1d(x, w_np, bias=None, pad=0, stride=1, dilate=1):
    """Depthwise conv1d with CAUSAL (left-only) padding matching C's depthwise_conv1d.
    
    w_np: shape [OC, 1, K] (PyTorch format, from load_gguf_tensor)
    """
    C, one, K = w_np.shape
    assert one == 1
    w = torch.from_numpy(w_np).float()  # already [C, 1, K]
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float()  # [1, C, L]
    
    # Causal left-only padding: left_pad = pad * 2 (from C code)
    left_pad = pad * 2
    if left_pad > 0:
        x_t = F.pad(x_t, (left_pad, 0), "constant", 0)
    
    out = F.conv1d(x_t, w, bias=None, stride=stride, padding=0, dilation=dilate, groups=C)
    if bias is not None:
        b_t = torch.from_numpy(bias).float()
        out = out + b_t.view(1, -1, 1)
    return out.squeeze(0).permute(1, 0).detach().numpy()

def do_transpose_conv1d(x, w_np, bias=None, stride=1):
    """Transpose conv1d.
    
    w_np: weight loaded from load_gguf_tensor.
    For ggml conv_transpose, weight stored as [K, OC, IC].
    load_gguf_tensor returns shape (IC, OC, K) = PyTorch F.conv_transpose1d format.
    """
    # Already in [IC, OC, K] format from load_gguf_tensor
    w = torch.from_numpy(w_np).float()
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float()  # [1, IC, L]
    out = F.conv_transpose1d(x_t, w, bias=None, stride=stride, padding=0)
    # Crop causal: remove last stride samples
    if stride > 1:
        out = out[..., :-stride]
    if bias is not None:
        b_t = torch.from_numpy(bias).float()
        out = out + b_t.view(1, -1, 1)
    return out.squeeze(0).permute(1, 0).detach().numpy()

def snake(x, alpha):
    """Snake activation: snake(x, a) = x + sin²(a*x) / a"""
    a = torch.from_numpy(alpha).float()
    x_t = torch.from_numpy(x).permute(1, 0).unsqueeze(0).float()  # [1, C, L]
    ax = a.view(1, -1, 1) * x_t
    out = x_t + torch.sin(ax)**2 / a.view(1, -1, 1)
    return out.squeeze(0).permute(1, 0).detach().numpy()

def load_c_test_input():
    """Load test_input.bin directly from C (guaranteed identical)."""
    path = "E:/voxcpm-cpp/test_input.bin"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found! Run test_vae_only.exe first.")
        return None
    with open(path, "rb") as f:
        N = np.frombuffer(f.read(8), dtype=np.int64)[0]
        C = np.frombuffer(f.read(8), dtype=np.int64)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
    # C stores in ggml order: data[channel * N + patch] for ne=[N, C]
    # Reshape to [N, C] where columns are channels
    arr = np.zeros((int(N), int(C)), dtype=np.float32)
    for p in range(int(N)):
        for d in range(int(C)):
            arr[p, d] = data[d * int(N) + p]
    return arr

def main():
    print("=== Load C snapshots ===")
    snaps = []
    for i in range(11):
        s = load_snapshot(i)
        if s is not None:
            snaps.append(s)
            print(f"  snap[{i}]: shape={s.shape} rms={rms(s):.6f}")
    
    # Load test input from C
    inp = load_c_test_input()
    if inp is None:
        return
    print(f"\n=== Test input from C: shape={inp.shape} rms={rms(inp):.6f}")
    print(f"  Frame 0 channels 0..7: {inp[0][0:8].tolist()}")
    print(f"  Frame 0 channels 8..15: {inp[0][8:16].tolist()}")
    
    print("\n=== Load weights ===")
    
    # ---- Model 0: first conv ----
    w0 = load_gguf_tensor("audio_vae.decoder.model.0.weight.weight")
    b0 = load_gguf_tensor("audio_vae.decoder.model.0.bias")
    print(f"  model.0 weight: shape={w0.shape}")
    
    # Determine if depthwise: ne[1] == 1 && ne[2] > 1
    is_depthwise = (w0.shape[1] == 1)
    print(f"  depthwise={is_depthwise}")
    
    # ---- Weight verification ----
    print("\n--- Weight verification ---")
    # Load C-dumped weights (ggml layout: raw bytes as stored)
    if os.path.exists("E:/voxcpm-cpp/model0_weight.bin"):
        with open("E:/voxcpm-cpp/model0_weight.bin", "rb") as f:
            raw_f16 = np.frombuffer(f.read(), dtype=np.float16)
        # Convert ggml layout to numpy: raw has K*IC*OC elements in ggml order
        # ggml ne=[K=7, IC=1, OC=64]: data[k + 0*7 + ch*7*1] = data[k + ch*7]
        # numpy (OC, IC, K): arr[ch, 0, k] = data[ch*7 + k] (since ne0=K=7 changes fastest)
        K, IC, OC = 7, 1, 64
        w0_from_c = raw_f16.reshape((OC, IC, K)).astype(np.float32)  # [64, 1, 7]
        print(f"  C weight (converted): shape={w0_from_c.shape}")
        print(f"  C weight channel 0 kernel: {w0_from_c[0, 0, :].tolist()}")
        print(f"  Python weight channel 0 kernel: {w0[0, 0, :].tolist()}")
        w0_diff = w0_from_c - w0
        print(f"  Weight diff max_abs: {np.abs(w0_diff).max():.8f}")
        print(f"  Weight diff RMS: {np.sqrt(np.mean(w0_diff**2)):.10f}")
    
    if os.path.exists("E:/voxcpm-cpp/model0_bias.bin"):
        with open("E:/voxcpm-cpp/model0_bias.bin", "rb") as f:
            b0_raw = np.frombuffer(f.read(), dtype=np.float32).reshape((64,))
        print(f"  C bias first 5: {b0_raw[:5].tolist()}")
        print(f"  Python bias first 5: {b0[:5].tolist()}")
        b0_diff = b0_raw - b0
        print(f"  Bias diff max_abs: {np.abs(b0_diff).max():.10f}")
    
    if is_depthwise:
        print(f"\n--- Model 0 (depthwise causal conv1d) ---")
        # w0 shape: [64, 1, 7] = [OC, IC, K], already PyTorch format
        out0 = do_depthwise_causal_conv1d(inp, w0, b0, pad=3)
        ok = compare(snaps[0], out0, "model.0 (PyTorch)")
        
        # Try manual conv with diagonal expanded weight
        OC_dw, IC_dw, K_dw = w0.shape  # [64, 1, 7]
        C_dw = OC_dw
        # Pad input manually
        inp_pad = np.zeros((inp.shape[0] + 6, inp.shape[1]), dtype=np.float32)
        inp_pad[6:, :] = inp  # left-pad by 6
        
        # Diagonal weight for standard conv: [K, C, C]
        w_diag = np.zeros((K_dw, C_dw, C_dw), dtype=np.float32)
        for ch in range(C_dw):
            for kp in range(K_dw):
                w_diag[kp, ch, ch] = w0[ch, 0, kp]  # w0[oc=ch, ic=0, k=kp]
        
        # Manual conv: uses [OC, IC, K] format
        out0_manual = manual_conv1d(inp_pad, w0, b0)
        compare(snaps[0], out0_manual, "model.0 (manual)")
        
        # Also try F.conv1d with diagonal weight (groups=1)
        w_diag_t = torch.from_numpy(w_diag).permute(1, 2, 0).contiguous().float()  # [C, C, K]
        x_t = torch.from_numpy(inp_pad).permute(1, 0).unsqueeze(0).float()  # [1, C, 14]
        out_t = F.conv1d(x_t, w_diag_t, stride=1, padding=0, groups=1)
        if b0 is not None:
            out_t = out_t + torch.from_numpy(b0).float().view(1, -1, 1)
        out0_t = out_t.squeeze(0).permute(1, 0).detach().numpy()
        compare(snaps[0], out0_t, "model.0 (diag F.conv1d)")
    else:
        print(f"\n--- Model 0 (regular conv1d) ---")
        out0 = do_causal_conv1d(inp, w0, b0, pad=3)
        compare(snaps[0], out0, "model.0")
    
    # ---- Model 1: pointwise conv1d ----
    w1 = load_gguf_tensor("audio_vae.decoder.model.1.weight.weight")
    b1 = load_gguf_tensor("audio_vae.decoder.model.1.bias")
    print(f"\n--- Model 1 ---")
    print(f"  weight: {w1.shape}")  # [2048, 64, 1] = [OC, IC, K]
    
    OC1, IC1, K1 = w1.shape
    print(f"  OC={OC1}, IC={IC1}, K={K1} (snap[0] channels={snaps[0].shape[1]})")
    
    if snaps[0].shape[1] != IC1:
        print(f"  IC mismatch!")
    else:
        w1_t = torch.from_numpy(w1).float()  # already [OC, IC, K]
        x1_t = torch.from_numpy(snaps[0]).permute(1, 0).unsqueeze(0).float()
        out1 = F.conv1d(x1_t, w1_t, stride=1, padding=0)
        if b1 is not None:
            out1 = out1 + torch.from_numpy(b1).float().view(1, -1, 1)
        out1_np = out1.squeeze(0).permute(1, 0).detach().numpy()
        compare(snaps[1], out1_np, "model.1")
    
    # ---- Block 2: sr_cond + snake + upconv ----
    w_up = load_gguf_tensor("audio_vae.decoder.model.2.block.1.weight.weight")
    b_up = load_gguf_tensor("audio_vae.decoder.model.2.block.1.bias")
    a2 = load_gguf_tensor("audio_vae.decoder.model.2.block.0.alpha")
    sr_scale = load_gguf_tensor("audio_vae.decoder.sr_cond_model.2.scale_embed.weight")
    sr_bias = load_gguf_tensor("audio_vae.decoder.sr_cond_model.2.bias_embed.weight")
    
    print(f"\n--- Block 2 Upconv ---")
    print(f"  upconv weight: {w_up.shape}")  # [IC=2048, OC=1024, K=16] from load_gguf_tensor
    
    if sr_scale is not None:
        print(f"  sr_scale shape: {sr_scale.shape}")
        # sr_embed weight: GGUF ne=[1, 256, 2048]. load_gguf_tensor returns (2048, 256, 1)
        # Actually, sr_embed is [K=1, IC=256, OC=2048] in ggml, so load_gguf returns (2048, 256, 1)
        # The embedding for sr_idx=3 is at sr_embed[..., 3] if last dim is 256
        if sr_scale.shape[0] == 2048 and sr_scale.shape[1] == 256:
            # [2048, 256, 1] → scale_vec = sr_scale[:, :, 0][:, 3]
            # Actually sr_scale is [2048, 256, 1]. After squeeze: [2048, 256]
            s = sr_scale[:, :, 0]  # [2048, 256]
            b = sr_bias[:, :, 0]   # [2048, 256]
            scale_vec = s[:, 3]   # [2048]
            bias_vec = b[:, 3]    # [2048]
            print(f"  sr_cond scale: mean={scale_vec.mean():.6f}")
            print(f"  sr_cond bias: mean={bias_vec.mean():.6f}")
            s = scale_vec.reshape(1, -1)
            b = bias_vec.reshape(1, -1)
            h_in = snaps[1].copy() * s + b
            print(f"  After sr_cond: rms={rms(h_in):.6f}")
        else:
            print(f"  Unexpected sr_scale shape: {sr_scale.shape}, skipping sr_cond")
            h_in = snaps[1].copy()
    else:
        h_in = snaps[1].copy()
    
    # Snake
    if a2 is not None:
        h_in = snake(h_in, a2)
        print(f"  After snake: rms={rms(h_in):.6f}")
    
    # Upconv
    IC_up, OC_up, K_up = w_up.shape  # [IC, OC, K] format for conv_transpose1d
    print(f"  Upconv: IC={IC_up}, OC={OC_up}, K={K_up}")
    if h_in.shape[1] != IC_up:
        print(f"  IC mismatch: h_in has {h_in.shape[1]} channels, weight IC={IC_up}")
    else:
        out_up = do_transpose_conv1d(h_in, w_up, b_up, stride=8)
        compare(snaps[2], out_up, "block.2 upconv")
    
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
