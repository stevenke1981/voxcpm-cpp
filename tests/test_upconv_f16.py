"""Reproduce ggml_conv_transpose_1d logic in Python with F16 weights.
Compare F32 vs F16 precision to see if F16 explains the RMS difference."""
import torch
import torch.nn.functional as F
import numpy as np
import gguf
import os
import glob

os.chdir("E:/voxcpm-cpp")
reader = gguf.GGUFReader("voxcpm2_v2_full.gguf")

# Load upconv weight for block 2
w_name = "audio_vae.decoder.model.2.block.1.weight.weight"
for t in reader.tensors:
    if t.name == w_name:
        w_f16 = t.data
        break

# Convert to float32 (exact F16 quantization)
K, OC, IC = 16, 1024, 2048

# In GGUF, shape is [16, 1024, 2048] which is [K, OC, IC] in ggml order
# The raw data is stored as float16 in this shape
w_np = w_f16.ravel().astype(np.float32).reshape(IC, OC, K)  # GGUF stores as ne[2], ne[1], ne[0]
# Actually let me check the actual storage order
print(f"GGUF tensor shape: {w_f16.shape}")
print(f"GGUF tensor data_shape: {w_f16.data.shape}")
# GGUF reader returns shape in ggml order (ne)
# The data is flat, we need to figure out correct reshape
# In memory: ne[0] varies fastest, then ne[1], then ne[2]
# So data is stored as: dim0 varies fastest, then dim1, then dim2
total = w_f16.size
print(f"Total elements: {total}, expected: {K*OC*IC}")

# Data layout: ne=[16, 1024, 2048]
# In ggml row-major: element (k, oc, ic) is at data[k + oc*16 + ic*16*1024]
# In numpy: element (k, oc, ic) at [ic, oc, k]... actually let me verify
w_flat = w_f16.ravel().astype(np.float32)

# For GGUF ne=[16, 1024, 2048] (K, OC, IC), memory layout is:
# index = k + oc*K + ic*K*OC
# So ic varies slowest, then oc, then k
# Numpy reshape to [IC, OC, K] would give:
# w_flat.reshape(IC, OC, K)[ic, oc, k] = w_flat[k + oc*K + ic*K*OC]
# And [K, OC, IC] accessed as w_flat[k + oc*K + ic*K*OC] = w_flat.reshape(IC, OC, K)[ic, oc, k]

w_ggml_3d = w_flat.reshape(IC, OC, K)  # [IC, OC, K] in memory layout
# PyTorch ConvTranspose1d weight: [IC, OC, K]
w_pt = torch.from_numpy(w_ggml_3d)  # [IC, OC, K]
print(f"PyTorch weight: {w_pt.shape}")

# Load the input (C model.1 output) from reference fixture
fixtures = sorted(glob.glob("fixtures/ref/model_1*.npy"))
print(f"Fixtures matching model_1: {fixtures}")
if not fixtures:
    # Try other patterns
    fixtures = sorted(glob.glob("fixtures/ref/*model_1*.npy"))
    print(f"Alternative: {fixtures}")

if fixtures:
    inp = np.load(fixtures[0])
    print(f"Loaded input: {inp.shape}, range=[{inp.min():.4f}, {inp.max():.4f}]")
    # PyTorch expects [B, C, T]
    inp_pt = torch.from_numpy(inp).unsqueeze(0)  # [1, 2048, 8]
    print(f"PyTorch input: {inp_pt.shape}")
    
    with torch.no_grad():
        # F32 reference
        out_f32 = F.conv_transpose1d(inp_pt, w_pt.float(), stride=8, padding=0)
        print(f"\nF32 conv_transpose1d: {out_f32.shape}")
        print(f"  range=[{out_f32.min():.6f}, {out_f32.max():.6f}]")
        print(f"  mean={out_f32.mean():.6f}, rms={out_f32.pow(2).mean().sqrt():.6f}")
        
        out_f32_crop = out_f32[..., :-8]
        print(f"F32 cropped: {out_f32_crop.shape}")
        print(f"  rms={out_f32_crop.pow(2).mean().sqrt():.6f}")
        
        # F16 computation (matches ggml)
        out_f16 = F.conv_transpose1d(inp_pt, w_pt.half(), stride=8, padding=0)
        print(f"\nF16 conv_transpose1d: {out_f16.shape}")
        print(f"  range=[{out_f16.min():.6f}, {out_f16.max():.6f}]")
        print(f"  mean={out_f16.mean():.6f}, rms={out_f16.pow(2).mean().sqrt():.6f}")
        
        out_f16_crop = out_f16[..., :-8]
        print(f"F16 cropped: {out_f16_crop.shape}")
        print(f"  rms={out_f16_crop.pow(2).mean().sqrt():.6f}")
        
        diff = (out_f32_crop - out_f16_crop.float()).abs()
        print(f"\nF32 vs F16 diff:")
        print(f"  max abs: {diff.max():.8f}")
        print(f"  rms diff: {diff.pow(2).mean().sqrt():.8f}")
        print(f"  relative error: {diff.pow(2).mean().sqrt() / out_f32_crop.pow(2).mean().sqrt():.8f}")
else:
    print("No input fixture found!")
    # Create dummy input matching model.1 output shape
    inp = np.random.randn(2048, 8).astype(np.float32)
    inp_pt = torch.from_numpy(inp).unsqueeze(0)
    print(f"Using dummy input: {inp_pt.shape}")
    
    with torch.no_grad():
        out_f32 = F.conv_transpose1d(inp_pt, w_pt.float(), stride=8, padding=0)
        print(f"\nF32 conv_transpose1d (dummy): {out_f32.shape}, rms={out_f32.pow(2).mean().sqrt():.6f}")
