"""Verify model.9 conv1d: compare C output vs manual Python computation."""
import numpy as np
import struct, os

# 1. Load C debug tensors
dbg08 = np.fromfile(r'E:\voxcpm-cpp\build\Release\c_dbg_08.bin', dtype=np.float32)
dbg09 = np.fromfile(r'E:\voxcpm-cpp\build\Release\c_dbg_09.bin', dtype=np.float32)

# Debug tensor 08 = model.8 snake output = model.9 input
# Shape: [61440, 32] (ne[0]=61440=time, ne[1]=32=channels)
# In ggml column-major: data[t + 61440*ch] for (t, ch)
# So for channel ch, the data is dbg08[ch*61440 : (ch+1)*61440]
# Let's verify by checking RMS per channel
N = 61440
IC = 32
OC = 1
K = 7

# Check input has 61440*32 = 1966080 elements
print(f'dbg08 (model.9 input): {len(dbg08)} floats, expected {N*IC}')
assert len(dbg08) == N * IC

# Reshape to [N, IC] where each row has IC channels at time t
# ggml column-major: ne[0]=N, ne[1]=IC
# element (t, ch) = data[ch * N + t]
input_ggml = np.column_stack([dbg08[ch*N:(ch+1)*N] for ch in range(IC)])
print(f'Input reshaped: {input_ggml.shape}')
print(f'Input RMS: {np.sqrt(np.mean(input_ggml**2)):.6f}')

# 2. Load model.9 weight from Python
# check_model9.py already verified: weight has shape [7,32,1] in gguf = (1,32,7) in numpy
# Let's load directly from the GGUF file
# Use the check_model9 module
import sys
sys.path.insert(0, r'E:\voxcpm-cpp\tools')

# Simpler: load the weight from the dumped format
# Let's read the weight struct fields from the GGUF directly
# We know the weight from check_model9.py output
# Weight data (224 F16 values):
# In ggml: [K, IC, OC] = [7, 32, 1]
#   data[ic*K + k + 7*32*oc] = weight at (k, ic, oc)
# For oc=0: data[k + 7*ic] for kernel tap k, input channel ic
#
# PyTorch weight shape: (OC, IC, K) = (1, 32, 7)
#   weight_py[0, ic, k] = data[k + 7*ic] 
# So for conv1d, for output channel 0:
#   out[t] = sum_{ic=0..31} sum_{k=0..6} weight_py[0, ic, k] * input[t + pad - k, ic]
#   where pad=3 (same padding)

# Let me check dbg08 stats more carefully - compute per-channel RMS
rms_per_ch = [np.sqrt(np.mean(dbg08[ch*N:(ch+1)*N]**2)) for ch in range(IC)]
print(f'Input per-channel RMS (first 5): {rms_per_ch[:5]}')
print(f'Input per-channel RMS (last 5): {rms_per_ch[-5:]}')
print(f'Input range: min={input_ggml.min():.6f}, max={input_ggml.max():.6f}')

# Debug tensor 09 = model.9 output
print(f'\ndbg09 (model.9 output): {len(dbg09)} floats')
print(f'dbg09 RMS: {np.sqrt(np.mean(dbg09**2)):.6f}')

# Load model.9 weight using same approach as check_model9.py
# Read GGUF directly
import gguf
# Try multiple paths
gguf_paths = [
    r'E:\voxcpm-cpp\voxcpm2_v1.gguf',
    r'E:\voxcpm-cpp\voxcpm2_v1_f32.gguf',
    r'E:\voxcpm-cpp\voxcpm2_v2_full.gguf',
]
reader = None
for p in gguf_paths:
    if os.path.exists(p):
        reader = gguf.GGUFReader(p)
        print(f'Loaded GGUF: {p}')
        break
if reader is None:
    print('ERROR: no GGUF file found')
    exit(1)

# Get model.9 weight
tensor_name = 'audio_vae.decoder.model.9.weight.weight'
w9 = None
for t in reader.tensors:
    if t.name == tensor_name:
        w9 = t
        break

if w9 is None:
    print("ERROR: Could not find tensor")
    exit(1)

print(f'\nWeight: {w9.name}')
print(f'  shape: {w9.shape}')  # ggml shape
print(f'  dtype: {w9.tensor_type}')
print(f'  nbytes: {w9.n_bytes}')

# Get F32 values
weight_f32 = w9.data.astype(np.float32) if w9.tensor_type.name == 'F16' else w9.data
print(f'  weight F32 shape: {weight_f32.shape}')  # numpy shape from gguf reader

# The gguf reader flattens to 1D for F16
# For ggml shape [7, 32, 1] in col-major:
#   data[k + 7*ic + 7*32*oc] = weight at (k=ne[0], ic=ne[1], oc=ne[2])
# For oc=0: data[k + 7*ic]
# Let's reshape properly
# The gguf reader already gives us the weight in numpy shape (OC, IC, K) = (1, 32, 7)
w_pyt = weight_f32.astype(np.float32)
print(f'Weight shape: {w_pyt.shape}')
w_rms = np.sqrt(np.mean(w_pyt**2))
print(f'Weight RMS: {w_rms:.6f}')
# Verify against check_model9.py first value
print(f'Weight[0,0,0] = {w_pyt[0,0,0]:.10f} (expect -0.0004057884)')

# Also load bias
b9_name = 'audio_vae.decoder.model.9.bias'
b9 = None
for t in reader.tensors:
    if t.name == b9_name:
        b9 = t
        break

bias = 0.0
if b9:
    bval = b9.data.astype(np.float32) if b9.tensor_type.name == 'F16' else b9.data
    bias = float(bval[0] if hasattr(bval, '__getitem__') else bval)
print(f'Bias: {bias:.10f}')

# 3. Manual conv1d computation in Python
# PyTorch conv1d with padding=3, stride=1, dilation=1
# Input: [N, IC] -> reshape to [1, IC, N] for PyTorch style
# Weight: [OC, IC, K]
# Output: [1, OC, N] with same padding

# Manual implementation:
N = input_ggml.shape[0]
OW = N  # same padding with K=7, pad=3, stride=1, dilation=1

# Pad input: reflect pad left and right
# For same padding with K=7, pad=3: pad left by 3, right by 3
# Then output OW = (N + 6 - 6)/1 + 1 = N
pad = 3
padded = np.zeros((N + 2*pad, IC), dtype=np.float32)
padded[pad:pad+N] = input_ggml
# Reflect padding
padded[:pad] = input_ggml[pad:0:-1] if pad > 0 else 0
padded[pad+N:] = input_ggml[N-1:N-1-pad:-1] if pad > 0 else 0

# Manual conv
out_manual = np.zeros((OW, OC), dtype=np.float32)
for t in range(OW):
    for oc in range(OC):
        s = 0.0
        for ic in range(IC):
            for k in range(K):
                s += w_pyt[oc, ic, k] * padded[t + k, ic]  # stride=1, dilation=1
        out_manual[t, oc] = s + bias

print(f'\nManual conv output: shape {out_manual.shape}')
print(f'Manual conv RMS: {np.sqrt(np.mean(out_manual**2)):.6f}')
print(f'Manual conv range: [{out_manual.min():.6f}, {out_manual.max():.6f}]')

# 4. Compare with C debug tensor 09
# dbg09 is column-major: [OW, OC] = [61440, 1]
# element (t, oc) = data[oc * OW + t] for oc=0 => just data[:OW]
dbg09_output = dbg09.reshape((OC, OW), order='F')  # OC=1, OW=61440 -> [1, 61440]
print(f'\ndbg09 RMS: {np.sqrt(np.mean(dbg09**2)):.6f}')
print(f'dbg09 range: [{dbg09.min():.6f}, {dbg09.max():.6f}]')

# Compare: manual vs C
diff = out_manual[:, 0] - dbg09
diff_rms = np.sqrt(np.mean(diff**2))
max_diff = np.max(np.abs(diff))
print(f'\nComparison: manual vs C debug tensor 09')
print(f'  Diff RMS: {diff_rms:.10f}')
print(f'  Max |diff|: {max_diff:.10f}')
print(f'  C output / manual output RMS ratio: {np.sqrt(np.mean(dbg09**2)) / np.sqrt(np.mean(out_manual[:, 0]**2)):.6f}')

# 5. Compare dbg09 with Python reference
vae_ref = np.load(r'E:\voxcpm-cpp\fixtures\ref\vae_decode_raw.npy')
# vae_ref shape: [1, 1, 61440]
vae_ref_f = vae_ref[0, 0, :]  # [61440]
# tanh inverse: before_tanh = atanh(vae_ref) but limited
before_tanh = np.arctanh(np.clip(vae_ref_f, -0.9999, 0.9999))

print(f'\nPython reference output:')
print(f'  RMS: {np.sqrt(np.mean(vae_ref_f**2)):.6f}')
print(f'  Range: [{vae_ref_f.min():.6f}, {vae_ref_f.max():.6f}]')
print(f'  Before-tanh RMS (approximate): {np.sqrt(np.mean(before_tanh**2)):.6f}')

# Compare model.9 output from C with Python reference before tanh
print(f'\nC model.9 output vs Python reference (before tanh):')
diff2 = dbg09[:OW] - before_tanh[:OW]
print(f'  Diff RMS: {np.sqrt(np.mean(diff2**2)):.10f}')
print(f'  Max |diff|: {np.max(np.abs(diff2)):.10f}')
print(f'  C / Python ratio: {np.sqrt(np.mean(dbg09**2)) / np.sqrt(np.mean(before_tanh**2)):.6f}')

# Also check: does at least the SIGN pattern match?
dbg_sign = np.sign(dbg09[:1000])
ref_sign = np.sign(before_tanh[:1000])
sign_match = np.mean(dbg_sign == ref_sign) * 100
print(f'\nSign agreement (first 1000 samples): {sign_match:.1f}%')
