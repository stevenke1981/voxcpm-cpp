#!/usr/bin/env python3
"""Check VAE decoder weight shapes and routing."""
import gguf
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "voxcpm2_v2_full.gguf"
reader = gguf.GGUFReader(path)

def get(name):
    for t in reader.tensors:
        if t.name == name:
            d = t.data
            if d.dtype == np.float16:
                d = d.astype(np.float32)
            return np.asarray(d)
    return None

print("=== All VAE decoder conv weights ===")
for t in reader.tensors:
    name = t.name
    if 'audio_vae.decoder' not in name:
        continue
    if not name.endswith('.weight.weight'):
        continue
    short = name.replace("audio_vae.decoder.", "")
    gshape = list(t.shape)  # ggml order
    nd = np.asarray(t.data)
    nshape = list(nd.shape)  # numpy order
    ne1 = gshape[1] if len(gshape) >= 2 else -1
    ne2 = gshape[2] if len(gshape) >= 3 else -1
    # Determine routing
    if len(gshape) >= 3:
        if ne1 == 1 and ne2 > 1:
            route = "depthwise"
        else:
            route = "regular"
    else:
        route = "bias/other"
    print(f"  {short:55s} ggml={str(gshape):20s} numpy={str(nshape):20s} ne1={ne1} ne2={ne2} => {route}")

print()
print("=== model.9 specific check ===")
w9_raw = get("audio_vae.decoder.model.9.weight.weight")
w9_ggml = np.transpose(w9_raw, (2, 1, 0))
print(f"model.9 raw numpy shape: {w9_raw.shape}")
print(f"model.9 ggml shape: {w9_ggml.shape}")
print(f"model.9 sum: {w9_ggml.sum():.8f}")

print()
print("=== Checking what test_vae_only uses ===")
# The test uses latent [8, 64]
# Let's see block.7 output from the test
# Actually let's trace with the actual test latent
import struct

# Read the test latent from the C source
print("See tools/test_vae_only.c for the test latent definition")
