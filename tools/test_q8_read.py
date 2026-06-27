#!/usr/bin/env python3
"""Test reading Q8_0 tensor data."""
import gguf
import numpy as np

r = gguf.GGUFReader("voxcpm2_v2_q8_0.gguf")
for t in r.tensors:
    if "input_layernorm.weight" in t.name:
        print(f"Tensor: {t.name}")
        print(f"  type: {t.tensor_type}")
        print(f"  shape: {t.shape}")
        data = np.array(t.data)
        print(f"  numpy dtype: {data.dtype}")
        print(f"  numpy shape: {data.shape}")
        print(f"  first 5 raw bytes: {data[:5]}")
        break

# Check if the GGUF library gives us Q8_0 data as int8 blocks or as floats
t = next(t for t in r.tensors if "input_layernorm.weight" in t.name)
data = np.array(t.data)
print(f"\nRaw data stats: min={data.min()}, max={data.max()}, mean={data.mean():.4f}")
print(f"Data type: {data.dtype}")
print(f"Total elements: {data.size}")
# If it's Q8_0 it should be blocks of 33 bytes (2 bytes scale + 32 int8)
# But the GGUF reader might already dequantize it
# Check by looking at expected vs actual size
expected_q8 = 2048  # 2048 elements
expected_blocks = expected_q8 // 32  # 64 blocks
expected_raw_size = expected_blocks * 33  # 2112 bytes
print(f"Expected raw size (Q8_0 blocks): {expected_raw_size}")
print(f"Actual numpy size: {data.size}")
print(f"Actual numpy dtype itemsize: {data.dtype.itemsize}")
print(f"Actual memory: {data.size * data.dtype.itemsize} bytes")
