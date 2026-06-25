"""Inspect model.safetensors structure."""
from safetensors import safe_open
from safetensors.torch import load_file
import os

f = 'E:/voxcpm-cpp/model_download/model.safetensors'
size_gb = os.path.getsize(f) / (1024**3)
print(f'File size: {size_gb:.2f} GB')

# Use PyTorch backend for bfloat16 support
tensors = []
with safe_open(f, framework='pt') as sf:
    for k in sf.keys():
        shape = list(sf.get_slice(k).get_shape())
        dtype = sf.get_tensor(k).dtype
        tensors.append((k, shape, str(dtype)))

print(f'Total tensors: {len(tensors)}')
tensors.sort(key=lambda x: x[0])
total_mb = 0
for k, shape, dtype_str in tensors:
    elements = 1
    for d in shape:
        elements *= d
    bytes_per = 4  # f32 (bf16 loaded as f32 in pt)
    if 'int' in dtype_str:
        bytes_per = 4
    size_mb = elements * bytes_per / (1024*1024)
    total_mb += size_mb
    print(f'  {k}: {shape} [{dtype_str}] ~{size_mb:.1f} MB')

print(f'\nTotal estimated size: {total_mb:.0f} MB')
