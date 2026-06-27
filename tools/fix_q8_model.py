#!/usr/bin/env python3
"""Convert Q8_0 GGUF to keep norms/biases in F16, rest stays Q8_0."""
import gguf
import numpy as np
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "voxcpm2_v2_q8_0.gguf"
dst = src.replace(".gguf", "_fixed.gguf")

print(f"Reading {src}...")
r = gguf.GGUFReader(src)

# Types
Q8_0 = 8
F16 = 1
F32 = 0

norms_and_biases = 0
other_tensors = 0

w = gguf.GGUFWriter(dst, "voxcpm2")

# Copy metadata
for k, v in r.fields.items():
    p = v.parts
    t = v.types
    if len(p) == 1:
        if gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.STRING:
            w.add_string(k, str(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.INT32:
            w.add_int(k, int(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.FLOAT32:
            w.add_float32(k, float(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.INT64:
            w.add_int(k, int(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.FLOAT64:
            w.add_float64(k, float(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.BOOL:
            w.add_bool(k, bool(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.UINT32:
            w.add_uint32(k, int(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.INT8:
            w.add_int(k, int(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.INT16:
            w.add_int(k, int(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.INT32:
            w.add_int(k, int(p[0]))
        elif gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.INT64:
            w.add_int(k, int(p[0]))
        else:
            w.add_key(k)
    elif len(p) > 1 and gguf.GGUFValueType(t[0]) == gguf.GGUFValueType.ARRAY:
        # Copy array values
        arr_type = gguf.GGUFValueType(t[1])
        if arr_type == gguf.GGUFValueType.STRING:
            w.add_string_array(k, [str(x) for x in p])
        elif arr_type == gguf.GGUFValueType.INT32:
            w.add_int_array(k, [int(x) for x in p])
        elif arr_type == gguf.GGUFValueType.FLOAT32:
            w.add_float32_array(k, [float(x) for x in p])
        elif arr_type == gguf.GGUFValueType.INT64:
            w.add_int_array(k, [int(x) for x in p])
        elif arr_type == gguf.GGUFValueType.FLOAT64:
            w.add_float64_array(k, [float(x) for x in p])
        else:
            w.add_key(k)
    else:
        w.add_key(k)

# Set file type
w.add_int("general.file_type", 1)  # F16 mixed

# Copy tensors, dequantize norms and biases
import struct
for t in r.tensors:
    name = t.name
    is_norm_or_bias = ('norm' in name or 'bias' in name or 
                      'ln' in name or 'rms' in name)
    
    if t.tensor_type == Q8_0 and is_norm_or_bias:
        # Dequantize Q8_0 to F16
        raw = np.array(t.data, dtype=np.uint8)
        shape = list(t.shape)
        total_elems = int(np.prod(shape))
        
        # Q8_0: blocks of 32 elements, each block = 2 bytes (f16 scale) + 32 bytes (int8 qs) = 34 bytes
        n_blocks = total_elems // 32
        assert total_elems % 32 == 0, f"{name}: size {total_elems} not divisible by 32"
        
        dequant = np.zeros(total_elems, dtype=np.float32)
        for b in range(n_blocks):
            offset = b * 34
            # f16 scale (2 bytes, little-endian)
            scale = struct.unpack('<e', raw[offset:offset+2].tobytes())[0]
            # 32 int8 quantized values
            qs = raw[offset+2:offset+34].astype(np.int8)
            for i in range(32):
                dequant[b * 32 + i] = float(qs[i]) * scale
        
        dequant_f16 = dequant.astype(np.float16)
        w.add_tensor(name, dequant_f16.reshape(shape))
        norms_and_biases += 1
        print(f"  Dequant {name}: {shape}")
    elif t.tensor_type in (F16, Q8_0, F32):
        # Keep as-is
        data = np.array(t.data)
        w.add_tensor(name, data)
        other_tensors += 1
    else:
        data = np.array(t.data)
        w.add_tensor(name, data)
        other_tensors += 1

print(f"\nDequantized {norms_and_biases} norms/biases to F16")
print(f"Kept {other_tensors} tensors unchanged")
print(f"Writing {dst}...")
w.write()
w.close()

print("Done!")
