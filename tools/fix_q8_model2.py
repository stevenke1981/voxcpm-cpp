#!/usr/bin/env python3
"""
Fix Q8_0 GGUF model: dequantize selected tensors from Q8_0 to F16.

Default mode (--minimal, recommended):
    Dequantize only embed_tokens + all norm/bias tensors.
    Leaves all matmul weights (attention Q/K/V/O, MLP gate/up/down) at Q8_0.
    This is sufficient to fix the NaN issue: embed_tokens must be F16
    because the C code reads it via pointer cast as F16.

Full mode (--full):
    Dequantize ALL base_lm tensors (embed_tokens + all 40 layers).
    Produces a 4.0 GB model that also produces valid audio but uses more VRAM.

Usage:
    python tools/fix_q8_model2.py [input.gguf] [--full]
    Default input: voxcpm2_v2_q8_0.gguf
    Output: <input>_fixed.gguf
"""
import gguf
import numpy as np
import struct
import sys

# Parse args
full_dequant = "--full" in sys.argv
src = "voxcpm2_v2_q8_0.gguf"
for a in sys.argv[1:]:
    if a.startswith("--") or a.startswith("-"):
        continue
    src = a
    break
dst = src.replace(".gguf", "_fixed.gguf")
if src == dst:
    dst = src.rsplit(".", 1)[0] + "_fixed.gguf"

print(f"Reading {src}...")
print(f"Mode: {'FULL (all base_lm tensors)' if full_dequant else 'MINIMAL (embed_tokens + norms/biases)'}")

print(f"Reading {src}...")
r = gguf.GGUFReader(src)

w = gguf.GGUFWriter(dst, "voxcpm2")

def get_scalar_from_field(field):
    """Get scalar value from a ReaderField."""
    # data[0] is the index into parts for the value
    arr = np.asarray(field.parts[field.data[0]])
    if arr.ndim == 0:
        return arr.item()
    return arr.flatten()[0]

def get_str_from_field(field):
    """Get string from a ReaderField."""
    b = np.asarray(field.parts[field.data[0]]).tobytes()
    return b.rstrip(b"\x00").decode("utf-8", errors="replace")

# Copy metadata
skipped_keys = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "general.file_type"}

for k, v in r.fields.items():
    if k in skipped_keys:
        continue
    
    gt = gguf.GGUFValueType(v.types[0])
    
    try:
        if gt == gguf.GGUFValueType.STRING:
            s = get_str_from_field(v)
            if k == "tokenizer.ggml.model":
                w.add_tokenizer_model(s)
            elif k == "general.name":
                w.add_name(s)
            elif k == "general.description":
                w.add_description(s)
            else:
                w.add_string(k, s)
        elif gt == gguf.GGUFValueType.INT32:
            val = int(get_scalar_from_field(v))
            w.add_int32(k, val)
        elif gt == gguf.GGUFValueType.INT64:
            w.add_int64(k, int(get_scalar_from_field(v)))
        elif gt == gguf.GGUFValueType.FLOAT32:
            w.add_float32(k, float(get_scalar_from_field(v)))
        elif gt == gguf.GGUFValueType.FLOAT64:
            w.add_float64(k, float(get_scalar_from_field(v)))
        elif gt == gguf.GGUFValueType.BOOL:
            w.add_bool(k, bool(get_scalar_from_field(v)))
        elif gt == gguf.GGUFValueType.UINT32:
            w.add_uint32(k, int(get_scalar_from_field(v)))
        elif gt == gguf.GGUFValueType.ARRAY:
            arr_type = gguf.GGUFValueType(v.types[1])
            if arr_type == gguf.GGUFValueType.STRING:
                strs = [bytes(np.asarray(v.parts[idx])).decode("utf-8", errors="replace") for idx in v.data]
                if k == "tokenizer.ggml.tokens":
                    w.add_token_list(strs)
                elif k == "tokenizer.ggml.merges":
                    w.add_token_merges(strs)
                else:
                    w.add_array(k, strs)
            elif arr_type == gguf.GGUFValueType.INT32:
                vals = [int(np.asarray(v.parts[idx]).item()) for idx in v.data]
                w.add_array(k, vals)
            elif arr_type == gguf.GGUFValueType.FLOAT32:
                vals = [float(np.asarray(v.parts[idx]).item()) for idx in v.data]
                if k == "tokenizer.ggml.scores":
                    w.add_token_scores(vals)
                else:
                    w.add_array(k, vals)
            else:
                print(f"  Cannot copy array field {k} with element type {arr_type}")
        else:
            print(f"  Cannot copy field {k} with type {gt}")
    except Exception as e:
        print(f"  Error copying field {k}: {e}")

# Ensure file type is present (use INT32 to match GGUF convention)
w.add_int32("general.file_type", 1)

print(f"Metadata copied. Adding tensors...")

n_dequant = 0
n_kept = 0

def dequantize_q8_to_f16(raw_bytes, shape, name):
    """Dequantize Q8_0 bytes to F16 numpy array with proper shape."""
    raw = np.asarray(raw_bytes, dtype=np.uint8).flatten()
    total_elems = int(np.prod(shape))
    n_blocks = total_elems // 32
    
    if total_elems % 32 != 0:
        print(f"  SKIP (non-divisible): {name} shape={shape} elems={total_elems}")
        return None
    expected_raw = n_blocks * 34
    if raw.size != expected_raw:
        print(f"  SKIP (raw size mismatch): {name} raw_size={raw.size} expected={expected_raw}")
        return None
    
    dequant = np.zeros(total_elems, dtype=np.float32)
    try:
        for b in range(n_blocks):
            offset = b * 34
            scale_bytes = raw[offset:offset+2]
            if len(scale_bytes) < 2:
                raise ValueError("bad block")
            scale = struct.unpack("<e", scale_bytes.tobytes())[0]
            qs = raw[offset+2:offset+34].astype(np.int8)
            for i in range(32):
                dequant[b * 32 + i] = float(qs[i]) * scale
    except Exception as e:
        print(f"  SKIP (dequant error {e}): {name}")
        return None
    
    dequant_f16 = dequant.astype(np.float16)
    numpy_shape = list(reversed(shape))
    return dequant_f16.reshape(numpy_shape)

for t in r.tensors:
    name = t.name
    is_norm_or_bias = ("norm" in name or "bias" in name or 
                       "ln" in name or "rms" in name)
    is_embed_tokens = "embed_tokens" in name
    is_base_lm = name.startswith("base_lm.")
    
    should_dequant = is_norm_or_bias or is_embed_tokens or (full_dequant and is_base_lm)
    
    if t.tensor_type == 8 and should_dequant:
        arr = dequantize_q8_to_f16(t.data, list(t.shape), name)
        if arr is None:
            w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            n_kept += 1
            continue
        w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
        n_dequant += 1
        if n_dequant <= 5 or n_dequant % 50 == 0:
            print(f"  Dequant #{n_dequant}: {name} {list(t.shape)} (elems={int(np.prod(t.shape))})")
    else:
        # Pass raw Q8_0 or other data with correct raw_dtype (preserving byte shape)
        if t.tensor_type == 8:  # Q8_0
            w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif t.tensor_type == 1:  # F16
            w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.F16)
        elif t.tensor_type == 0:  # F32
            w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            w.add_tensor(name, t.data)
        n_kept += 1

mode_str = "all base_lm" if full_dequant else "embed_tokens + norms/biases"
print(f"\nDequantized {n_dequant} tensors ({mode_str}) from Q8_0 to F16")
print(f"Kept {n_kept} tensors unchanged")
print(f"Writing {dst}...")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("Done!")
