#!/usr/bin/env python3
"""
Binary-search which base_lm Q8_0 layers produce NaN in the MiniCPM4 forward pass.

Usage:
    python tools/bisect_q8.py               # full auto binary search
    python tools/bisect_q8.py --test-range 0 10  # test layers 0-9 as Q8_0, rest F16
"""
import gguf
import numpy as np
import struct
import sys
import os
import subprocess
import re

SRC = "E:\\voxcpm-cpp\\voxcpm2_v2_q8_0.gguf"
DST_TEMPLATE = "E:\\voxcpm-cpp\\bisect_test.gguf"
VCPM_BIN = "E:\\voxcpm-cpp\\build\\Release\\voxcpm-c.exe"
WORK_DIR = "E:\\voxcpm-cpp"

def dequantize_q8_to_f16(raw_bytes, shape, name):
    raw = np.asarray(raw_bytes, dtype=np.uint8).flatten()
    total_elems = int(np.prod(shape))
    n_blocks = total_elems // 32
    if total_elems % 32 != 0 or raw.size != n_blocks * 34:
        return None
    dequant = np.zeros(total_elems, dtype=np.float32)
    try:
        for b in range(n_blocks):
            offset = b * 34
            scale = struct.unpack("<e", bytes(raw[offset:offset+2].tolist()))[0]
            qs = raw[offset+2:offset+34].astype(np.int8)
            for i in range(32):
                dequant[b * 32 + i] = float(qs[i]) * scale
    except Exception:
        return None
    dequant_f16 = dequant.astype(np.float16)
    numpy_shape = list(reversed(shape))
    return dequant_f16.reshape(numpy_shape)

def parse_layer(name):
    """Extract layer index from tensor name, e.g. 'base_lm.blk.12.self_attn.q_proj.weight' -> 12"""
    m = re.search(r'base_lm\.blk\.(\d+)', name)
    if m:
        return int(m.group(1))
    return None

def make_model(dst_path, q8_layer_range):
    """
    Create model where base_lm layers NOT in q8_layer_range are dequantized to F16.
    Layers in q8_layer_range remain Q8_0.
    """
    r = gguf.GGUFReader(SRC)
    w = gguf.GGUFWriter(dst_path, "voxcpm2")

    # Copy metadata
    for k, v in r.fields.items():
        if k in {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture", "general.file_type"}:
            continue
        gt = gguf.GGUFValueType(v.types[0])
        try:
            if gt == gguf.GGUFValueType.STRING:
                s = bytes(np.asarray(v.parts[v.data[0]]).tobytes()).rstrip(b"\x00").decode("utf-8", errors="replace")
                w.add_string(k, s)
            elif gt == gguf.GGUFValueType.INT32:
                w.add_int32(k, int(np.asarray(v.parts[v.data[0]]).item()))
            elif gt == gguf.GGUFValueType.INT64:
                w.add_int64(k, int(np.asarray(v.parts[v.data[0]]).item()))
            elif gt == gguf.GGUFValueType.FLOAT32:
                w.add_float32(k, float(np.asarray(v.parts[v.data[0]]).item()))
            elif gt == gguf.GGUFValueType.FLOAT64:
                w.add_float64(k, float(np.asarray(v.parts[v.data[0]]).item()))
            elif gt == gguf.GGUFValueType.UINT32:
                w.add_uint32(k, int(np.asarray(v.parts[v.data[0]]).item()))
            elif gt == gguf.GGUFValueType.BOOL:
                w.add_bool(k, bool(np.asarray(v.parts[v.data[0]]).item()))
            elif gt == gguf.GGUFValueType.ARRAY:
                arr_type = gguf.GGUFValueType(v.types[1])
                if arr_type == gguf.GGUFValueType.STRING:
                    strs = [bytes(np.asarray(v.parts[idx])).decode("utf-8", errors="replace") for idx in v.data]
                    w.add_array(k, strs)
                elif arr_type == gguf.GGUFValueType.INT32:
                    vals = [int(np.asarray(v.parts[idx]).item()) for idx in v.data]
                    w.add_array(k, vals)
                elif arr_type == gguf.GGUFValueType.FLOAT32:
                    vals = [float(np.asarray(v.parts[idx]).item()) for idx in v.data]
                    w.add_array(k, vals)
        except Exception as e:
            print(f"  Warning: skipping metadata {k}: {e}")

    w.add_int32("general.file_type", 1)

    n_dequant = 0
    n_q8 = 0
    n_other = 0

    for t in r.tensors:
        name = t.name
        layer = parse_layer(name)
        
        # Determine if this tensor should be Q8_0 (test range) or dequantized to F16
        is_norm_or_bias = ("norm" in name or "bias" in name or "ln" in name or "rms" in name)
        is_base_lm_weight = layer is not None and not is_norm_or_bias
        
        # For base_lm weight tensors: Q8_0 if layer in test range, else dequantize
        if t.tensor_type == 8 and is_base_lm_weight and layer is not None:
            if q8_layer_range is None or (q8_layer_range[0] <= layer <= q8_layer_range[1]):
                # Keep as Q8_0
                w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                n_q8 += 1
            else:
                # Dequantize to F16
                arr = dequantize_q8_to_f16(t.data, list(t.shape), name)
                if arr is not None:
                    w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
                    n_dequant += 1
                else:
                    w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                    n_q8 += 1
        elif t.tensor_type == 8 and (is_norm_or_bias or name.startswith("base_lm.")):
            # Dequantize all base_lm norms to F16 (they're safe and needed)
            arr = dequantize_q8_to_f16(t.data, list(t.shape), name)
            if arr is not None:
                w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F16)
                n_dequant += 1
            else:
                w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                n_q8 += 1
        else:
            if t.tensor_type == 8:
                w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                n_q8 += 1
            elif t.tensor_type == 1:
                w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.F16)
            elif t.tensor_type == 0:
                w.add_tensor(name, t.data, raw_dtype=gguf.GGMLQuantizationType.F32)
            else:
                w.add_tensor(name, t.data)
            n_other += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {
        'dequant': n_dequant,
        'q8': n_q8,
        'other': n_other,
        'test_range': q8_layer_range,
    }

def run_inference(model_path):
    """Run inference and extract NaN count from prompt_base_hidden."""
    env = os.environ.copy()
    env["VCPM_DEBUG_SHAPES"] = "1"
    
    try:
        result = subprocess.run(
            [VCPM_BIN, "tts", "--model", model_path, "--text", "Hi",
             "--steps", "1", "--min-len", "1", "--max-len", "1",
             "--out", os.path.join(WORK_DIR, "bisect_test.wav"),
             "--backend", "cpu"],
            capture_output=True, text=True, timeout=300,
            env=env, cwd=WORK_DIR
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    
    stderr = result.stderr
    
    # Extract NaN count from prompt_base_hidden
    m = re.search(r'VCPM_NAN "prompt_base_hidden" \[(\d+)\]: NaN=(\d+)', stderr)
    if m:
        total_elems = int(m.group(1))
        nan_count = int(m.group(2))
        return {"nan": nan_count, "total": total_elems}
    
    # Check for assertion failure
    if "Assertion failed" in stderr:
        return {"error": "assertion_failure"}
    
    return {"error": "no_nan_output", "stderr_tail": stderr[-500:]}

def binary_search():
    """Binary search to find the first Q8_0 layer that causes NaN."""
    # We know layers 0-39 all at Q8_0 produces NaN (check)
    # We know all at F16 produces valid (check)
    
    # First confirm all-F16 works
    print("=== Step 0: All layers F16 (baseline valid) ===")
    stats = make_model(DST_TEMPLATE, None)  # None means dequantize all
    # Actually let me use range (-1, -1) to mean dequantize all
    # Wait, I used None to mean dequantize all base_lm weights.
    # Actually looking at my code: if q8_layer_range is None, ALL base_lm weights are dequantized
    
    result = run_inference(DST_TEMPLATE)
    print(f"  Result: {result}")
    if result.get("nan", -1) != 0:
        print("ERROR: all-F16 baseline produces NaN. Something is wrong.")
        return
    
    print("  All-F16 baseline: OK (NaN=0)")
    
    # Binary search on layers
    # Strategy: find which LAYER (when Q8_0) introduces NaN
    # We test consecutive layer NO - find the transition point
    
    # First test: all layers Q8_0 -> should produce NaN
    print("\n=== Full Q8_0 (baseline NaN) ===")
    stats = make_model(DST_TEMPLATE, (0, 39))
    result = run_inference(DST_TEMPLATE)
    print(f"  Result: {result}")
    has_nan_full = result.get("nan", 0) > 0
    
    if not has_nan_full:
        print("  Full Q8_0 does NOT produce NaN. All layers safe at Q8_0?")
        print("  The issue might be elsewhere (norms, not weights).")
        return
    
    print("  Full Q8_0: NaN confirmed")
    
    # Binary search: find the FIRST layer that produces NaN when Q8_0
    # (with all earlier layers F16, all later layers F16)
    # We test each layer individually
    
    print("\n=== Testing individual layers ===")
    nan_layers = []
    for layer in range(40):
        # Keep only this layer at Q8_0, rest F16
        stats = make_model(DST_TEMPLATE, (layer, layer))
        result = run_inference(DST_TEMPLATE)
        nan_count = result.get("nan", -1)
        status = "NaN" if nan_count > 0 else "OK"
        print(f"  Layer {layer:2d} Q8_0: {status} (NaN={result.get('nan', '?')})")
        if nan_count > 0:
            nan_layers.append(layer)
    
    print(f"\n=== Layers that cause NaN when Q8_0: {nan_layers} ===")
    
    if nan_layers:
        # Test each problematic layer in isolation
        print("\n=== Testing problematic layers one at a time ===")
        for l in nan_layers:
            stats = make_model(DST_TEMPLATE, (l, l))
            result = run_inference(DST_TEMPLATE)
            nan_count = result.get("nan", 0)
            print(f"  Single layer {l} Q8_0 (rest F16): NaN={nan_count}")

if __name__ == "__main__":
    if "--test-range" in sys.argv:
        idx = sys.argv.index("--test-range")
        lo, hi = int(sys.argv[idx+1]), int(sys.argv[idx+2])
        print(f"Testing Q8_0 range: layers [{lo}, {hi}]")
        stats = make_model(DST_TEMPLATE, (lo, hi))
        print(f"  Stats: {stats}")
        result = run_inference(DST_TEMPLATE)
        print(f"  Result: {result}")
    else:
        binary_search()
