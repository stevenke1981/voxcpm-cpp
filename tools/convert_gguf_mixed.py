#!/usr/bin/env python3
"""Convert GGUF: LM/RALM/DiT/Proj to F32, keep AudioVAE as F16."""

import sys, logging, numpy as np, gguf
from gguf import GGUFValueType, GGMLQuantizationType
log = logging.getLogger(__name__)

def get_value(v):
    try:
        return v.contents()
    except Exception:
        return None

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 3:
        print("Usage: python convert_gguf_mixed.py input.gguf output.gguf")
        return 1
    in_path, out_path = sys.argv[1], sys.argv[2]

    log.info("Reading %s...", in_path)
    reader = gguf.GGUFReader(in_path)
    arch = None
    for k, v in reader.fields.items():
        if k == 'general.architecture':
            arch = get_value(v); break
    if not arch:
        log.error("Architecture not found"); return 1

    log.info("Architecture: %s, tensors: %d", arch, len(reader.tensors))
    writer = gguf.GGUFWriter(out_path, arch)

    # Copy KV metadata
    skip = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}
    for k, v in reader.fields.items():
        if k in skip: continue
        try:
            val = get_value(v)
            if val is None: continue
            tn = [t.name for t in v.types]
            if tn == ['STRING']:     writer.add_string(k, val)
            elif tn == ['BOOL']:     writer.add_bool(k, val)
            elif tn in (['INT8'],['INT16'],['INT32']): writer.add_int32(k, val)
            elif tn == ['UINT8']:    writer.add_uint8(k, val)
            elif tn == ['UINT16']:   writer.add_uint16(k, val)
            elif tn == ['UINT32']:   writer.add_uint32(k, val)
            elif tn in (['UINT64'],['INT64']): writer.add_uint64(k, val)
            elif tn == ['FLOAT32']:  writer.add_float32(k, val)
            elif tn == ['FLOAT64']:  writer.add_float64(k, val)
            elif tn[0] == 'ARRAY':   writer.add_array(k, val)
            else:                    writer.add_string(k, str(val))
        except Exception as e:
            log.warning("Skipping KV %s: %s", k, e)

    log.info("KV metadata copied.")

    # Write tensors: convert to F32 except audio_vae.* (keep as F16)
    f16_kept = f16_to_f32 = f32_kept = other = 0
    total = len(reader.tensors)
    for idx, t in enumerate(reader.tensors):
        is_audio_vae = t.name.startswith("audio_vae.")
        arr = t.data

        if is_audio_vae:
            # Keep AudioVAE tensors as-is (F16)
            if arr.dtype == np.float16:
                writer.add_tensor(t.name, arr, raw_dtype=GGMLQuantizationType.F16)
                f16_kept += 1
            elif t.tensor_type == GGMLQuantizationType.F32:
                writer.add_tensor(t.name, arr, raw_dtype=GGMLQuantizationType.F32)
                f32_kept += 1
            else:
                writer.add_tensor(t.name, arr)
                other += 1
        else:
            # Non-AudioVAE: convert F16 to F32
            if arr.dtype == np.float16:
                writer.add_tensor(t.name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
                f16_to_f32 += 1
            elif t.tensor_type == GGMLQuantizationType.F32:
                writer.add_tensor(t.name, arr, raw_dtype=GGMLQuantizationType.F32)
                f32_kept += 1
            else:
                writer.add_tensor(t.name, arr)
                other += 1

        if (idx+1) % 100 == 0:
            log.info("  %d/%d tensors...", idx+1, total)

    log.info("F16->F32: %d, F16 kept (AudioVAE): %d, F32 kept: %d, other: %d, total: %d",
             f16_to_f32, f16_kept, f32_kept, other, total)
    log.info("Writing %s...", out_path)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("Done: %s", out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
