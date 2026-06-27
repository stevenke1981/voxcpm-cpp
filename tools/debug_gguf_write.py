"""Debug GGUF write - check how Q8_0 tensors are stored."""
import gguf
import numpy as np

r = gguf.GGUFReader("voxcpm2_v2_q8_0.gguf")

# Check a Q8_0 tensor 
for t in r.tensors:
    if "input_layernorm.weight" in t.name:
        print(f"Tensor: {t.name}")
        print(f"  type: {t.tensor_type}")
        print(f"  shape: {t.shape}")
        print(f"  n_dims: {len(t.shape)}")
        print(f"  data shape: {np.asarray(t.data).shape}")
        print(f"  data dtype: {np.asarray(t.data).dtype}")
        
        # Check GGUF_QUANT_SIZES
        sizes = gguf.GGML_QUANT_SIZES
        block_size, type_size = sizes[8]  # Q8_0
        print(f"  Q8_0 block_size={block_size}, type_size={type_size}")
        
        # Expected
        n_elems = int(np.prod(list(t.shape)))
        expected_raw = n_elems // block_size * type_size
        actual_raw = np.asarray(t.data).size
        print(f"  n_elems={n_elems}, expected_raw={expected_raw}, actual_raw={actual_raw}")
        
        # Check quantization version
        print(f"  quantization_version: not available")
        break

# Also check: what happens when we re-read the written tensor?
# Write a temp model with one tensor
w = gguf.GGUFWriter("test_q8_debug.gguf", "test")
w.add_file_type(17)  # Q8_0
w.add_int32("test_key", 42)

# Copy the norm tensor as Q8_0 (raw copy)
for t in r.tensors:
    if "input_layernorm.weight" in t.name:
        data = np.asarray(t.data)
        print(f"\nRaw data shape: {data.shape}, dtype: {data.dtype}")
        print(f"Data bytes: {data.nbytes}")
        w.add_tensor("test_tensor", data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        break

print("\nWriting...")
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("Done!")

# Now read it back
print("\nReading back...")
r2 = gguf.GGUFReader("test_q8_debug.gguf")
for t2 in r2.tensors:
    print(f"  Read back: {t2.name}, type={t2.tensor_type}, shape={t2.shape}")
    d2 = np.asarray(t2.data)
    print(f"  Data shape: {d2.shape}, dtype: {d2.dtype}")
