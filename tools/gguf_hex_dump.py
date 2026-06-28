"""Hex dump GGUF metadata section."""
import struct
import sys

with open(sys.argv[1], "rb") as f:
    data = f.read()

# GGUF3 header: magic(4) + version(4) + tensor_count(8) + kv_count(8)
magic = data[0:4]
assert magic == b"GGUF", f"Not GGUF: {magic}"
version = struct.unpack("<I", data[4:8])[0]
tensor_count = struct.unpack("<Q", data[8:16])[0]
kv_count = struct.unpack("<Q", data[16:24])[0]

print(f"GGUF version={version}, tensors={tensor_count}, KVs={kv_count}")

# Walk metadata keys
pos = 24
for _ in range(kv_count):
    # key length (string)
    key_len = struct.unpack("<Q", data[pos:pos+8])[0]
    pos += 8
    key = data[pos:pos+key_len].decode("utf-8")
    pos += key_len
    
    # value type
    val_type = struct.unpack("<I", data[pos:pos+4])[0]
    pos += 4
    
    if val_type == 0:  # UINT8
        val = data[pos]
        pos += 1
        print(f"  {key} (UINT8) = {val}")
    elif val_type == 1:  # INT8
        val = struct.unpack("<b", data[pos:pos+1])[0]
        pos += 1
        print(f"  {key} (INT8) = {val}")
    elif val_type == 2:  # UINT16
        val = struct.unpack("<H", data[pos:pos+2])[0]
        pos += 2
        print(f"  {key} (UINT16) = {val}")
    elif val_type == 3:  # INT16
        val = struct.unpack("<h", data[pos:pos+2])[0]
        pos += 2
        print(f"  {key} (INT16) = {val}")
    elif val_type == 4:  # UINT32
        val = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        print(f"  {key} (UINT32) = {val}")
    elif val_type == 5:  # INT32
        val = struct.unpack("<i", data[pos:pos+4])[0]
        pos += 4
        print(f"  {key} (INT32) = {val}")
    elif val_type == 6:  # UINT64
        val = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 8
        print(f"  {key} (UINT64) = {val}")
    elif val_type == 7:  # INT64
        val = struct.unpack("<q", data[pos:pos+8])[0]
        pos += 8
        print(f"  {key} (INT64) = {val}")
    elif val_type == 8:  # FLOAT32
        val = struct.unpack("<f", data[pos:pos+4])[0]
        pos += 4
        print(f"  {key} (FLOAT32) = {val}")
    elif val_type == 9:  # FLOAT64
        val = struct.unpack("<d", data[pos:pos+8])[0]
        pos += 8
        print(f"  {key} (FLOAT64) = {val}")
    elif val_type == 10:  # BOOL
        val = bool(data[pos])
        pos += 1
        print(f"  {key} (BOOL) = {val}")
    elif val_type == 11:  # STRING
        str_len = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 8
        val = data[pos:pos+str_len].decode("utf-8", errors="replace")
        pos += str_len
        print(f"  {key} (STRING) = {repr(val)}")
    elif val_type == 12:  # ARRAY
        arr_type = struct.unpack("<I", data[pos:pos+4])[0]
        pos += 4
        arr_len = struct.unpack("<Q", data[pos:pos+8])[0]
        pos += 8
        print(f"  {key} (ARRAY type={arr_type} len={arr_len})")
        # Skip array data
        if arr_type <= 9:  # primitive types
            elem_sizes = [1, 1, 2, 2, 4, 4, 8, 8, 4, 8]
            elem_size = elem_sizes[arr_type] if arr_type < len(elem_sizes) else 4
            pos += arr_len * elem_size
        elif arr_type == 10:  # bool
            pos += arr_len
        elif arr_type == 11:  # string
            for _ in range(arr_len):
                slen = struct.unpack("<Q", data[pos:pos+8])[0]
                pos += 8 + slen
    else:
        print(f"  {key} (UNKNOWN type={val_type}) at pos {pos}")
        break
