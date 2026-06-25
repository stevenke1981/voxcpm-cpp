#!/usr/bin/env python3
"""Debug GGUFReader field values for different types."""
import gguf

reader = gguf.GGUFReader(r'E:\voxcpm-cpp\voxcpm2_v1.gguf')

for k, v in list(reader.fields.items())[3:12]:
    type_name = v.types[0].name
    val = v.contents() if hasattr(v, 'contents') else None
    try:
        contents = v.contents()
        print(f'{k}: type={type_name}, contents={contents}')
    except Exception as e:
        print(f'{k}: type={type_name}, contents error: {e}')
    # Show parts
    for i, p in enumerate(v.parts):
        print(f'  part[{i}]: dtype={p.dtype}, shape={p.shape}, data={p[:10] if p.size < 100 else f"size={p.size}"}')
    print()
