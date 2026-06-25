#!/usr/bin/env python3
"""Debug array fields in GGUF reader."""
import gguf

reader = gguf.GGUFReader(r'E:\voxcpm-cpp\voxcpm2_v1.gguf')

# Find array fields
for k, v in reader.fields.items():
    type_names = [t.name for t in v.types]
    if 'ARRAY' in type_names:
        try:
            contents = v.contents()
            if isinstance(contents, list):
                print(f'{k}: type_names={type_names}, contents_len={len(contents)}, contents_type={type(contents[0]).__name__ if contents else "empty"}, first={str(contents[0])[:60] if contents else "N/A"}')
            else:
                print(f'{k}: type_names={type_names}, contents={type(contents).__name__} = {str(contents)[:80]}')
        except Exception as e:
            print(f'{k}: type_names={type_names}, contents error: {e}')
