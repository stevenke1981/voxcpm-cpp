#!/usr/bin/env python3
"""Debug GGUFReader field structure."""
import gguf

reader = gguf.GGUFReader(r'E:\voxcpm-cpp\voxcpm2_v1.gguf')
k, v = list(reader.fields.items())[3]
print(f'Key: {k}')
print(f'  type: {type(v).__name__}')
print(f'  types: {v.types}')
print(f'  types names: {[t.name for t in v.types]}')

# Check all attributes
for attr in dir(v):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(v, attr)
        if callable(val):
            try:
                r = val()
                print(f'  {attr}(): -> {type(r).__name__} = {str(r)[:100]}')
            except Exception as e:
                print(f'  {attr}(): error: {e}')
        else:
            print(f'  {attr}: {type(val).__name__} = {str(val)[:200]}')
    except Exception as e:
        print(f'  {attr}: error: {e}')
