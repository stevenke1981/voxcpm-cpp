"""Inspect audiovae.pth structure for converter integration."""
import torch
import os

f = 'E:/voxcpm-cpp/model_download/audiovae.pth'
size_mb = os.path.getsize(f) / (1024*1024)
print(f'File size: {size_mb:.1f} MB')

ckpt = torch.load(f, map_location='cpu', weights_only=True)
print(f'Type: {type(ckpt)}')

if isinstance(ckpt, dict):
    print(f'Top-level keys ({len(ckpt)}):')
    for k, v in list(ckpt.items())[:5]:
        if hasattr(v, 'shape'):
            print(f'  {k}: {v.shape} [{v.dtype}]')
        else:
            print(f'  {k}: {type(v)}')

    # Check for state_dict
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        print(f'state_dict keys ({len(sd)}):')
        for k, v in list(sd.items()):
            print(f'  {k}: {v.shape} [{v.dtype}]')
    elif hasattr(next(iter(ckpt.values())), 'shape'):
        # Flat dict of tensors
        print(f'All tensors ({len(ckpt)}):')
        for k, v in ckpt.items():
            print(f'  {k}: {v.shape} [{v.dtype}]')
    else:
        print('Mixed content:')
        for k, v in ckpt.items():
            if hasattr(v, 'shape'):
                print(f'  [tensor] {k}: {v.shape} [{v.dtype}]')
            elif isinstance(v, dict):
                print(f'  [dict] {k}: {len(v)} keys')
                for k2, v2 in list(v.items())[:3]:
                    if hasattr(v2, 'shape'):
                        print(f'    {k2}: {v2.shape} [{v2.dtype}]')
                    else:
                        print(f'    {k2}: {type(v2)}')
