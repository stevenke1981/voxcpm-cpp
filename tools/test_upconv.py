"""Test CausalTransposeConv1d with and without weight_norm."""
import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding
    def forward(self, x):
        raw = super().forward(x)
        trim = self.__padding * 2 - self.__output_padding
        return raw[..., :-trim] if trim > 0 else raw

def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))

# Without weight_norm
mod1 = CausalTransposeConv1d(2048, 1024, kernel_size=16, stride=8, padding=4, output_padding=0)
x = torch.randn(1, 2048, 8)

print("=== Without weight_norm ===")
print(f"type(mod1) = {type(mod1).__name__}")
has_padding = hasattr(mod1, "_CausalTransposeConv1d__padding")
print(f"has _CausalTransposeConv1d__padding = {has_padding}")
y1 = mod1(x)
print(f"Output shape: {y1.shape}")

# Now with weight_norm
print()
print("=== With weight_norm ===")
mod2 = WNCausalTransposeConv1d(2048, 1024, kernel_size=16, stride=8, padding=4, output_padding=0)
print(f"type(mod2) = {type(mod2).__name__}")
print(f"has .module: {hasattr(mod2, 'module')}")
if hasattr(mod2, "module"):
    inner = mod2.module
    print(f"inner type = {type(inner).__name__}")
    inner_has_padding = hasattr(inner, "_CausalTransposeConv1d__padding")
    print(f"inner has _CausalTransposeConv1d__padding = {inner_has_padding}")
    if hasattr(inner, "_CausalTransposeConv1d__padding"):
        print(f"  padding = {inner._CausalTransposeConv1d__padding}")
        print(f"  output_padding = {inner._CausalTransposeConv1d__output_padding}")
    # Call inner.forward directly
    raw = inner.forward(x)
    print(f"inner.forward(x) directly: {raw.shape}")

# Call the wrapped module
y2 = mod2(x)
print(f"Wrapped module output shape: {y2.shape}")

# Compare
print()
print("=== Comparison ===")
print(f"mod1 output: {y1.shape}")
print(f"mod2 output: {y2.shape}")
print(f"Same shape: {y1.shape == y2.shape}")
if y1.shape == y2.shape:
    print(f"Close: {torch.allclose(y1, y2)}")
