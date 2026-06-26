"""Debug CausalTransposeConv1d forward behavior."""
import torch
import torch.nn as nn
import math

class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding=0, output_padding=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding
        print(f"  __init__: args={args}, padding={padding}, output_padding={output_padding}")
        print(f"  self.padding (from base) = {self.padding}")
        print(f"  self._CausalTransposeConv1d__padding = {self._CausalTransposeConv1d__padding}")
        
    def forward(self, x):
        raw = super().forward(x)
        trim = self.__padding * 2 - self.__output_padding
        print(f"  forward: raw shape={raw.shape}, __padding={self.__padding}, __output_padding={self.__output_padding}, trim={trim}")
        if trim > 0:
            result = raw[..., :-trim]
            print(f"    after trim {trim}: {result.shape}")
        else:
            result = raw
        return result

mod1 = CausalTransposeConv1d(2048, 1024, kernel_size=16, stride=8, padding=4, output_padding=0)
x = torch.randn(1, 2048, 8)
y1 = mod1(x)
print(f"Output: {y1.shape}")
print()

# Also test with F.conv_transpose1d directly for comparison
w = torch.randn(2048, 1024, 16)
x2 = torch.randn(1, 2048, 8)

for p in [0, 4, 8]:
    y = torch.nn.functional.conv_transpose1d(x2, w, stride=8, padding=p)
    print(f"F.conv_transpose1d(padding={p}): {y.shape}")

# What does the standard formula give?
for L_in, s, p, k in [(8, 8, 4, 16), (8, 8, 0, 16), (8, 8, 8, 16)]:
    L = (L_in - 1) * s - 2 * p + 1 * (k - 1) + 0 + 1
    print(f"Formula(L_in={L_in}, s={s}, p={p}, k={k}): L_out={L}")
