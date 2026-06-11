import torch

from models.generator import UpsampleBlock

x = torch.randn(4, 256, 64, 64)

block = UpsampleBlock(
    in_channels=256,
    out_channels=128
)

y = block(x)

print("Input :", x.shape)
print("Output:", y.shape)