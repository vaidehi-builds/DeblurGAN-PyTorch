import torch

from models.generator import DownsampleBlock

x = torch.randn(4, 64, 256, 256)

block = DownsampleBlock(
    in_channels=64,
    out_channels=128
)

y = block(x)

print("Input :", x.shape)
print("Output:", y.shape)