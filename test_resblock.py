import torch

from models.generator import ResidualBlock

x = torch.randn(4, 64, 256, 256)

block = ResidualBlock(64)

y = block(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)