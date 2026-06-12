import torch

from models.discriminator import DiscriminatorBlock

x = torch.randn(4, 64, 128, 128)

block = DiscriminatorBlock(
    in_channels=64,
    out_channels=128,
    stride=2
)

y = block(x)

print("Input :", x.shape)
print("Output:", y.shape)