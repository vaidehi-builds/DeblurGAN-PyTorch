import torch

from models.discriminator import Discriminator

x = torch.randn(4, 3, 256, 256)

disc = Discriminator()

y = disc(x)

print("Input :", x.shape)
print("Output:", y.shape)