import torch

from models.generator import Generator

x = torch.randn(4, 3, 256, 256)

generator = Generator()

y = generator(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)