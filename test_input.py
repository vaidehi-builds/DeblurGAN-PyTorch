import torch

from models.generator import InputBlock

x = torch.randn(4, 3, 256, 256)

model = InputBlock()

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)