import torch

from models.discriminator import Discriminator
from losses.wgan_gp import gradient_penalty

device = "cuda" if torch.cuda.is_available() else "cpu"

critic = Discriminator().to(device)

real = torch.randn(
    4,
    3,
    256,
    256,
    device=device
)

fake = torch.randn(
    4,
    3,
    256,
    256,
    device=device
)

gp = gradient_penalty(
    critic,
    real,
    fake,
    device
)

print("Gradient Penalty:", gp.item())