import torch

from models.discriminator import Discriminator
from losses.wgan_gp import (
    critic_loss,
    generator_loss
)

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

d_loss = critic_loss(
    critic,
    real,
    fake,
    lambda_gp=10,
    device=device
)

g_loss = generator_loss(
    critic,
    fake
)

print("Critic Loss   :", d_loss.item())
print("Generator Loss:", g_loss.item())