import torch

from losses.perceptual_loss import VGGPerceptualLoss

loss_fn = VGGPerceptualLoss()

fake = torch.randn(
    1, 3, 256, 256,
    requires_grad=True
)

sharp = torch.randn(
    1, 3, 256, 256
)

loss = loss_fn(fake, sharp)

loss.backward()

print("Loss:", loss.item())
print("Gradient exists:",
      fake.grad is not None)
