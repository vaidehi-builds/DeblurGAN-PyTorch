import torch

from losses.perceptual_loss import VGGPerceptualLoss

loss_fn = VGGPerceptualLoss()

x = torch.randn(1, 3, 256, 256)

features = loss_fn.features(x)

print("Input shape   :", x.shape)
print("Feature shape :", features.shape)