import torch
import torch.nn as nn
from torchvision import models


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1
        )

        self.features = nn.Sequential(
            *list(vgg.features.children())[:35]
        )

        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

    def forward(self, generated, target):

        generated_features = self.features(generated)
        target_features = self.features(target)

        loss = self.criterion(
            generated_features,
            target_features
        )

        return loss