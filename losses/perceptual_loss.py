import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg19(
            weights=models.VGG19_Weights.IMAGENET1K_V1
        )
        
        self.features = nn.Sequential(
            *list(vgg.features.children())[:35]
        )
        vgg.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )

    def forward(self, generated, target):
        generated = (generated + 1) / 2
        target = (target + 1) / 2

        generated = self.normalize(generated)
        target = self.normalize(target)
        generated_features = self.features(generated)

        with torch.no_grad():
            target_features = self.features(target)

        loss = self.criterion(
            generated_features,
            target_features
        )
        
        return loss