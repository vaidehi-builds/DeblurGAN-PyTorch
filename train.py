import torch
from torch.utils.data import DataLoader

from datasets.gopro_dataset import GoProDataset

from models.generator import Generator
from models.discriminator import Discriminator

from losses.perceptual_loss import VGGPerceptualLoss
from losses.wgan_gp import (
    critic_loss,
    generator_loss
)
from torchvision.utils import save_image
import os

def save_samples(
    generator,
    blur,
    sharp,
    epoch
):
    generator.eval()

    with torch.no_grad():
        fake = generator(blur)

    os.makedirs(
        "outputs/samples",
        exist_ok=True
    )

    save_image(
        (blur + 1) / 2,
        f"outputs/samples/epoch_{epoch}_blur.png"
    )

    save_image(
        (fake + 1) / 2,
        f"outputs/samples/epoch_{epoch}_fake.png"
    )

    save_image(
        (sharp + 1) / 2,
        f"outputs/samples/epoch_{epoch}_sharp.png"
    )

    generator.train()
# device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# dataset
dataset = GoProDataset(
    root_dir=r"D:\Datasets\GOPRO",
    split="train",
    patch_size=256
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)
sample_blur, sample_sharp = next(iter(loader))
# models
generator = Generator().to(device)
critic = Discriminator().to(device)
# send samples to GPU
sample_blur = sample_blur.to(device)
sample_sharp = sample_sharp.to(device)
# loss
perceptual_loss = VGGPerceptualLoss().to(device)

#optimizers
g_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=1e-4,
    betas=(0.5, 0.999)
)

d_optimizer = torch.optim.Adam(
    critic.parameters(),
    lr=1e-4,
    betas=(0.5, 0.999)
)

save_samples(
    generator,
    sample_blur,
    sample_sharp,
    epoch=0
)

num_epochs = 2

for epoch in range(num_epochs):

    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, (blur, sharp) in enumerate(loader):

        blur = blur.to(device)
        sharp = sharp.to(device)

        fake = generator(blur)

        d_optimizer.zero_grad()

        d_loss = critic_loss(
        critic,
        sharp,
        fake,
        lambda_gp=10,
        device=device
        )

        d_loss.backward()

        d_optimizer.step()

        g_optimizer.zero_grad()

        fake = generator(blur)

        perc_loss = perceptual_loss(
        fake,
        sharp
        )

        adv_loss = generator_loss(
        critic,
        fake
        )

        g_loss = (
        perc_loss
        + 0.001 * adv_loss
        )

        g_loss.backward()

        g_optimizer.step()

        if batch_idx % 100 == 0:
            print(
            f"Batch {batch_idx+1}: "
            f"D={d_loss.item():.2f} "
            f"P={perc_loss.item():.4f} "
            f"A={adv_loss.item():.4f} "
            f"G={g_loss.item():.4f}"
            )
    save_samples(
        generator,
        sample_blur,
        sample_sharp,
        epoch=epoch + 1
        )