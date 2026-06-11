import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class InputBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class OutputBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=9):
        super().__init__()

        self.input_block = InputBlock()

        self.down1 = DownsampleBlock(
            in_channels=64,
            out_channels=128
        )

        self.down2 = DownsampleBlock(
            in_channels=128,
            out_channels=256
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(256)
                for _ in range(num_residual_blocks)
            ]
        )

        self.up1 = UpsampleBlock(
            in_channels=256,
            out_channels=128
        )

        self.up2 = UpsampleBlock(
            in_channels=128,
            out_channels=64
        )

        self.output_block = OutputBlock()

    def forward(self, x):
        x = self.input_block(x)

        x = self.down1(x)
        x = self.down2(x)

        x = self.residual_blocks(x)

        x = self.up1(x)
        x = self.up2(x)

        x = self.output_block(x)

        return x