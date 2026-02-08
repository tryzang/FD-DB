import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, padding_mode="reflect", affine=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(channels, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(channels, affine=affine),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvBlock(nn.Module):
    """
    Basic conv -> norm -> ReLU followed by a residual block, stride=1 to keep resolution.
    """

    def __init__(self, in_channels, out_channels, padding_mode="reflect", affine=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_channels, affine=affine),
            nn.ReLU(inplace=False),
            ResidualBlock(out_channels, padding_mode=padding_mode, affine=affine),
        )

    def forward(self, x):
        return self.net(x)
