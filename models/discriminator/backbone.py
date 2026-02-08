import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, use_norm=True, affine=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not use_norm,
            )
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=affine))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PatchBackbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_layers=3, affine=True):
        super().__init__()
        layers = []
        current_channels = in_channels
        out_channels = base_channels
        for layer_idx in range(num_layers):
            stride = 2 if layer_idx < num_layers - 1 else 1
            use_norm = layer_idx > 0
            layers.append(
                ConvBlock(
                    current_channels,
                    out_channels,
                    stride=stride,
                    use_norm=use_norm,
                    affine=affine,
                )
            )
            current_channels = out_channels
            out_channels = min(out_channels * 2, 512)
        self.model = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x):
        return self.model(x)
