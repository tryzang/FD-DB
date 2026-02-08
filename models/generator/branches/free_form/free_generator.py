import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...blocks import ResidualBlock


class FreeFormGenerator(nn.Module):
    """
    CUT-style ResNet generator: 7x7 stem -> 2x downsampling -> ResBlocks -> 2x upsampling -> 7x7 head.
    No noise concatenation; matches paper structure and exports symmetric features for PatchNCE.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_channels = config.get("in_channels", 3)
        self.base_channels = config.get("base_channels", 64)
        self.num_res_blocks = max(int(config.get("num_blocks", 9)), 1)
        self.use_tanh = bool(config.get("use_tanh", True))
        self.use_checkpoint = bool(config.get("checkpoint", False))
        self.padding_mode = config.get("padding_mode", "reflect")
        self.use_affine_norm = bool(config.get("affine_norm", False))
        self.num_blocks = self.num_res_blocks

        model = [
            nn.Conv2d(
                self.in_channels,
                self.base_channels,
                kernel_size=7,
                padding=3,
                padding_mode=self.padding_mode,
                bias=False,
            ),
            nn.InstanceNorm2d(self.base_channels, affine=self.use_affine_norm),
            nn.ReLU(inplace=False),
        ]

        mult = 1
        self._feature_indices = []
        self._feature_indices.append(0)  # first conv index
        self._upsample_indices = []
        for _ in range(2):
            conv = nn.Conv2d(
                self.base_channels * mult,
                self.base_channels * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                padding_mode=self.padding_mode,
            )
            model += [
                conv,
                nn.InstanceNorm2d(self.base_channels * mult * 2, affine=self.use_affine_norm),
                nn.ReLU(inplace=False),
            ]
            self._feature_indices.append(len(model) - 3)  # conv index in model list
            mult *= 2

        # ResBlocks
        for _ in range(self.num_res_blocks):
            model += [
                ResidualBlock(
                    self.base_channels * mult,
                    padding_mode=self.padding_mode,
                    affine=self.use_affine_norm,
                )
            ]

        for _ in range(2):
            convt = nn.ConvTranspose2d(
                self.base_channels * mult,
                self.base_channels * mult // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            )
            model += [
                convt,
                nn.InstanceNorm2d(self.base_channels * mult // 2, affine=self.use_affine_norm),
                nn.ReLU(inplace=False),
            ]
            self._upsample_indices.append(len(model) - 3)  # convtranspose index
            mult //= 2

        model += [
            nn.Conv2d(
                self.base_channels,
                self.in_channels,
                kernel_size=7,
                padding=3,
                padding_mode=self.padding_mode,
            )
        ]

        self.model = nn.ModuleList(model)
        self.final_act = nn.Tanh() if self.use_tanh else nn.Identity()
        self._feature_channels = [
            self.base_channels,  # after first 7x7 conv
            self.base_channels * 2,  # after downsample1
            self.base_channels * 4,  # after downsample2 / bottleneck
            self.base_channels * 2,  # after upsample1
            self.base_channels,  # after upsample2
        ]

    @property
    def enco_channels(self):
        return self._feature_channels

    def forward(self, x, out_mode: str = "full"):
        h = x
        features = []

        for idx, layer in enumerate(self.model):
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)

            if idx in self._feature_indices or idx in self._upsample_indices:
                features.append(h)

        y_hat = self.final_act(h)

        enc_feats = features
        dec_feats = features

        if str(out_mode).lower() == "residual":
            y_out = y_hat - x
        else:
            y_out = y_hat
        return y_out, enc_feats, dec_feats
