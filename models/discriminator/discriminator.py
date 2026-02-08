import torch.nn as nn

from .backbone import PatchBackbone


class PatchDiscriminator(nn.Module):
    """
    Unconditional PatchGAN discriminator without attention.
    Takes an image and outputs local logits.
    """

    def __init__(self, config):
        super().__init__()
        in_channels = config.get("in_channels", 3)
        base_channels = config.get("base_channels", 64)
        num_layers = config.get("num_layers", 3)
        affine = config.get("affine", True)
        self.backbone = PatchBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            num_layers=num_layers,
            affine=affine,
        )
        self.pred_head = nn.Conv2d(self.backbone.out_channels, 1, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.pred_head(features)
        return logits
