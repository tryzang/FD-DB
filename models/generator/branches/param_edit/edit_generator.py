from typing import Dict, Optional

import torch
import torch.nn as nn

from .edit_ops import EditOps
from .param_spec import ParamSpecSet, build_param_specs


class EditParamNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        base_channels: int = 32,
        num_layers: int = 3,
        mlp_hidden: int = 128,
    ):
        super().__init__()
        layers = []
        ch_in = in_channels
        ch = base_channels
        for _ in range(max(1, num_layers)):
            layers.append(
                nn.Conv2d(ch_in, ch, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.InstanceNorm2d(ch, affine=True))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch
            ch = min(ch * 2, 256)
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch_in, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        return self.mlp(pooled)


class EditBranch(nn.Module):
    def __init__(self, config: Dict, in_channels: int = 3):
        super().__init__()
        self.config = config
        param_cfg = config.get("param_spec", {})
        self.param_specs: ParamSpecSet = build_param_specs(param_cfg, in_channels=in_channels)

        net_cfg = config.get("param_net", {})
        base_channels = int(net_cfg.get("base_channels", 32))
        num_layers = int(net_cfg.get("num_layers", 3))
        mlp_hidden = int(net_cfg.get("mlp_hidden", 128))

        self.param_net = EditParamNet(
            in_channels=in_channels,
            out_dim=self.param_specs.total_dim,
            base_channels=base_channels,
            num_layers=num_layers,
            mlp_hidden=mlp_hidden,
        )

        spec_bias = self.param_specs.identity_raw_vector()
        if spec_bias:
            with torch.no_grad():
                nn.init.zeros_(self.param_net.mlp[-1].weight)
                self.param_net.mlp[-1].bias.copy_(
                    torch.tensor(spec_bias, dtype=self.param_net.mlp[-1].bias.dtype)
                )

        blur_spec = self.param_specs.get_spec("blur_sigma")
        blur_sigma_max = float(blur_spec.max_val) if blur_spec is not None else 0.0
        self.edit_ops = EditOps(config.get("ops", {}), blur_sigma_max=blur_sigma_max)

    def forward(
        self,
        x: torch.Tensor,
        op_flags: Optional[Dict[str, float]] = None,
        noise: Optional[torch.Tensor] = None,
    ):
        if self.param_specs.total_dim <= 0:
            return x, {}, {}
        raw = self.param_net(x)
        theta = self.param_specs.raw_to_theta(raw)
        y_edit = self.edit_ops.apply(x, theta, op_flags=op_flags, noise=noise)
        return y_edit, theta, raw
