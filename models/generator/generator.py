from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .branches.free_form import FreeFormGenerator
from .branches.param_edit import EditBranch
from ..ops.frequency import HighPassFilter, LowPassFilter, SigmaScaler


class GeneratorOutput(tuple):
    def __new__(
        cls,
        y: torch.Tensor,
        enc_feats: List[torch.Tensor],
        dec_feats: List[torch.Tensor],
        y_final: Optional[torch.Tensor] = None,
        y_edit: Optional[torch.Tensor] = None,
        r_free: Optional[torch.Tensor] = None,
        theta: Optional[Dict[str, torch.Tensor]] = None,
        gates: Optional[Dict[str, float]] = None,
    ):
        obj = super().__new__(cls, (y, enc_feats, dec_feats))
        obj.y_final = y_final
        obj.y_edit = y_edit
        obj.r_free = r_free
        obj.theta = theta
        obj.gates = gates
        return obj

    @property
    def y(self) -> torch.Tensor:
        return self[0]

    @property
    def enc_feats(self) -> List[torch.Tensor]:
        return self[1]

    @property
    def dec_feats(self) -> List[torch.Tensor]:
        return self[2]

    def _asdict(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "y": self.y,
            "enc_feats": self.enc_feats,
            "dec_feats": self.dec_feats,
            "y_final": self.y_final,
            "y_edit": self.y_edit,
            "r_free": self.r_free,
            "theta": self.theta,
            "gates": self.gates,
        }


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.free_branch = FreeFormGenerator(config)

        param_cfg = config.get("param_edit", {}) or {}
        self.param_edit_enabled = bool(param_cfg.get("enable", False))
        self.edit_branch = None
        if self.param_edit_enabled:
            self.edit_branch = EditBranch(param_cfg, in_channels=self.free_branch.in_channels)

        fusion_cfg = config.get("fusion", {}) or {}
        g_res_cfg = fusion_cfg.get("g_res", fusion_cfg.get("g_res_default", 1.0))
        if isinstance(g_res_cfg, dict):
            self.g_res_default = float(g_res_cfg.get("default", 1.0))
        else:
            self.g_res_default = float(g_res_cfg)
        self.sigma_fuse = fusion_cfg.get("sigma_fuse")
        

        freq_cfg = config.get("frequency", {}) or {}
        self.sigma_scaler = SigmaScaler.from_config(freq_cfg)
        padding = str(freq_cfg.get("padding", "reflect")).lower()
        self._lowpass = LowPassFilter(padding=padding)
        self._highpass = HighPassFilter(self._lowpass)

        self.in_channels = self.free_branch.in_channels
        self.base_channels = self.free_branch.base_channels
        self.num_blocks = self.free_branch.num_blocks
        self.use_tanh = self.free_branch.use_tanh
        self.use_checkpoint = self.free_branch.use_checkpoint
        self.param_specs = getattr(self.edit_branch, "param_specs", None)
        self.noise_scale = 0.0

    @property
    def enco_channels(self):
        return self.free_branch.enco_channels

    def _resolve_g_res(self, gates: Optional[Dict[str, float]]) -> torch.Tensor:
        g_res = self.g_res_default
        if gates is not None and gates.get("g_res") is not None:
            g_res = float(gates["g_res"])
        g_res = max(0.0, min(1.0, g_res))
        return torch.tensor(g_res)

    def _resolve_sigma_fuse(self, x: torch.Tensor) -> float:
        sigma_override = None
        if self.sigma_fuse is not None:
            sigma_override = float(self.sigma_fuse)
        return float(self.sigma_scaler.resolve(x, sigma=sigma_override))

    def forward(
        self,
        x: torch.Tensor,
        gates: Optional[Dict[str, float]] = None,
        op_flags: Optional[Dict[str, float]] = None,
        edit_noise: Optional[torch.Tensor] = None,
        spatial_weight: Optional[torch.Tensor] = None,
        use_edit_as_source: bool = False,
    ) -> GeneratorOutput:
        if not self.param_edit_enabled or self.edit_branch is None:
            y_free, enc_feats, dec_feats = self.free_branch(
                x, out_mode="full"
            )
            return GeneratorOutput(
                y=y_free, enc_feats=enc_feats, dec_feats=dec_feats, y_final=y_free
            )

        y_edit, theta, _ = self.edit_branch(x, op_flags=op_flags, noise=edit_noise)
        g_res = self._resolve_g_res(gates)
        free_input = y_edit.detach() if use_edit_as_source else x
        r_free, enc_feats, dec_feats = self.free_branch(
            free_input, out_mode="residual"
        )

        r_train = r_free
        weight = None
        if spatial_weight is not None:
            weight = spatial_weight
            if weight.dim() == 3:
                weight = weight.unsqueeze(1)
            if weight.shape[2:] != r_train.shape[2:]:
                weight = torch.nn.functional.interpolate(
                    weight, size=r_train.shape[2:], mode="bilinear", align_corners=False
                )
            r_train = r_train * weight

        g_res = g_res.to(device=y_edit.device, dtype=y_edit.dtype)
        y_train = (y_edit + g_res * r_train).clamp(-1.0, 1.0)

        sigma_fuse = self._resolve_sigma_fuse(r_free)
        if sigma_fuse > 0:
            r_hp = self._highpass(r_free, sigma_fuse)
            if weight is not None:
                r_hp = r_hp * weight
        else:
            r_hp = r_train
        y_final = (y_edit + g_res * r_hp).clamp(-1.0, 1.0)
        if sigma_fuse <= 0:
            y_final = y_train

        gates_out = {"g_res": float(g_res.item())}
        if gates is not None:
            gates_out.update({k: float(v) for k, v in gates.items() if k != "g_res"})
        return GeneratorOutput(
            y=y_train,
            enc_feats=enc_feats,
            dec_feats=dec_feats,
            y_final=y_final,
            y_edit=y_edit,
            r_free=r_free,
            theta=theta,
            gates=gates_out,
        )
