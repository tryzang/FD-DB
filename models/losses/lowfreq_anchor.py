from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from ..ops.frequency import LowPassFilter, SigmaScaler


class LowFreqAnchorLoss(nn.Module):
    def __init__(self, cfg: Optional[Dict], freq_cfg: Optional[Dict]):
        super().__init__()
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enable", False))
        self.sigmas: List[float] = [float(s) for s in cfg.get("sigmas", [2.0])]
        weights = cfg.get("weights", None)
        self.weights = self._normalize_weights(weights, self.sigmas)
        self.sigma_scaler = SigmaScaler.from_config(freq_cfg)
        padding = str((freq_cfg or {}).get("padding", "reflect")).lower()
        self.lowpass = LowPassFilter(padding=padding)

    @staticmethod
    def _normalize_weights(
        weights: Optional[Iterable[float]], sigmas: List[float]
    ) -> List[float]:
        if not weights:
            weights = [1.0 for _ in sigmas]
        weights = [float(w) for w in weights]
        if len(weights) != len(sigmas):
            weights = [1.0 for _ in sigmas]
        total = sum(weights)
        if total <= 0:
            return [1.0 / max(len(sigmas), 1) for _ in sigmas]
        return [w / total for w in weights]

    def forward(
        self, y: torch.Tensor, y_edit: torch.Tensor, return_details: bool = False
    ):
        if not self.enabled or y_edit is None:
            zero = y.new_tensor(0.0)
            return (zero, None) if return_details else zero

        y01 = (y + 1.0) * 0.5
        y_edit01 = (y_edit + 1.0) * 0.5

        sigmas = self.sigma_scaler.resolve_many(y01, self.sigmas)
        total = 0.0
        details = {}
        for idx, sigma in enumerate(sigmas):
            lp_y = self.lowpass(y01, sigma)
            lp_edit = self.lowpass(y_edit01, sigma)
            diff = (lp_y - lp_edit).abs().mean()
            weight = self.weights[idx]
            total = total + weight * diff
            details[f"scale_{idx}"] = float(diff.detach().item())

        details["total"] = float(total.detach().item())
        return (total, details) if return_details else total
