import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5


def _to_11(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def _clamp01(val: float) -> float:
    return float(max(0.0, min(1.0, float(val))))


def apply_white_balance(x: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    return x * gains.unsqueeze(-1).unsqueeze(-1)


def apply_exposure(x: torch.Tensor, ev: torch.Tensor) -> torch.Tensor:
    scale = torch.pow(2.0, ev)
    return x * scale.unsqueeze(-1).unsqueeze(-1)


def apply_contrast(x: torch.Tensor, contrast: torch.Tensor) -> torch.Tensor:
    return (x - 0.5) * contrast.unsqueeze(-1).unsqueeze(-1) + 0.5


def apply_saturation(
    x: torch.Tensor, saturation: torch.Tensor, luma_weights: Sequence[float]
) -> torch.Tensor:
    if x.shape[1] != 3:
        return x
    w = torch.tensor(luma_weights, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    gray = (x * w).sum(dim=1, keepdim=True)
    return gray + saturation.unsqueeze(-1).unsqueeze(-1) * (x - gray)


def _gaussian_blur_separable(
    x: torch.Tensor,
    sigma: torch.Tensor,
    kernel_size: int,
    sigma_eps: float = 1e-3,
) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    sigma_max = float(sigma.max().item()) if sigma.numel() > 0 else 0.0
    if sigma_max < sigma_eps:
        return x

    b, c, h, w = x.shape
    sigma = sigma.view(b, 1).clamp(min=sigma_eps)

    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype)
    coords = coords.view(1, -1) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = kernel / (kernel.sum(dim=1, keepdim=True) + 1e-6)

    weight_h = kernel.view(b, 1, 1, kernel_size).repeat(1, c, 1, 1)
    weight_h = weight_h.view(b * c, 1, 1, kernel_size)

    x_in = x.view(1, b * c, h, w)
    out = F.conv2d(x_in, weight_h, padding=(0, kernel_size // 2), groups=b * c)
    out = out.view(b, c, h, w)

    weight_v = kernel.view(b, 1, kernel_size, 1).repeat(1, c, 1, 1)
    weight_v = weight_v.view(b * c, 1, kernel_size, 1)

    out = F.conv2d(
        out.view(1, b * c, h, w),
        weight_v,
        padding=(kernel_size // 2, 0),
        groups=b * c,
    )
    out = out.view(b, c, h, w)
    return out


def apply_blur(
    x: torch.Tensor,
    sigma: torch.Tensor,
    kernel_size: int,
    sigma_eps: float = 1e-3,
) -> torch.Tensor:
    return _gaussian_blur_separable(x, sigma, kernel_size, sigma_eps=sigma_eps)


def apply_grain(
    x: torch.Tensor,
    amp: torch.Tensor,
    grain_size: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    rgb: bool = False,
    sigma_scale: float = 1.0,
    sigma_eps: float = 1e-3,
) -> torch.Tensor:
    """Add color/grayscale grain with differentiable size via Gaussian smoothing.

    - Generate full-resolution white noise (1c or 3c)
    - Apply per-sample Gaussian blur with sigma = sigma_scale * grain_size
    - Multiply by amplitude and add to x
    """
    b, c, h, w = x.shape
    target_channels = c if rgb else 1

    # Prepare noise at full resolution
    if noise is None:
        noise = torch.randn(b, target_channels, h, w, device=x.device, dtype=x.dtype)
    else:
        if noise.shape[1] != target_channels:
            if noise.shape[1] == 1:
                noise = noise.repeat(1, target_channels, 1, 1)
            else:
                noise = noise[:, :target_channels]

    # Per-sample separable Gaussian blur with continuous sigma
    sigma = (grain_size.view(b, 1).clamp(min=sigma_eps)) * float(sigma_scale)
    sigma_max = float(sigma.max().item()) if sigma.numel() > 0 else 0.0
    kernel_size = int(2 * math.ceil(3 * sigma_max) + 1) if sigma_max > sigma_eps else 1
    smoothed = _gaussian_blur_separable(noise, sigma, kernel_size, sigma_eps=sigma_eps)

    # Match channels to input
    if smoothed.shape[1] != c:
        if smoothed.shape[1] == 1:
            smoothed = smoothed.repeat(1, c, 1, 1)
        else:
            smoothed = smoothed[:, :c]

    return x + smoothed * amp.unsqueeze(-1).unsqueeze(-1)


class EditOps:
    def __init__(self, ops_cfg: Dict, blur_sigma_max: float):
        self.ops_cfg = ops_cfg or {}
        self.enable = {
            "wb": bool(self.ops_cfg.get("wb", {}).get("enable", True)),
            "exposure": bool(self.ops_cfg.get("exposure", {}).get("enable", True)),
            "contrast": bool(self.ops_cfg.get("contrast", {}).get("enable", True)),
            "saturation": bool(self.ops_cfg.get("saturation", {}).get("enable", True)),
            "blur": bool(self.ops_cfg.get("blur", {}).get("enable", True)),
            "grain": bool(self.ops_cfg.get("grain", {}).get("enable", True)),
        }
        self.luma_weights = self.ops_cfg.get("saturation", {}).get(
            "luma_weights", [0.299, 0.587, 0.114]
        )
        self.blur_sigma_eps = float(self.ops_cfg.get("blur", {}).get("sigma_eps", 1e-3))
        self.grain_rgb = bool(self.ops_cfg.get("grain", {}).get("rgb", False))
        self.grain_size_default = float(self.ops_cfg.get("grain", {}).get("size", 1.0))
        self.grain_sigma_scale = float(self.ops_cfg.get("grain", {}).get("sigma_scale", 1.0))
        self.grain_sigma_eps = float(self.ops_cfg.get("grain", {}).get("sigma_eps", 1e-3))
        kernel_cfg = self.ops_cfg.get("blur", {}).get("kernel_size")
        if kernel_cfg is not None:
            kernel_size = int(kernel_cfg)
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.blur_kernel_size = kernel_size
        else:
            sigma_max = float(blur_sigma_max)
            self.blur_kernel_size = (
                int(2 * math.ceil(3 * sigma_max) + 1) if sigma_max > 0 else 1
            )

    def _merge_flags(self, op_flags: Optional[Dict]) -> Dict[str, float]:
        global_flag = None
        if op_flags is not None and "all" in op_flags:
            global_flag = _clamp01(op_flags.get("all", 1.0))
        merged = {}
        for key, enabled in self.enable.items():
            if not enabled:
                merged[key] = 0.0
                continue
            if global_flag is not None:
                flag = global_flag
            else:
                flag = op_flags.get(key, 1.0) if op_flags else 1.0
                flag = _clamp01(flag)
            merged[key] = flag
        return merged

    def apply(
        self,
        x: torch.Tensor,
        theta: Dict[str, torch.Tensor],
        op_flags: Optional[Dict[str, float]] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flags = self._merge_flags(op_flags)
        y = _to_01(x)

        if "wb_gain" in theta and flags["wb"] > 0:
            gains = 1.0 + (theta["wb_gain"] - 1.0) * flags["wb"]
            y = apply_white_balance(y, gains)

        if "exposure" in theta and flags["exposure"] > 0:
            ev = theta["exposure"] * flags["exposure"]
            y = apply_exposure(y, ev)

        if "contrast" in theta and flags["contrast"] > 0:
            contrast = 1.0 + (theta["contrast"] - 1.0) * flags["contrast"]
            y = apply_contrast(y, contrast)

        if "saturation" in theta and flags["saturation"] > 0:
            saturation = 1.0 + (theta["saturation"] - 1.0) * flags["saturation"]
            y = apply_saturation(y, saturation, self.luma_weights)

        if "blur_sigma" in theta and flags["blur"] > 0:
            sigma = theta["blur_sigma"] * flags["blur"]
            y = apply_blur(y, sigma, self.blur_kernel_size, sigma_eps=self.blur_sigma_eps)

        if "grain_amp" in theta and flags["grain"] > 0:
            amp = theta["grain_amp"] * flags["grain"]
            size_tensor = theta.get("grain_size")
            if size_tensor is None:
                size_tensor = amp.new_full(amp.shape, float(self.grain_size_default))
            y = apply_grain(
                y,
                amp,
                grain_size=size_tensor,
                noise=noise,
                rgb=self.grain_rgb,
                sigma_scale=self.grain_sigma_scale,
                sigma_eps=self.grain_sigma_eps,
            )

        y = y.clamp(0.0, 1.0)
        return _to_11(y)
