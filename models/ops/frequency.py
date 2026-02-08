import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F


_KERNEL_CACHE: "OrderedDict[Tuple[float, torch.device, torch.dtype], torch.Tensor]" = OrderedDict()
_KERNEL_CACHE_MAX = 32


def _kernel_size_from_sigma(sigma: float) -> int:
    return int(2 * math.ceil(3 * float(sigma)) + 1)


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    sigma = max(float(sigma), 1e-6)
    ks = _kernel_size_from_sigma(sigma)
    center = (ks - 1) * 0.5
    coords = torch.arange(ks, device=device, dtype=dtype) - center
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel


def _get_cached_kernel_1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (float(sigma), device, dtype)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        _KERNEL_CACHE.move_to_end(key)
        return cached
    with torch.no_grad():
        kernel = _gaussian_kernel_1d(sigma, device=device, dtype=dtype)
    _KERNEL_CACHE[key] = kernel
    if len(_KERNEL_CACHE) > _KERNEL_CACHE_MAX:
        _KERNEL_CACHE.popitem(last=False)
    return kernel


def gaussian_blur(
    x: torch.Tensor, sigma: float, padding: str = "reflect"
) -> torch.Tensor:
    if sigma <= 0:
        return x
    b, c, h, w = x.shape
    if h <= 1 or w <= 1:
        return x

    dtype = x.dtype if x.dtype != torch.float16 else torch.float32
    kernel = _get_cached_kernel_1d(sigma, device=x.device, dtype=dtype)
    kernel = kernel.to(dtype=x.dtype)

    ks = kernel.numel()
    pad = ks // 2
    if pad <= 0 or pad >= min(h, w):
        return x

    weight_h = kernel.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
    weight_v = kernel.view(1, 1, ks, 1).repeat(c, 1, 1, 1)

    x_pad = F.pad(x, (pad, pad, 0, 0), mode=padding)
    out = F.conv2d(x_pad, weight_h, groups=c)
    out_pad = F.pad(out, (0, 0, pad, pad), mode=padding)
    out = F.conv2d(out_pad, weight_v, groups=c)
    return out


@dataclass
class SigmaScaler:
    ref_size: int = 256
    use_scale: bool = True
    sigma_base: float = 2.0
    sigma: float = 2.0

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "SigmaScaler":
        cfg = cfg or {}
        return cls(
            ref_size=int(cfg.get("ref_size", 256)),
            use_scale=bool(cfg.get("use_scale", True)),
            sigma_base=float(cfg.get("sigma_base", 2.0)),
            sigma=float(cfg.get("sigma", 2.0)),
        )

    def resolve(self, x: torch.Tensor, sigma: Optional[float] = None) -> float:
        if sigma is None:
            sigma = self.sigma_base if self.use_scale else self.sigma
        sigma = float(sigma)
        if not self.use_scale:
            return sigma
        h, w = int(x.shape[-2]), int(x.shape[-1])
        base = float(min(h, w)) / max(float(self.ref_size), 1.0)
        return sigma * base

    def resolve_many(self, x: torch.Tensor, sigmas: Iterable[float]) -> List[float]:
        return [self.resolve(x, sigma=s) for s in sigmas]


class LowPassFilter:
    def __init__(self, padding: str = "reflect"):
        self.padding = padding

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return gaussian_blur(x, sigma, padding=self.padding)


class HighPassFilter:
    def __init__(self, lowpass: LowPassFilter):
        self.lowpass = lowpass

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        return x - self.lowpass(x, sigma)
