from typing import Dict, Optional, Tuple

import torch


_PARAM_TO_FLAG = {
    "wb_gain": "wb",
    "exposure": "exposure",
    "contrast": "contrast",
    "saturation": "saturation",
    "blur_sigma": "blur",
    "grain_amp": "grain",
    "grain_size": "grain",
}


def _param_flag(name: str, op_flags: Optional[Dict[str, float]]) -> float:
    if op_flags is None:
        return 1.0
    key = _PARAM_TO_FLAG.get(name)
    if key is None:
        return 1.0
    return float(op_flags.get(key, 1.0))


def compute_edit_regularizers(
    theta: Dict[str, torch.Tensor],
    param_specs,
    weights: Optional[Dict[str, float]] = None,
    op_flags: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weights = weights or {}
    losses: Dict[str, torch.Tensor] = {}
    param_terms = []

    for spec in getattr(param_specs, "specs", []):
        name = spec.name
        if name not in theta:
            continue
        flag = _param_flag(name, op_flags)
        if flag <= 0:
            continue
        target = torch.as_tensor(spec.identity, device=theta[name].device, dtype=theta[name].dtype)
        diff = (theta[name] - target).abs().mean()
        losses[f"param_{name}"] = diff
        param_terms.append(diff)

    if param_terms:
        losses["param"] = sum(param_terms) / max(len(param_terms), 1)

    if "blur_sigma" in theta:
        losses["blur"] = theta["blur_sigma"].abs().mean()
    
    # Grain regularization: include both grain_amp and grain_size
    grain_terms = []
    if "grain_amp" in theta:
        grain_terms.append(theta["grain_amp"].abs().mean())
    if "grain_size" in theta:
        grain_terms.append(theta["grain_size"].abs().mean())
    if grain_terms:
        losses["grain"] = sum(grain_terms) / len(grain_terms)

    total = None
    if losses:
        total = 0.0
        for key in ("param", "blur", "grain"):
            if key not in losses:
                continue
            weight = float(weights.get(key, 1.0))
            total = total + weight * losses[key]
        losses["total"] = total
    return total, losses


def compute_theta_stats(theta: Dict[str, torch.Tensor]) -> Dict[str, float]:
    stats = {}
    for name, value in theta.items():
        if value is None:
            continue
        stats[f"theta/{name}_mean"] = float(value.mean())
        stats[f"theta/{name}_std"] = float(value.std())
    return stats


def compute_theta_saturation(
    theta: Dict[str, torch.Tensor],
    param_specs,
    eps_ratio: float = 0.05,
) -> Optional[float]:
    ratios = []
    for name in ("blur_sigma", "grain_amp"):
        if name not in theta:
            continue
        spec = param_specs.get_spec(name) if param_specs is not None else None
        if spec is None:
            continue
        value = theta[name]
        span = max(spec.max_val - spec.min_val, 1e-6)
        eps = eps_ratio * span
        near = (value <= spec.min_val + eps) | (value >= spec.max_val - eps)
        ratios.append(near.float().mean())
    if not ratios:
        return None
    return float(torch.stack(ratios).mean().item())
