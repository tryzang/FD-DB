import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


_DEFAULT_PARAM_SPECS = {
    "wb_gain": {
        "dim": 3,
        "map": "log_tanh",
        "min": 0.7,
        "max": 1.4,
        "identity": 1.0,
        "identity_eps": 1e-4,
    },
    "exposure": {
        "dim": 1,
        "map": "tanh",
        "min": -1.0,
        "max": 1.0,
        "identity": 0.0,
        "identity_eps": 1e-4,
    },
    "contrast": {
        "dim": 1,
        "map": "sigmoid",
        "min": 0.7,
        "max": 1.3,
        "identity": 1.0,
        "identity_eps": 1e-4,
    },
    "saturation": {
        "dim": 1,
        "map": "sigmoid",
        "min": 0.0,
        "max": 2.0,
        "identity": 1.0,
        "identity_eps": 1e-4,
    },
    "blur_sigma": {
        "dim": 1,
        "map": "sigmoid",
        "min": 0.0,
        "max": 2.0,
        "identity": 0.0,
        "identity_eps": 1e-4,
    },
    "grain_amp": {
        "dim": 1,
        "map": "sigmoid",
        "min": 0.0,
        "max": 2.0,
        "identity": 0.0,
        "identity_eps": 1e-4,
    },
    "grain_size": {
        "dim": 1,
        "map": "sigmoid",
        "min": 1.0,
        "max": 8.0,
        "identity": 4.0,
        "identity_eps": 1e-4,
    },
}


@dataclass
class ParamSpec:
    name: str
    dim: int
    map_type: str
    min_val: float
    max_val: float
    identity: float
    identity_eps: float = 1e-4

    def raw_to_theta(self, raw: torch.Tensor) -> torch.Tensor:
        if self.map_type == "sigmoid":
            theta = self.min_val + (self.max_val - self.min_val) * torch.sigmoid(raw)
        elif self.map_type == "tanh":
            theta = (
                0.5 * (self.max_val - self.min_val) * torch.tanh(raw)
                + 0.5 * (self.max_val + self.min_val)
            )
        elif self.map_type == "log_tanh":
            min_val = max(self.min_val, 1e-6)
            max_val = max(self.max_val, min_val + 1e-6)
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            mid = 0.5 * (log_max + log_min)
            half = 0.5 * (log_max - log_min)
            theta = torch.exp(mid + half * torch.tanh(raw))
        elif self.map_type == "log_sigmoid":
            min_val = max(self.min_val, 1e-6)
            max_val = max(self.max_val, min_val + 1e-6)
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            theta = torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw))
        elif self.map_type == "identity":
            theta = raw.new_full(raw.shape, float(self.identity))
        else:
            raise ValueError(f"Unknown map type: {self.map_type}")
        return theta

    def identity_raw(self) -> List[float]:
        if self.map_type == "sigmoid":
            denom = max(self.max_val - self.min_val, 1e-6)
            p = (self.identity - self.min_val) / denom
            p = min(max(p, self.identity_eps), 1.0 - self.identity_eps)
            raw = math.log(p / (1.0 - p))
        elif self.map_type == "tanh":
            denom = max(self.max_val - self.min_val, 1e-6)
            v = 2.0 * (self.identity - self.min_val) / denom - 1.0
            v = min(max(v, -1.0 + self.identity_eps), 1.0 - self.identity_eps)
            raw = 0.5 * math.log((1.0 + v) / (1.0 - v))
        elif self.map_type == "log_tanh":
            min_val = max(self.min_val, 1e-6)
            max_val = max(self.max_val, min_val + 1e-6)
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            mid = 0.5 * (log_max + log_min)
            half = max(0.5 * (log_max - log_min), 1e-6)
            log_id = math.log(max(self.identity, 1e-6))
            v = (log_id - mid) / half
            v = min(max(v, -1.0 + self.identity_eps), 1.0 - self.identity_eps)
            raw = 0.5 * math.log((1.0 + v) / (1.0 - v))
        elif self.map_type == "log_sigmoid":
            min_val = max(self.min_val, 1e-6)
            max_val = max(self.max_val, min_val + 1e-6)
            log_min = math.log(min_val)
            log_max = math.log(max_val)
            denom = max(log_max - log_min, 1e-6)
            log_id = math.log(max(self.identity, 1e-6))
            p = (log_id - log_min) / denom
            p = min(max(p, self.identity_eps), 1.0 - self.identity_eps)
            raw = math.log(p / (1.0 - p))
        elif self.map_type == "identity":
            raw = float(self.identity)
        else:
            raise ValueError(f"Unknown map type: {self.map_type}")

        if self.dim <= 1:
            return [float(raw)]
        return [float(raw) for _ in range(self.dim)]


class ParamSpecSet:
    def __init__(self, specs: List[ParamSpec]):
        self.specs = specs
        self._slices = {}
        offset = 0
        for spec in specs:
            self._slices[spec.name] = slice(offset, offset + spec.dim)
            offset += spec.dim
        self.total_dim = offset

    def split_raw(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: raw[:, sl] for name, sl in self._slices.items()}

    def raw_to_theta(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_dict = self.split_raw(raw)
        theta = {}
        for spec in self.specs:
            theta[spec.name] = spec.raw_to_theta(raw_dict[spec.name])
        return theta

    def identity_raw_vector(self) -> List[float]:
        values: List[float] = []
        for spec in self.specs:
            values.extend(spec.identity_raw())
        return values

    def get_spec(self, name: str) -> Optional[ParamSpec]:
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None


def build_param_specs(cfg: Optional[Dict], in_channels: int = 3) -> ParamSpecSet:
    cfg = cfg or {}
    specs: List[ParamSpec] = []
    for name, defaults in _DEFAULT_PARAM_SPECS.items():
        spec_cfg = dict(defaults)
        override = cfg.get(name, {})
        if override is None:
            override = {}
        spec_cfg.update(override)
        dim = int(spec_cfg.get("dim", defaults.get("dim", 1)))
        map_type = str(spec_cfg.get("map", spec_cfg.get("map_type", defaults.get("map")))).lower()
        min_val = float(spec_cfg.get("min", defaults.get("min", 0.0)))
        max_val = float(spec_cfg.get("max", defaults.get("max", 1.0)))
        identity = float(spec_cfg.get("identity", defaults.get("identity", 0.0)))
        identity_eps = float(spec_cfg.get("identity_eps", defaults.get("identity_eps", 1e-4)))

        if in_channels != 3 and name in ("wb_gain", "saturation"):
            continue

        specs.append(
            ParamSpec(
                name=name,
                dim=dim,
                map_type=map_type,
                min_val=min_val,
                max_val=max_val,
                identity=identity,
                identity_eps=identity_eps,
            )
        )
    return ParamSpecSet(specs)
