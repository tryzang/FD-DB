from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass
class LoggerConfig:
    enabled: bool = False
    log_dir: str = "runs"
    run_name: str = "default"
    purge_step: Optional[int] = 0
    level: str = "MIN"  # MIN | DEBUG
    scalars_every_steps: int = 50
    images_every_steps: int = 1000
    debug_every_steps: int = 200
    flush_every_steps: int = 200
    image_max_side: int = 512
    num_fixed_samples: int = 2
    seed: int = 42
    log_hparams: bool = False
    log_meta: bool = False
    log_losses: bool = True
    log_images_panel: bool = True
    log_edit_images: bool = True
    log_edit_panel: bool = True
    log_schedule: bool = True
    log_d_hist: bool = False
    log_theta_stats: bool = False
    log_theta_hist: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "LoggerConfig":
        cfg = cfg or {}
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        init_kwargs = {}
        for k, v in cfg.items():
            if k in fields:
                init_kwargs[k] = v
        return cls(**init_kwargs)

    @property
    def is_debug(self) -> bool:
        return str(self.level).upper() == "DEBUG"
