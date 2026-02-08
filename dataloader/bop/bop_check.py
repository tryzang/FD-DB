import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.bop.translation import get_bop_translation_dataloaders  # noqa: E402
from logger import InfoLogger  # noqa: E402


def _scene_count(adapter) -> int:
    return len({r.scene_id for r in adapter.records_all})


def main():
    logger = InfoLogger(name="BOP_Check")
    parser = argparse.ArgumentParser(description="Quick BOP adapter smoke test.")
    parser.add_argument("--config", type=str, default=os.path.join("configs", "default.yaml"))
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    if data_cfg.get("type") != "bop":
        raise ValueError("Config data.type must be 'bop' for bop_check.")

    train_loader, val_loader = get_bop_translation_dataloaders(data_cfg)
    train_dataset = train_loader.dataset

    syn_adapter = train_dataset.syn_adapter
    real_adapter = train_dataset.real_adapter

    logger.info(f"[syn] split={syn_adapter.split_dir} total={len(syn_adapter.records_all)} scenes={_scene_count(syn_adapter)}")
    logger.info(
        f"[syn] train={len(syn_adapter.records_train)} val={len(syn_adapter.records_val)} "
        f"train_scenes={len({r.scene_id for r in syn_adapter.records_train})} "
        f"val_scenes={len({r.scene_id for r in syn_adapter.records_val})}"
    )
    logger.info(f"[real] split={real_adapter.split_dir} total={len(real_adapter.records_all)} scenes={_scene_count(real_adapter)}")
    logger.info(
        f"[real] train={len(real_adapter.records_train)} val={len(real_adapter.records_val)} "
        f"train_scenes={len({r.scene_id for r in real_adapter.records_train})} "
        f"val_scenes={len({r.scene_id for r in real_adapter.records_val})}"
    )

    sample_count = min(10, len(train_dataset))
    for i in range(sample_count):
        _ = train_dataset[i]
    logger.info(f"Sampled {sample_count} train items successfully.")

    preview_dir = Path("artifacts/preview")
    preview_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Preview directory prepared at: {preview_dir.resolve()} (write images here if needed)")


if __name__ == "__main__":
    main()
