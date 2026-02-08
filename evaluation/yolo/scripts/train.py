#!/usr/bin/env python3
"""Train a YOLOv8 segmentation model with Ultralytics."""
import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Deque

from collections import deque

import yaml
from ultralytics import YOLO


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_imgsz(train_cfg: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> Any:
    if train_cfg.get("imgsz") is not None:
        return train_cfg["imgsz"]
    return dataset_cfg.get("img_size")


def create_limited_dataset(data_yaml_path: Path, limit: int, output_dir: Path) -> Path:
    """
    Create a dataset config with a limited number of training samples.
    Keeps only the first N training samples; validation/test remain unchanged.
    """
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    
    base_path = data_yaml_path.parent
    train_img_dir = base_path / data_config.get("train", "images/train")
    train_lbl_dir = base_path / "labels" / "train"
    
    train_images = sorted(train_img_dir.glob("*"))
    if len(train_images) == 0:
        raise FileNotFoundError(f"Training image directory is empty: {train_img_dir}")
    
    limited_images = train_images[:limit]
    print(f"[limit_samples] Original training samples: {len(train_images)}, limited to: {len(limited_images)}")
    
    temp_data_root = output_dir / "limited_data"
    temp_train_img = temp_data_root / "images" / "train"
    temp_train_lbl = temp_data_root / "labels" / "train"
    temp_train_img.mkdir(parents=True, exist_ok=True)
    temp_train_lbl.mkdir(parents=True, exist_ok=True)
    
    for img_path in limited_images:
        dst_img = temp_train_img / img_path.name
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)
        
        lbl_name = img_path.stem + ".txt"
        lbl_path = train_lbl_dir / lbl_name
        dst_lbl = temp_train_lbl / lbl_name
        if lbl_path.exists() and not dst_lbl.exists():
            try:
                dst_lbl.symlink_to(lbl_path.resolve())
            except OSError:
                shutil.copy2(lbl_path, dst_lbl)
    
    for split in ["val", "test"]:
        for subdir in ["images", "labels"]:
            src_dir = base_path / subdir / split
            dst_dir = temp_data_root / subdir / split
            if src_dir.exists() and not dst_dir.exists():
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst_dir.symlink_to(src_dir.resolve())
                except OSError:
                    shutil.copytree(src_dir, dst_dir)
    
    new_data_config = data_config.copy()
    new_data_config["path"] = str(temp_data_root.resolve())
    new_data_config["train"] = "images/train"
    new_data_config["val"] = "images/val"
    new_data_config["test"] = "images/test"
    
    temp_yaml = temp_data_root / "data.yaml"
    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(new_data_config, f, allow_unicode=True)
    
    return temp_yaml


def merge_train_cfg(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base) if base else {}
    if overrides:
        merged.update(overrides)
    return merged


def train_variant(cfg: Dict[str, Any], variant: str, train_cfg: Dict[str, Any], *, weights_override: Optional[Path] = None, save_name: str = "train", save_dir_name: Optional[str] = None) -> Path:
    dataset_cfg = cfg["dataset"]
    output_cfg = cfg["output"]
    yolo_data = Path(output_cfg["yolo_data_root"]) / variant / "data.yaml"
    if not yolo_data.exists():
        raise FileNotFoundError(f"Converted data.yaml not found: {yolo_data}. Run prepare_bop_to_yolo.py first.")

    limit_samples = train_cfg.get("limit_samples")
    if limit_samples and limit_samples > 0:
        print(f"[train_variant] Limiting training samples to: {limit_samples}")
        save_dir = save_dir_name if save_dir_name else variant
        temp_output = Path(output_cfg["runs_root"]) / save_dir
        temp_output.mkdir(parents=True, exist_ok=True)
        yolo_data = create_limited_dataset(yolo_data, limit_samples, temp_output)
        print(f"[train_variant] Using limited dataset config: {yolo_data}")

    imgsz = resolve_imgsz(train_cfg, dataset_cfg) or 640
    save_dir = save_dir_name if save_dir_name else variant
    project_dir = Path(output_cfg["runs_root"]) / save_dir

    if weights_override is not None:
        model_path = Path(weights_override)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights do not exist: {model_path}")
        model_path_str = str(model_path)
    else:
        model_path_str = str(train_cfg["model"])

    model = YOLO(model_path_str)

    metric_key = None
    early_cfg = train_cfg.get("early_stop")
    if early_cfg:
        metric_key = early_cfg.get("metric", "metrics/mAP50(M)")
    early_cb = build_plateau_callback(early_cfg, metric_key=metric_key or "metrics/mAP50(M)")
    if early_cb:
        model.add_callback("on_fit_epoch_end", early_cb)

    results = model.train(
        data=str(yolo_data),
        project=str(project_dir),
        name=save_name,
        epochs=train_cfg.get("epochs", 50),
        batch=train_cfg.get("batch", 16),
        imgsz=imgsz,
        workers=train_cfg.get("workers", 8),
        device=train_cfg.get("device", 0),
        patience=train_cfg.get("patience", 20),
        lr0=train_cfg.get("lr0", 0.01),
        lrf=train_cfg.get("lrf", 0.01),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
        exist_ok=True,
    )
    print(f"[dataset: {variant}, save_dir: {save_dir}] training finished, results saved to: {results.save_dir}")
    return Path(results.save_dir) / "weights" / "best.pt"


def run_staged_training(cfg: Dict[str, Any]) -> None:
    train_cfg = cfg.get("train", {})
    staged_cfg = train_cfg.get("staged")
    if not staged_cfg:
        raise ValueError("staged mode requires train.staged in config")

    variants_cfg = cfg["dataset"]["variants"]
    pretrain_variant = staged_cfg.get("pretrain_variant")
    finetune_variant = staged_cfg.get("finetune_variant")
    finetune_from_cfg = staged_cfg.get("finetune_from")
    finetune_save_name = staged_cfg.get("finetune_save_name")  # Optional custom directory name for finetune outputs

    pretrain_best = None
    finetune_from: Path

    if finetune_from_cfg:
        finetune_from = Path(finetune_from_cfg)
        if not finetune_from.exists():
            print(f"[staged] finetune_from does not exist; fallback to pretraining: {finetune_from}")
            finetune_from_cfg = None  # fallback to normal flow
    if not finetune_from_cfg:
        if not pretrain_variant:
            raise ValueError("Missing pretrain variant: train.staged.pretrain_variant")
        if pretrain_variant not in variants_cfg:
            raise ValueError(f"Unknown pretrain variant: {pretrain_variant}")
        print(f"[staged] Pretrain variant: {pretrain_variant}")
        pretrain_best = train_variant(cfg, pretrain_variant, train_cfg)
        finetune_from = pretrain_best

    if not finetune_variant:
        print("[staged] No finetune_variant specified; skipping finetuning")
        return
    if finetune_variant not in variants_cfg:
        raise ValueError(f"Unknown finetune variant: {finetune_variant}")

    if finetune_from_cfg and finetune_from.exists():
        print(f"[staged] Using external pretrain weights: {finetune_from}")
    elif pretrain_best is None or not finetune_from.exists():
        raise FileNotFoundError(f"Finetune start weights not found: {finetune_from}")

    finetune_cfg = merge_train_cfg(train_cfg, staged_cfg.get("finetune_overrides"))
    actual_save_name = finetune_save_name if finetune_save_name else finetune_variant
    print(f"[staged] Finetune dataset: {finetune_variant}, loading weights: {finetune_from}, saving to: {actual_save_name}")
    train_variant(cfg, finetune_variant, finetune_cfg, weights_override=finetune_from, save_dir_name=actual_save_name)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 segmentation by variant")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--variant", default="real", help="Variant to train (single-stage mode)")
    parser.add_argument("--staged", action="store_true", help="Enable staged training (pretrain + optional finetune), using train.staged")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    variants_cfg = cfg["dataset"]["variants"]

    if args.staged:
        run_staged_training(cfg)
        return

    if args.variant not in variants_cfg:
        raise ValueError(f"Unknown variant: {args.variant}")
    base_train_cfg = cfg.get("train", {})
    train_variant(cfg, args.variant, base_train_cfg)


def build_plateau_callback(early_cfg: Optional[Dict[str, Any]], metric_key: str) -> Optional[Callable]:
    """
    Monitor a metric in results.csv (default: metrics/seg/miou).
    Use relative gain between two windows (prev vs recent) for early stop.
    """
    if not early_cfg or not early_cfg.get("enabled", False):
        return None
    window = int(early_cfg.get("window", 5))
    min_delta = float(early_cfg.get("min_delta", 0.001))
    min_epochs = int(early_cfg.get("min_epochs", window))
    history: Deque[float] = deque(maxlen=window * 2)
    header_idx = {"idx": None}

    def _cb(trainer):
        results_path = Path(trainer.save_dir) / "results.csv"
        if not results_path.exists():
            return
        lines = results_path.read_text().strip().splitlines()
        if len(lines) < 2:
            return
        header = next(csv.reader([lines[0]]))
        if header_idx["idx"] is None:
            try:
                header_idx["idx"] = header.index(metric_key)
            except ValueError:
                print(f"[early-stop] Column not found in results.csv: {metric_key}")
                return
        last_row = next(csv.reader([lines[-1]]))
        try:
            value = float(last_row[header_idx["idx"]])
        except (ValueError, IndexError):
            return
        history.append(value)
        current_epoch = getattr(trainer, "epoch", len(history) - 1) + 1
        if len(history) < 2 * window or current_epoch < min_epochs:
            return
        prev_avg = sum(list(history)[-2 * window : -window]) / float(window)
        recent_avg = sum(list(history)[-window:]) / float(window)
        rel_gain = (recent_avg - prev_avg) / max(abs(prev_avg), 1e-8)
        if rel_gain < min_delta:
            setattr(trainer, "stop_training", True)
            if hasattr(trainer, "stopper"):
                trainer.stopper.stop = True
            print(
                f"[early-stop] Last {window} epochs {metric_key} relative gain {rel_gain:.4f} < {min_delta}, "
                f"prev_avg={prev_avg:.4f}, recent_avg={recent_avg:.4f}, early stopping."
            )

    return _cb


if __name__ == "__main__":
    main()
