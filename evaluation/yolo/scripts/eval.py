#!/usr/bin/env python3
"""Evaluate YOLOv8 segmentation on a split and report mIoU and Dice (F1)."""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO
from tqdm import tqdm


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def polygon_to_mask(poly: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)
    if pts.shape[0] >= 3:
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_gt(label_path: Path, shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
    h, w = shape
    per_class: Dict[int, np.ndarray] = {}
    if not label_path.exists():
        return per_class
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        if len(coords) < 6 or len(coords) % 2 != 0:
            continue
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        mask = polygon_to_mask(pts, (h, w))
        if cls_id not in per_class:
            per_class[cls_id] = mask
        else:
            per_class[cls_id] = np.logical_or(per_class[cls_id], mask).astype(np.uint8)
    return per_class


def pred_to_masks(result, shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
    h, w = shape
    per_class: Dict[int, np.ndarray] = {}
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return per_class
    polys = masks.xy
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    for cls_id, poly in zip(cls_ids, polys):
        mask = polygon_to_mask(poly, (h, w))
        if cls_id not in per_class:
            per_class[cls_id] = mask
        else:
            per_class[cls_id] = np.logical_or(per_class[cls_id], mask).astype(np.uint8)
    return per_class


def accumulate(metrics: Dict[int, Dict[str, float]], gt_masks: Dict[int, np.ndarray], pred_masks: Dict[int, np.ndarray]) -> None:
    class_ids = set(gt_masks.keys()) | set(pred_masks.keys())
    for cid in class_ids:
        gt = gt_masks.get(cid)
        pred = pred_masks.get(cid)
        if gt is None:
            if pred is None:
                continue  # Skip if neither side exists
            gt = np.zeros_like(next(iter(pred_masks.values())))
        if pred is None:
            pred = np.zeros_like(gt)
        if gt.shape != pred.shape:
            h, w = max(gt.shape[0], pred.shape[0]), max(gt.shape[1], pred.shape[1])
            if gt.shape != (h, w):
                gt_resized = np.zeros((h, w), dtype=gt.dtype)
                gt_resized[:gt.shape[0], :gt.shape[1]] = gt
                gt = gt_resized
            if pred.shape != (h, w):
                pred_resized = np.zeros((h, w), dtype=pred.dtype)
                pred_resized[:pred.shape[0], :pred.shape[1]] = pred
                pred = pred_resized
        inter = float(np.logical_and(gt, pred).sum())
        union = float(np.logical_or(gt, pred).sum())
        gt_sum = float(gt.sum())
        pred_sum = float(pred.sum())
        if cid not in metrics:
            metrics[cid] = {"inter": 0.0, "union": 0.0, "gt": 0.0, "pred": 0.0}
        metrics[cid]["inter"] += inter
        metrics[cid]["union"] += union
        metrics[cid]["gt"] += gt_sum
        metrics[cid]["pred"] += pred_sum


def compute_scores(metrics: Dict[int, Dict[str, float]]) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
    per_class_scores: Dict[int, Dict[str, float]] = {}
    ious, dices = [], []
    for cid, m in metrics.items():
        inter = m["inter"]
        union = m["union"]
        gt_sum = m["gt"]
        pred_sum = m["pred"]
        iou = inter / union if union > 0 else 0.0
        dice = (2 * inter) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
        per_class_scores[cid] = {"iou": iou, "dice": dice, "inter": inter, "union": union, "gt": gt_sum, "pred": pred_sum}
        if union > 0:
            ious.append(iou)
        if (gt_sum + pred_sum) > 0:
            dices.append(dice)
    miou = float(np.mean(ious)) if ious else 0.0
    mdice = float(np.mean(dices)) if dices else 0.0
    return miou, mdice, per_class_scores


def eval_variant(cfg: Dict[str, Any], variant: str) -> None:
    dataset_cfg = cfg["dataset"]
    eval_cfg = cfg.get("eval", {})
    output_cfg = cfg["output"]
    split = eval_cfg.get("split", "test")
    yolo_data_root = Path(output_cfg["yolo_data_root"]) / variant
    images_dir = yolo_data_root / "images" / split
    labels_dir = yolo_data_root / "labels" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}. Convert data first.")
    classes = sorted(((int(k), v) for k, v in dataset_cfg["classes"].items()), key=lambda kv: kv[0])
    class_names = [name for _, name in classes]
    weights = Path(output_cfg["runs_root"]) / variant / "train" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found {weights}. Train first or provide a different model.")

    model = YOLO(str(weights))
    imgsz = eval_cfg.get("imgsz") or cfg.get("train", {}).get("imgsz") or dataset_cfg.get("img_size")

    metrics: Dict[int, Dict[str, float]] = {}
    image_paths = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
        image_paths.extend(images_dir.glob(ext))
    image_paths = sorted(image_paths)
    if not image_paths:
        raise RuntimeError(f"{images_dir} has no usable images (png/jpg supported)")

    limit = eval_cfg.get("limit")
    if limit is not None:
        image_paths = image_paths[: int(limit)]

    batch_size = eval_cfg.get("batch", 1)
    chunk_size = eval_cfg.get("chunk_size")
    chunk_size = int(chunk_size) if chunk_size else len(image_paths)
    use_half = bool(eval_cfg.get("half", True))
    empty_cache = bool(eval_cfg.get("empty_cache", True))

    global_pbar = tqdm(total=len(image_paths), desc=f"[{variant}] eval@{split} total", dynamic_ncols=True)

    for start in range(0, len(image_paths), chunk_size):
        end = min(len(image_paths), start + chunk_size)
        batch_paths = image_paths[start:end]
        predictions = model.predict(
            source=batch_paths,
            imgsz=imgsz,
            batch=batch_size,
            conf=eval_cfg.get("conf", 0.25),
            iou=eval_cfg.get("iou", 0.5),
            device=eval_cfg.get("device", 0),
            max_det=eval_cfg.get("max_det", 300),
            stream=True,
            verbose=False,
            half=use_half,
        )

        pbar = tqdm(
            predictions,
            total=len(batch_paths),
            desc=f"[{variant}] eval@{split} [{start}:{end}]",
            leave=False,
            dynamic_ncols=True,
        )
        for res in pbar:
            global_pbar.update(1)
            img_path = Path(res.path)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_label = labels_dir / f"{img_path.stem}.txt"
            gt = load_gt(gt_label, (h, w))
            pred_masks = pred_to_masks(res, (h, w))
            accumulate(metrics, gt, pred_masks)
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_pbar.close()

    miou, mdice, per_class_scores = compute_scores(metrics)
    per_class_serializable = {
        int(cid): {
            "iou": float(scores.get("iou", 0.0)),
            "dice": float(scores.get("dice", 0.0)),
            "inter": float(scores.get("inter", 0.0)),
            "union": float(scores.get("union", 0.0)),
            "gt": float(scores.get("gt", 0.0)),
            "pred": float(scores.get("pred", 0.0)),
        }
        for cid, scores in per_class_scores.items()
    }
    out_dir = Path(output_cfg["runs_root"]) / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": variant,
        "split": split,
        "miou": miou,
        "dice": mdice,
        "per_class": per_class_serializable,
        "num_images": len(image_paths),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    csv_path = out_dir / "per_class_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "iou", "dice", "inter", "union", "gt_pixels", "pred_pixels"])
        for idx, name in enumerate(class_names):
            scores = per_class_serializable.get(idx, {})
            writer.writerow([
                idx,
                name,
                scores.get("iou", 0.0),
                scores.get("dice", 0.0),
                scores.get("inter", 0.0),
                scores.get("union", 0.0),
                scores.get("gt", 0.0),
                scores.get("pred", 0.0),
            ])

    print(f"[{variant}] evaluation complete: split={split} mIoU={miou:.4f} Dice={mdice:.4f} (images={len(image_paths)}, batch={batch_size})")
    print(f"  result files: {out_dir/'metrics.json'}, {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 segmentation (mIoU/Dice)")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--variant", default="real", help="Variant to evaluate")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    variants_cfg = cfg["dataset"]["variants"]
    if args.variant not in variants_cfg:
        raise ValueError(f"Unknown variant: {args.variant}")
    eval_variant(cfg, args.variant)


if __name__ == "__main__":
    main()
