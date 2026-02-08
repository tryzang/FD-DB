#!/usr/bin/env python3
"""
Convert YCB-V in BOP format to YOLOv8 segmentation format (images/labels/data.yaml).
Features:
- Supports multiple variants (real/synthetic/edit/free/edit_free) defined in config.
- Builds segmentation polygons from visible masks (mask_visib); can switch to mask.
- Optional resize (img_size=None keeps source resolution; otherwise [H, W]).
- Splits validation from train with a deterministic ratio.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def load_config(cfg_path: Path) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def try_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def md5_ratio_token(scene: str, im_id: int, seed: int) -> float:
    key = f"{scene}_{im_id}_{seed}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF


def load_image(path: Path, target_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if target_hw:
        h, w = target_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def load_mask(path: Path, target_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if target_hw:
        h, w = target_hw
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def mask_to_polygons(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        poly = c.reshape(-1, 2)
        polys.append(poly)
    return polys


def normalize_polygon(poly: np.ndarray, width: int, height: int) -> List[float]:
    poly = poly.astype(np.float32)
    poly[:, 0] /= float(width)
    poly[:, 1] /= float(height)
    flat = poly.flatten().tolist()
    return [float(f"{p:.6f}") for p in flat]


def scene_iterator(scene_dir: Path, mask_folder: str):
    scene_gt_path = scene_dir / "scene_gt.json"
    if not scene_gt_path.exists():
        raise FileNotFoundError(f"Missing scene_gt.json: {scene_gt_path}")
    with open(scene_gt_path, "r") as f:
        scene_gt = json.load(f)
    for im_id_str, instances in scene_gt.items():
        im_id = int(im_id_str)
        yield im_id, instances, scene_dir / mask_folder


def write_label(label_path: Path, lines: List[str]) -> None:
    ensure_dir(label_path.parent)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def prepare_split_dirs(base_out: Path) -> Dict[str, Dict[str, Path]]:
    splits = {}
    for split in ["train", "val", "test"]:
        img_dir = base_out / "images" / split
        lbl_dir = base_out / "labels" / split
        ensure_dir(img_dir)
        ensure_dir(lbl_dir)
        splits[split] = {"img_dir": img_dir, "lbl_dir": lbl_dir}
    return splits


def find_rgb_path(scene_dir: Path, im_id: int) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        cand = scene_dir / "rgb" / f"{im_id:06d}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(f"RGB image not found (png/jpg supported):{scene_dir}/rgb/{im_id:06d}.*")


def convert_variant(cfg: Dict, variant: str) -> None:
    dataset_cfg = cfg["dataset"]
    variant_cfg = dataset_cfg["variants"][variant]
    root = Path(dataset_cfg["root"])
    train_folder = root / variant_cfg["train_folder"]
    test_folder = root / variant_cfg["test_folder"]
    if not train_folder.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_folder}")
    if not test_folder.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_folder}")

    img_size_cfg = dataset_cfg.get("img_size")
    target_hw = None
    if img_size_cfg is not None:
        if not (isinstance(img_size_cfg, (list, tuple)) and len(img_size_cfg) == 2):
            raise ValueError("img_size must be [H, W] or null")
        target_hw = (int(img_size_cfg[0]), int(img_size_cfg[1]))

    class_items = sorted(((int(k), v) for k, v in dataset_cfg["classes"].items()), key=lambda kv: kv[0])
    class_names = [name for _, name in class_items]
    obj_to_yolo = {obj_id: idx for idx, (obj_id, _) in enumerate(class_items)}

    output_root = Path(cfg["output"]["yolo_data_root"]) / variant
    splits = prepare_split_dirs(output_root)

    val_ratio = float(dataset_cfg.get("val_split_ratio", 0.0))
    val_seed = int(dataset_cfg.get("val_seed", 0))
    use_visible_mask = bool(dataset_cfg.get("use_visible_mask", True))
    mask_folder = dataset_cfg.get("mask_folder", "mask_visib" if use_visible_mask else "mask")
    resize_tag = f"{target_hw}" if target_hw else "original"

    stats = {"train_images": 0, "val_images": 0, "test_images": 0, "instances": 0, "skipped": 0}
    image_written = {}

    def process_split(split_name: str, folder: Path, enable_val_split: bool) -> None:
        nonlocal stats
        scene_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])
        iterator = []
        for scene_dir in scene_dirs:
            iterator.append((scene_dir.name, scene_dir))
        pbar = tqdm(iterator, desc=f"[{variant}] {split_name} ({resize_tag})", unit="scene")
        for scene_name, scene_dir in pbar:
            for im_id, instances, mask_dir in scene_iterator(scene_dir, mask_folder):
                label_lines: List[str] = []
                split_target = split_name
                if enable_val_split and val_ratio > 0:
                    ratio_token = md5_ratio_token(scene_name, im_id, val_seed)
                    split_target = "val" if ratio_token < val_ratio else "train"

                mask_dir_actual = mask_dir
                rgb_path = find_rgb_path(scene_dir, im_id)

                if split_target not in splits:
                    continue

                if enable_val_split and val_ratio <= 0 and split_name == "train":
                    split_target = "train"

                dst_img_dir = splits[split_target]["img_dir"]
                dst_lbl_dir = splits[split_target]["lbl_dir"]
                out_stem = f"{scene_name}_{im_id:06d}"
                out_name = out_stem + rgb_path.suffix.lower()
                dst_img_path = dst_img_dir / out_name
                dst_lbl_path = dst_lbl_dir / f"{out_stem}.txt"

                key = (split_target, scene_name, im_id)
                if key not in image_written:
                    if target_hw:
                        img = load_image(rgb_path, target_hw)
                        ensure_dir(dst_img_path.parent)
                        cv2.imwrite(str(dst_img_path), img)
                    else:
                        try_symlink(rgb_path, dst_img_path)
                    image_written[key] = dst_img_path
                    stats[f"{split_target}_images"] += 1

                mask_base = f"{im_id:06d}"
                for inst_idx, inst in enumerate(instances):
                    obj_id = int(inst["obj_id"])
                    if obj_id not in obj_to_yolo:
                        stats["skipped"] += 1
                        continue
                    mask_path = mask_dir_actual / f"{mask_base}_{inst_idx:06d}.png"
                    if not mask_path.exists():
                        stats["skipped"] += 1
                        continue
                    mask = load_mask(mask_path, target_hw)
                    polys = mask_to_polygons(mask)
                    if not polys:
                        stats["skipped"] += 1
                        continue
                    h, w = mask.shape
                    for poly in polys:
                        norm = normalize_polygon(poly, w, h)
                        if len(norm) < 6:
                            stats["skipped"] += 1
                            continue
                        line = " ".join([str(obj_to_yolo[obj_id])] + [f"{v:.6f}" for v in norm])
                        label_lines.append(line)
                        stats["instances"] += 1

                if label_lines:
                    write_label(dst_lbl_path, label_lines)
                else:
                    write_label(dst_lbl_path, [])

    process_split("train", train_folder, enable_val_split=True)
    process_split("test", test_folder, enable_val_split=False)

    val_dir = splits["val"]["img_dir"]
    if val_ratio <= 0:
        val_dir = splits["train"]["img_dir"]

    data_yaml = {
        "path": str(output_root.resolve()),
        "train": str((splits["train"]["img_dir"]).resolve()),
        "val": str(val_dir.resolve()),
        "test": str((splits["test"]["img_dir"]).resolve()),
        "nc": len(class_names),
        "names": class_names,
    }
    with open(output_root / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"[{variant}] conversion completed -> {output_root}")
    print(f"  stats: train={stats['train_images']} val~{stats['val_images']} test={stats['test_images']} instances={stats['instances']} skipped={stats['skipped']}")


def main():
    parser = argparse.ArgumentParser(description="Convert BOP YCB-V to a YOLO segmentation dataset")
    parser.add_argument("--config", required=True, help="Config file path, e.g., configs/ycbv_default.yaml")
    parser.add_argument("--variant", default="all", help="Variant to process, or all")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    variants_cfg = cfg["dataset"]["variants"]
    target_variants = list(variants_cfg.keys()) if args.variant == "all" else [args.variant]
    for v in target_variants:
        if v not in variants_cfg:
            raise ValueError(f"Unknown variant:{v}")
        convert_variant(cfg, v)


if __name__ == "__main__":
    main()
