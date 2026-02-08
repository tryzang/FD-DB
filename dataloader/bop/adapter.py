import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    from bop_toolkit_lib import dataset_params
except ImportError as exc:  # pragma: no cover - installation-time guard
    raise ImportError(
        "bop_toolkit_lib is required for BOPDatasetAdapter. "
        "Install via `pip install -r requirements.txt` (which installs bop_toolkit from GitHub) or install bop_toolkit manually."
    ) from exc

from .indexing import (
    BOPRecord,
    build_records_for_split,
    load_index_json,
    record_from_dict,
    record_to_dict,
    save_index_json,
)
from .splitting import load_split_json, save_split_json, split_by_scene


class _LRUCache:
    """Tiny LRU cache for decoded RGB images."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key not in self._data:
            return None
        val = self._data.pop(key)
        self._data[key] = val
        return val

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_size <= 0:
            return
        if key in self._data:
            self._data.pop(key)
        elif len(self._data) >= self.max_size:
            self._data.popitem(last=False)
        self._data[key] = value


def _parse_split_dir(split_dir: str) -> Tuple[str, Optional[str]]:
    """Guess (split, split_type) from a split directory name."""
    if "_" in split_dir:
        parts = split_dir.split("_", 1)
        return parts[0], parts[1]
    return split_dir, None


def _resolve_split_path(
    datasets_root: str, dataset_name: str, split_dir: str
) -> Tuple[Path, Optional[dict]]:
    """
    Resolve split path using bop_toolkit if possible, fallback to direct join.
    """
    root = Path(datasets_root)
    parsed_split, split_type = _parse_split_dir(split_dir)
    dp_split = None
    try:
        dp_split = dataset_params.get_split_params(root, dataset_name, parsed_split, split_type)
    except Exception:
        dp_split = None

    if dp_split:
        split_path = Path(dp_split["split_path"])
    else:
        split_path = root / dataset_name / split_dir

    # Prefer the actual filesystem path if it exists; otherwise rely on bop template.
    if not split_path.exists():
        fallback = root / dataset_name / split_dir
        if fallback.exists():
            split_path = fallback
        else:
            # Keep the bop_toolkit path (may still be valid even if not present yet)
            pass

    if dp_split:
        dp_split = dict(dp_split)
        dp_split["split_path"] = str(split_path)
    return split_path, dp_split


def _expected_rgb_dirs(dp_split: Optional[dict], scene_id: int) -> List[str]:
    if dp_split is None:
        return ["rgb"]
    tpath_keys = dataset_params.scene_tpaths_keys(
        dp_split.get("eval_modality"), dp_split.get("eval_sensor"), scene_id
    )
    rgb_key = tpath_keys.get("rgb_tpath", "rgb_tpath")
    template = dp_split.get(rgb_key) or dp_split.get("rgb_tpath")
    if template:
        try:
            formatted = template.format(scene_id=scene_id, im_id=0)
            candidate = Path(formatted).parent.name
            return [candidate, "rgb"]
        except Exception:
            return ["rgb"]
    return ["rgb"]


def _expected_extensions(dp_split: Optional[dict]) -> List[str]:
    if dp_split is None:
        return [".png", ".jpg", ".jpeg", ".tif"]
    # Try to infer from rgb_tpath
    template = dp_split.get("rgb_tpath")
    if template:
        ext = Path(template).suffix
        if ext:
            return [ext, ".png", ".jpg", ".jpeg", ".tif"]
    return [".png", ".jpg", ".jpeg", ".tif"]


def _decorate_cache_filename(base: str, domain: str, split_dir: str) -> str:
    if base in ("index.json", "split.json"):
        return f"{base.split('.')[0]}_{domain}_{split_dir}.json"
    return base


def _coerce_records(
    records: Sequence[Union[BOPRecord, dict]],
    domain: str,
    dataset_name: str,
    split_dir: str,
) -> List[BOPRecord]:
    normalized: List[BOPRecord] = []
    for rec in records:
        if isinstance(rec, BOPRecord):
            candidate = rec
        else:
            data = dict(rec)
            data.setdefault("domain", domain)
            data.setdefault("dataset_name", dataset_name)
            data.setdefault("split_dir", split_dir)
            candidate = record_from_dict(data)

        if candidate.domain != domain:
            raise ValueError(f"Injected record domain mismatch: {candidate.domain} != {domain}")
        if candidate.dataset_name != dataset_name:
            raise ValueError(
                f"Injected record dataset_name mismatch: {candidate.dataset_name} != {dataset_name}"
            )
        if candidate.split_dir != split_dir:
            raise ValueError(
                f"Injected record split_dir mismatch: {candidate.split_dir} != {split_dir}"
            )
        normalized.append(candidate)
    normalized.sort(key=lambda r: (r.scene_id, r.im_id))
    return normalized


class BOPDatasetAdapter:
    """
    Adapter for BOP RGB data. Supports scanning a split directory or using injected records.
    """

    def __init__(
        self,
        datasets_root: str,
        dataset_name: str,
        split_dir: str,
        domain: Literal["syn", "real"],
        val_ratio: float = 0.1,
        seed: int = 42,
        cache_enable: bool = True,
        cache_dir_name: str = "__cache__",
        index_cache_file: str = "index.json",
        split_cache_file: str = "split.json",
        scene_cache_max: int = 64,
        injected_records_train: Optional[Sequence[Union[BOPRecord, dict]]] = None,
        injected_records_val: Optional[Sequence[Union[BOPRecord, dict]]] = None,
        mode: Literal["train", "val", "all"] = "train",
    ):
        self.datasets_root = str(datasets_root)
        self.dataset_name = dataset_name
        self.split_dir = split_dir
        self.domain = domain
        self.val_ratio = val_ratio
        self.seed = seed
        self.mode = mode

        self.cache_enable = cache_enable
        self.cache_dir = Path(self.datasets_root) / dataset_name / cache_dir_name
        self.index_cache_path = self.cache_dir / _decorate_cache_filename(
            index_cache_file, domain, split_dir
        )
        self.split_cache_path = self.cache_dir / _decorate_cache_filename(
            split_cache_file, domain, split_dir
        )

        self._image_cache = _LRUCache(scene_cache_max)
        self._scene_info_cache: _LRUCache = _LRUCache(scene_cache_max)

        self.records_all: List[BOPRecord] = []
        self.records_train: List[BOPRecord] = []
        self.records_val: List[BOPRecord] = []

        if injected_records_train is not None and injected_records_val is not None:
            self.records_train = _coerce_records(
                injected_records_train, domain=domain, dataset_name=dataset_name, split_dir=split_dir
            )
            self.records_val = _coerce_records(
                injected_records_val, domain=domain, dataset_name=dataset_name, split_dir=split_dir
            )
            self.records_all = sorted(self.records_train + self.records_val, key=lambda r: (r.scene_id, r.im_id))
        else:
            self._build_from_split(
                datasets_root=self.datasets_root,
                dataset_name=dataset_name,
                split_dir=split_dir,
                val_ratio=val_ratio,
                seed=seed,
            )

        self.set_mode(mode)

    def _build_from_split(
        self, datasets_root: str, dataset_name: str, split_dir: str, val_ratio: float, seed: int
    ) -> None:
        split_path, dp_split = _resolve_split_path(datasets_root, dataset_name, split_dir)
        if not split_path.exists():
            raise FileNotFoundError(f"Split path does not exist: {split_path}")

        expected_dirs = _expected_rgb_dirs(dp_split, scene_id=0)
        extensions = _expected_extensions(dp_split)

        records_all: Optional[List[BOPRecord]] = None
        if self.cache_enable and self.index_cache_path.exists():
            try:
                data = load_index_json(self.index_cache_path)
                if (
                    data.get("version") == 1
                    and data.get("dataset_name") == dataset_name
                    and data.get("domain") == self.domain
                    and data.get("split_dir") == split_dir
                ):
                    records_all = [
                        record_from_dict({**rec, "dataset_name": dataset_name, "split_dir": split_dir, "domain": self.domain})
                        for rec in data.get("records", [])
                    ]
            except Exception:
                records_all = None

        if records_all is None:
            records_all = build_records_for_split(
                str(split_path),
                split_dir=split_dir,
                domain=self.domain,
                dataset_name=dataset_name,
                expected_rgb_dirs=expected_dirs,
                extensions=extensions,
            )
            if self.cache_enable:
                payload = {
                    "version": 1,
                    "datasets_root": str(datasets_root),
                    "dataset_name": dataset_name,
                    "domain": self.domain,
                    "split_dir": split_dir,
                    "records": [record_to_dict(r) for r in records_all],
                }
                save_index_json(self.index_cache_path, payload)

        self.records_all = records_all

        # Split cache
        train_records: Optional[List[BOPRecord]] = None
        val_records: Optional[List[BOPRecord]] = None
        if self.cache_enable and self.split_cache_path.exists():
            try:
                split_data = load_split_json(self.split_cache_path)
                if (
                    split_data.get("version") == 1
                    and split_data.get("strategy") == "by_scene"
                    and split_data.get("seed") == seed
                    and float(split_data.get("val_ratio", -1)) == float(val_ratio)
                    and split_data.get("domain") == self.domain
                    and split_data.get("split_dir") == split_dir
                ):
                    val_scene_ids = set(split_data.get("val_scenes", []))
                    train_scene_ids = set(split_data.get("train_scenes", []))
                    train_records = [r for r in records_all if r.scene_id in train_scene_ids]
                    val_records = [r for r in records_all if r.scene_id in val_scene_ids]
            except Exception:
                train_records = None
                val_records = None

        if train_records is None or val_records is None:
            train_records, val_records, split_info = split_by_scene(records_all, val_ratio, seed)
            if self.cache_enable:
                payload = {
                    "version": 1,
                    "strategy": "by_scene",
                    "seed": seed,
                    "val_ratio": val_ratio,
                    "domain": self.domain,
                    "split_dir": split_dir,
                    "train_scenes": split_info["train_scenes"],
                    "val_scenes": split_info["val_scenes"],
                }
                save_split_json(self.split_cache_path, payload)

        self.records_train = train_records
        self.records_val = val_records

    def set_mode(self, mode: Literal["train", "val", "all"]) -> None:
        if mode not in ("train", "val", "all"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def _active_records(self) -> List[BOPRecord]:
        if self.mode == "train":
            return self.records_train
        if self.mode == "val":
            return self.records_val
        return self.records_all

    def __len__(self) -> int:
        return len(self._active_records())

    def __getitem__(self, idx: int) -> dict:
        records = self._active_records()
        rec = records[idx]

        cached = self._image_cache.get(rec.rgb_path)
        if cached is not None:
            img_np = cached
        else:
            try:
                with Image.open(rec.rgb_path) as img:
                    img_np = np.array(img.convert("RGB"), copy=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read RGB image at {rec.rgb_path} (scene_id={rec.scene_id}, im_id={rec.im_id})"
                ) from exc
            self._image_cache.put(rec.rgb_path, img_np)

        ann = self._load_ann_for_record(rec, img_np.shape[:2])

        return {
            "image": img_np,
            "meta": {
                "dataset_name": rec.dataset_name,
                "domain": rec.domain,
                "split_dir": rec.split_dir,
                "scene_id": rec.scene_id,
                "im_id": rec.im_id,
                "rgb_path": rec.rgb_path,
            },
            "ann": ann,
        }

    # ---- Annotation helpers (bbox/mask) ----

    def _scene_dir_from_rgb_path(self, rgb_path: str) -> Path:
        # .../<split>/<scene_id>/rgb/<im_id>.png -> .../<split>/<scene_id>
        p = Path(rgb_path)
        return p.parent.parent

    def _load_scene_gt_info_cached(self, scene_dir: Path) -> Optional[dict]:
        key = str(scene_dir)
        cached = self._scene_info_cache.get(key)
        if cached is not None:
            # Empty dict indicates cached miss.
            return cached or None
        info_path = scene_dir / "scene_gt_info.json"
        data: dict = {}
        if info_path.exists():
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        self._scene_info_cache.put(key, data)
        return data or None

    def _mask_path_for_instance(self, scene_dir: Path, im_id: int, gt_id: int) -> Tuple[Optional[Path], Optional[str]]:
        stem = f"{im_id:06d}_{gt_id:06d}.png"
        for subdir, source in (("mask_visib", "mask_visib"), ("mask", "mask")):
            candidate = scene_dir / subdir / stem
            if candidate.exists():
                return candidate, source
        return None, None

    def _load_mask_for_instance(
        self, scene_dir: Path, im_id: int, gt_id: int, rgb_hw: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        mask_path, source = self._mask_path_for_instance(scene_dir, im_id, gt_id)
        if mask_path is None:
            return None, None
        try:
            with Image.open(mask_path) as m_img:
                mask_np = np.array(m_img, copy=True)
        except Exception:
            return None, None

        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = (mask_np > 0).astype(np.uint8)

        if mask_np.shape[:2] != rgb_hw:
            return None, None
        return mask_np, source

    def _load_ann_for_record(self, rec: BOPRecord, rgb_hw: Tuple[int, int]) -> dict:
        ann = {
            "bboxes_xywh": [],
            "bbox_source": "",
            "masks": [],
            "mask_source": "",
            "gt_ids": [],
            "valid": False,
        }

        if rec.domain != "syn":
            return ann

        scene_dir = self._scene_dir_from_rgb_path(rec.rgb_path)
        scene_info = self._load_scene_gt_info_cached(scene_dir)
        if not scene_info:
            return ann

        im_key_str = str(rec.im_id)
        im_entries = scene_info.get(im_key_str)
        if im_entries is None and rec.im_id in scene_info:
            im_entries = scene_info.get(rec.im_id)
        if not im_entries:
            return ann

        instances = list(enumerate(im_entries))
        if isinstance(im_entries[0], dict) and "visib_fract" in im_entries[0]:
            instances = sorted(instances, key=lambda t: t[1].get("visib_fract", 0.0), reverse=True)
        instances = instances[:5]

        for gt_id, inst in instances:
            if not isinstance(inst, dict):
                continue
            bbox = inst.get("bbox_visib") or inst.get("bbox_obj")
            if bbox is not None and len(bbox) >= 4:
                ann["bboxes_xywh"].append([int(round(x)) for x in bbox[:4]])
                if not ann["bbox_source"]:
                    ann["bbox_source"] = "bbox_visib" if "bbox_visib" in inst else "bbox_obj"

            mask_np, mask_source = self._load_mask_for_instance(scene_dir, rec.im_id, gt_id, rgb_hw)
            if mask_np is not None:
                ann["masks"].append(mask_np)
                if not ann["mask_source"]:
                    ann["mask_source"] = mask_source or ""

            if bbox is not None or mask_np is not None:
                ann["gt_ids"].append(gt_id)

        ann["valid"] = bool(ann["bboxes_xywh"] or ann["masks"])
        return ann
