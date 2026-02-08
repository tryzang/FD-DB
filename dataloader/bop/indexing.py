import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class BOPRecord:
    """Single RGB entry in a BOP split."""

    domain: str  # "syn" or "real"
    dataset_name: str
    split_dir: str
    scene_id: int
    im_id: int
    rgb_path: str  # absolute path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def record_to_dict(record: BOPRecord) -> dict:
    return {
        "domain": record.domain,
        "dataset_name": record.dataset_name,
        "split_dir": record.split_dir,
        "scene_id": record.scene_id,
        "im_id": record.im_id,
        "rgb_path": record.rgb_path,
    }


def record_from_dict(data: dict) -> BOPRecord:
    required = ["domain", "dataset_name", "split_dir", "scene_id", "im_id", "rgb_path"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in record dict: {missing}")
    return BOPRecord(
        domain=data["domain"],
        dataset_name=data["dataset_name"],
        split_dir=data["split_dir"],
        scene_id=int(data["scene_id"]),
        im_id=int(data["im_id"]),
        rgb_path=str(data["rgb_path"]),
    )


def build_records_for_split(
    split_path: str,
    split_dir: str,
    domain: str,
    dataset_name: str,
    expected_rgb_dirs: Optional[Iterable[str]] = None,
    extensions: Optional[Iterable[str]] = None,
) -> List[BOPRecord]:
    """
    Scan a BOP split directory for RGB images and return sorted BOPRecord list.
    """

    if expected_rgb_dirs is None:
        expected_rgb_dirs = ["rgb"]
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".tif"]

    split_root = Path(split_path)
    if not split_root.exists():
        raise FileNotFoundError(f"Split path not found: {split_root}")

    scene_dirs = [p for p in split_root.iterdir() if p.is_dir()]
    if not scene_dirs:
        raise FileNotFoundError(f"No scene directories under split: {split_root}")

    records: List[BOPRecord] = []
    ext_lc = {e.lower() for e in extensions}

    for scene_dir in sorted(scene_dirs, key=lambda p: p.name):
        try:
            scene_id = int(scene_dir.name)
        except ValueError as exc:
            raise ValueError(f"Scene directory name must be int-convertible: {scene_dir.name}") from exc

        rgb_dir = None
        for candidate in expected_rgb_dirs:
            candidate_dir = scene_dir / candidate
            if candidate_dir.exists():
                rgb_dir = candidate_dir
                break
        if rgb_dir is None:
            raise FileNotFoundError(
                f"RGB directory not found in scene {scene_id}: tried {list(expected_rgb_dirs)} under {scene_dir}"
            )

        files = [p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_lc]
        if not files:
            raise FileNotFoundError(f"No RGB files found in {rgb_dir}")

        for img_path in sorted(files, key=lambda p: p.name):
            try:
                im_id = int(img_path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"Image filename must be int-convertible (without extension): {img_path.name}"
                ) from exc
            records.append(
                BOPRecord(
                    domain=domain,
                    dataset_name=dataset_name,
                    split_dir=split_dir,
                    scene_id=scene_id,
                    im_id=im_id,
                    rgb_path=str(img_path.resolve()),
                )
            )

    records.sort(key=lambda r: (r.scene_id, r.im_id))
    return records


def save_index_json(path: str, payload: dict) -> None:
    path_obj = Path(path)
    _ensure_parent(path_obj)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_index_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

