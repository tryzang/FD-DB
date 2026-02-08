import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

from .indexing import BOPRecord


def split_by_scene(
    records: Iterable[BOPRecord], val_ratio: float, seed: int
) -> Tuple[List[BOPRecord], List[BOPRecord], dict]:
    """
    Deterministically split records by scene_id.
    """

    scene_to_records = defaultdict(list)
    for rec in records:
        scene_to_records[rec.scene_id].append(rec)

    scene_ids = list(scene_to_records.keys())
    scene_ids.sort()

    rng = random.Random(seed)
    rng.shuffle(scene_ids)

    val_count = int(math.ceil(len(scene_ids) * val_ratio))
    val_scene_ids = set(scene_ids[:val_count])
    train_scene_ids = set(scene_ids[val_count:])

    train_records: List[BOPRecord] = []
    val_records: List[BOPRecord] = []
    for rec in records:
        if rec.scene_id in val_scene_ids:
            val_records.append(rec)
        else:
            train_records.append(rec)

    split_info = {
        "version": 1,
        "strategy": "by_scene",
        "seed": seed,
        "val_ratio": val_ratio,
        "train_scenes": sorted(train_scene_ids),
        "val_scenes": sorted(val_scene_ids),
    }
    return train_records, val_records, split_info


def save_split_json(path: str, payload: dict) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

