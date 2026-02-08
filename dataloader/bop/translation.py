import copy
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as transforms

from .adapter import BOPDatasetAdapter
from .splitting import split_by_scene
from .unpaired import UnpairedTranslationDataset


def _bop_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def _build_real_from_test(
    datasets_root: str,
    dataset_name: str,
    test_dir: str,
    holdout_ratio: float,
    holdout_seed: int,
    cache_dir: Path,
    cache_enable: bool,
    cache_dir_name: str,
    scene_cache_max: int,
) -> Tuple[list, list]:
    """
    Split test records into train/val for real domain.
    """
    holdout_file = cache_dir / f"split_holdout_{test_dir}_seed{holdout_seed}_val{holdout_ratio}.json"
    train_records = None
    val_records = None

    if cache_enable and holdout_file.exists():
        try:
            from .splitting import load_split_json

            data = load_split_json(str(holdout_file))
            if (
                data.get("version") == 1
                and data.get("mode") == "split_from_test"
                and data.get("seed") == holdout_seed
                and float(data.get("val_ratio", -1)) == float(holdout_ratio)
                and data.get("dataset_name") == dataset_name
                and data.get("split_dir") == test_dir
            ):
                val_scene_ids = set(data.get("val_scenes", []))
                train_scene_ids = set(data.get("train_scenes", []))
                tmp_adapter = BOPDatasetAdapter(
                    datasets_root=datasets_root,
                    dataset_name=dataset_name,
                    split_dir=test_dir,
                    domain="real",
                    val_ratio=0.0,
                    seed=holdout_seed,
                    cache_enable=cache_enable,
                    cache_dir_name=cache_dir_name,
                    index_cache_file="index.json",
                    split_cache_file="split.json",
                    scene_cache_max=scene_cache_max,
                    mode="all",
                )
                train_records = [r for r in tmp_adapter.records_all if r.scene_id in train_scene_ids]
                val_records = [r for r in tmp_adapter.records_all if r.scene_id in val_scene_ids]
        except Exception:
            train_records = None
            val_records = None

    if train_records is None or val_records is None:
        base_adapter = BOPDatasetAdapter(
            datasets_root=datasets_root,
            dataset_name=dataset_name,
            split_dir=test_dir,
            domain="real",
            val_ratio=0.0,
            seed=holdout_seed,
            cache_enable=cache_enable,
            cache_dir_name=cache_dir_name,
            index_cache_file="index.json",
            split_cache_file="split.json",
            scene_cache_max=scene_cache_max,
            mode="all",
        )
        train_records, val_records, split_info = split_by_scene(
            base_adapter.records_all, val_ratio=holdout_ratio, seed=holdout_seed
        )
        if cache_enable:
            from .splitting import save_split_json

            payload = {
                "version": 1,
                "mode": "split_from_test",
                "dataset_name": dataset_name,
                "split_dir": test_dir,
                "seed": holdout_seed,
                "val_ratio": holdout_ratio,
                "train_scenes": split_info["train_scenes"],
                "val_scenes": split_info["val_scenes"],
            }
            holdout_file.parent.mkdir(parents=True, exist_ok=True)
            save_split_json(str(holdout_file), payload)

    return train_records, val_records


def get_bop_translation_dataloaders(data_cfg: dict, return_adapters: bool = False):
    """
    Build unpaired BOP dataloaders (train/val).
    """
    bop_cfg = data_cfg.get("bop", {})
    datasets_root = bop_cfg["datasets_root"]
    dataset_name = bop_cfg["dataset_name"]
    syn_dir = bop_cfg.get("syn_dir", "train_pbr")

    real_source = bop_cfg.get("real_source", {"mode": "folder", "dir": "train_real"})
    real_mode = real_source.get("mode", "folder")

    val_cfg = bop_cfg.get("val", {"ratio": 0.1, "seed": 42})
    val_ratio = float(val_cfg.get("ratio", 0.1))
    val_seed = int(val_cfg.get("seed", 42))

    cache_cfg = bop_cfg.get("cache", {})
    cache_enable = cache_cfg.get("enable", True)
    cache_dir_name = cache_cfg.get("dir_name", "__cache__")
    index_cache_file = cache_cfg.get("index_file", "index.json")
    split_cache_file = cache_cfg.get("split_file", "split.json")
    scene_cache_max = int(cache_cfg.get("scene_cache_max", 64))
    cache_dir = Path(datasets_root) / dataset_name / cache_dir_name

    loader_cfg = bop_cfg.get("loader", {})
    image_size = loader_cfg.get("image_size", data_cfg.get("image_size", 256))
    batch_size = loader_cfg.get("batch_size", data_cfg.get("batch_size", 4))
    num_workers = loader_cfg.get("num_workers", data_cfg.get("num_workers", 4))
    shuffle = loader_cfg.get("shuffle", True)

    syn_adapter = BOPDatasetAdapter(
        datasets_root=datasets_root,
        dataset_name=dataset_name,
        split_dir=syn_dir,
        domain="syn",
        val_ratio=val_ratio,
        seed=val_seed,
        cache_enable=cache_enable,
        cache_dir_name=cache_dir_name,
        index_cache_file=index_cache_file,
        split_cache_file=split_cache_file,
        scene_cache_max=scene_cache_max,
        mode="train",
    )

    if real_mode == "folder":
        real_split_dir = real_source.get("dir", "train_real")
        real_adapter = BOPDatasetAdapter(
            datasets_root=datasets_root,
            dataset_name=dataset_name,
            split_dir=real_split_dir,
            domain="real",
            val_ratio=val_ratio,
            seed=val_seed,
            cache_enable=cache_enable,
            cache_dir_name=cache_dir_name,
            index_cache_file=index_cache_file,
            split_cache_file=split_cache_file,
            scene_cache_max=scene_cache_max,
            mode="train",
        )
        real_train_records = real_adapter.records_train
        real_val_records = real_adapter.records_val
    elif real_mode == "split_from_test":
        test_dir = real_source["test_dir"]
        holdout = real_source.get("holdout", {"val_ratio": 0.1, "seed": 42})
        holdout_ratio = float(holdout.get("val_ratio", 0.1))
        holdout_seed = int(holdout.get("seed", 42))
        real_train_records, real_val_records = _build_real_from_test(
            datasets_root=datasets_root,
            dataset_name=dataset_name,
            test_dir=test_dir,
            holdout_ratio=holdout_ratio,
            holdout_seed=holdout_seed,
            cache_dir=cache_dir,
            cache_enable=cache_enable,
            cache_dir_name=cache_dir_name,
            scene_cache_max=scene_cache_max,
        )
        real_adapter = BOPDatasetAdapter(
            datasets_root=datasets_root,
            dataset_name=dataset_name,
            split_dir=test_dir,
            domain="real",
            val_ratio=val_ratio,
            seed=val_seed,
            cache_enable=False,  # injection skips scanning/caching
            cache_dir_name=cache_dir_name,
            index_cache_file=index_cache_file,
            split_cache_file=split_cache_file,
            scene_cache_max=scene_cache_max,
            injected_records_train=real_train_records,
            injected_records_val=real_val_records,
            mode="train",
        )
    else:
        raise ValueError(f"Unsupported real_source.mode: {real_mode}")

    # Build val adapters via injection to avoid mutating the training adapters.
    syn_val_adapter = BOPDatasetAdapter(
        datasets_root=datasets_root,
        dataset_name=dataset_name,
        split_dir=syn_dir,
        domain="syn",
        val_ratio=val_ratio,
        seed=val_seed,
        cache_enable=False,
        cache_dir_name=cache_dir_name,
        index_cache_file=index_cache_file,
        split_cache_file=split_cache_file,
        scene_cache_max=scene_cache_max,
        injected_records_train=copy.deepcopy(syn_adapter.records_train),
        injected_records_val=copy.deepcopy(syn_adapter.records_val),
        mode="val",
    )
    real_val_adapter = BOPDatasetAdapter(
        datasets_root=datasets_root,
        dataset_name=dataset_name,
        split_dir=real_adapter.split_dir,
        domain="real",
        val_ratio=val_ratio,
        seed=val_seed,
        cache_enable=False,
        cache_dir_name=cache_dir_name,
        index_cache_file=index_cache_file,
        split_cache_file=split_cache_file,
        scene_cache_max=scene_cache_max,
        injected_records_train=copy.deepcopy(real_train_records),
        injected_records_val=copy.deepcopy(real_val_records),
        mode="val",
    )

    transform = _bop_transform(image_size)
    train_dataset = UnpairedTranslationDataset(
        syn_adapter=syn_adapter,
        real_adapter=real_adapter,
        transform=transform,
        target_transform=transform,
        return_meta=True,
    )
    val_dataset = UnpairedTranslationDataset(
        syn_adapter=syn_val_adapter,
        real_adapter=real_val_adapter,
        transform=transform,
        target_transform=transform,
        return_meta=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        drop_last=False,
    )
    if return_adapters:
        return train_loader, val_loader, syn_adapter, real_adapter, transform
    return train_loader, val_loader
