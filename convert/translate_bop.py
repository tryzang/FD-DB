"""
Translate BOP synthetic RGB images to the target domain using a trained FD-DB generator.

Usage (example):
    python convert/translate_bop.py \
        --checkpoint run/checkpoint/step0005000_epoch0001.pt \
        --datasets-root /path/to/bop/datasets \
        --dataset-name ycbv \
        --split ycbv_train_pbr \
        --output-dir /path/to/output/ycbv_translated/ours \
        --device auto \
        --noise-mode fixed \
        --preserve-resolution
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.bop.adapter import BOPDatasetAdapter
from logger.tbLogger.image_utils import denormalize
from models.generator.generator import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BOP synthetic images with a trained FD-DB model.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt saved during training.")
    parser.add_argument("--output-dir", required=True, help="Directory to write translated images (mirrors source structure).")
    parser.add_argument("--datasets-root", default=None, help="BOP datasets root; defaults to config/ckpt value.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name, e.g., ycbv; defaults to config/ckpt value.")
    parser.add_argument("--split", default=None, help="Split directory name to translate, e.g., ycbv_train_pbr.")
    parser.add_argument("--config-image-size", type=int, default=None, help="Override image_size used when --resize-to-config is set (defaults to config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for conversion.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--noise-mode", choices=["fixed", "random"], default="fixed", help="Fixed zero noise for deterministic outputs or random to match training noise.")
    parser.add_argument("--preserve-resolution", action="store_true", help="Resize outputs back to original HxW after generation.")
    parser.add_argument("--resize-to-config", action="store_true", help="Resize inputs to config image_size before generation (default: keep original HxW).")
    parser.add_argument(
        "--scene-limit",
        "--limit",
        dest="scene_limit",
        type=int,
        default=None,
        help="Optional max number of scenes to convert (debug, ordered by scene id).",
    )
    parser.add_argument("--no-symlink-aux", action="store_true", help="Disable symlinking non-RGB auxiliary files (mask/depth/meta).")
    parser.add_argument("--g-res", type=float, default=None, help="Override residual gate g_res in [0,1]; None uses checkpoint/default.")
    parser.add_argument("--sigma-fuse", type=float, default=None, help="Override high-pass sigma for final output (0 disables HP); None uses checkpoint/default.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_transform(image_size: Optional[int]):
    # If no image_size is provided, keep the original resolution to avoid shrink-then-upsample.
    if image_size is None:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class BOPTranslationDataset(Dataset):
    def __init__(self, adapter: BOPDatasetAdapter, transform, split_root: Path):
        self.adapter = adapter
        self.transform = transform
        self.split_root = split_root

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.adapter[idx]
        img_np = rec["image"]
        meta = rec["meta"]
        h, w = img_np.shape[:2]
        img_t = self.transform(img_np)
        rgb_path = Path(meta.get("rgb_path", ""))
        try:
            rel_path = rgb_path.relative_to(self.split_root)
        except Exception:
            rel_path = Path(f"{int(meta.get('scene_id', 0)):06d}") / "rgb" / f"{int(meta.get('im_id', 0)):06d}.png"
        return {
            "image": img_t,
            "relative_path": rel_path,
            "orig_size": (h, w),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    rel_paths = [b["relative_path"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    return {"image": images, "relative_path": rel_paths, "orig_size": orig_sizes}


def select_scene_subset(
    adapter: BOPDatasetAdapter, max_scenes: Optional[int]
) -> Tuple[Optional[List[int]], List[int]]:
    """
    Return dataset indices belonging to the first `max_scenes` scene_ids (sorted).

    When max_scenes is None or <=0, returns (None, []) to indicate no filtering.
    """
    if max_scenes is None or max_scenes <= 0:
        return None, []

    if adapter.mode == "train":
        records = adapter.records_train
    elif adapter.mode == "val":
        records = adapter.records_val
    else:
        records = adapter.records_all

    indices: List[int] = []
    selected_scene_ids: List[int] = []
    for idx, rec in enumerate(records):
        if rec.scene_id not in selected_scene_ids:
            if len(selected_scene_ids) >= max_scenes:
                break
            selected_scene_ids.append(rec.scene_id)
        indices.append(idx)

    return indices, selected_scene_ids


def load_config_from_sources(ckpt: Dict, args: argparse.Namespace) -> Dict:
    cfg = ckpt.get("config", {}) or {}
    # No external config override for now; use saved config as source of truth.
    return cfg


def resolve_data_cfg(cfg: Dict, args: argparse.Namespace) -> Tuple[str, str, str, Optional[int], int]:
    data_cfg = cfg.get("data", {}) or {}
    bop_cfg = data_cfg.get("bop", {}) or {}
    datasets_root = args.datasets_root or bop_cfg.get("datasets_root")
    dataset_name = args.dataset_name or bop_cfg.get("dataset_name")
    split_dir = args.split or bop_cfg.get("syn_dir") or "train_pbr"
    image_size = None if not args.resize_to_config else (args.config_image_size or data_cfg.get("image_size", 256))
    batch_size = args.batch_size or data_cfg.get("batch_size", 4)
    if datasets_root is None or dataset_name is None:
        raise ValueError("datasets_root and dataset_name must be provided via args or config.")
    resolved_image_size = None if image_size is None else int(image_size)
    return str(datasets_root), str(dataset_name), str(split_dir), resolved_image_size, int(batch_size)


def build_generator(cfg: Dict, ckpt: Dict, device: torch.device) -> Generator:
    gen_cfg = dict(cfg.get("generator", {}) or {})
    if "frequency" in cfg:
        gen_cfg["frequency"] = cfg.get("frequency", {}) or {}
    generator = Generator(gen_cfg)
    state = ckpt.get("generator")
    if state is None:
        raise ValueError("Checkpoint missing 'generator' weights.")
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[convert] Warning: missing keys {missing}, unexpected keys {unexpected} when loading generator.")
    generator.to(device)
    generator.eval()
    return generator


def maybe_resize_to_orig(img: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = orig_hw
    if img.shape[-2:] == (h, w):
        return img
    return F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def symlink_auxiliary(src_split: Path, dst_split: Path) -> int:
    """
    Symlink non-RGB files/dirs (depth, mask, gt info, etc.) so structure matches source.
    """
    linked = 0
    if not src_split.exists():
        return linked
    dst_split.mkdir(parents=True, exist_ok=True)
    for entry in src_split.iterdir():
        if entry.name == "rgb":
            continue
        if entry.is_file():
            dst = dst_split / entry.name
            if not dst.exists():
                try:
                    os.symlink(entry, dst)
                    linked += 1
                except OSError:
                    pass
            continue
        if entry.is_dir():
            if entry.name.isdigit():
                # Scene directory: mirror contents except rgb.
                dst_scene = dst_split / entry.name
                dst_scene.mkdir(parents=True, exist_ok=True)
                for child in entry.iterdir():
                    if child.name == "rgb":
                        continue
                    dst_child = dst_scene / child.name
                    if dst_child.exists():
                        continue
                    try:
                        os.symlink(child, dst_child, target_is_directory=child.is_dir())
                        linked += 1
                    except OSError:
                        pass
            else:
                dst_dir = dst_split / entry.name
                if dst_dir.exists():
                    continue
                try:
                    os.symlink(entry, dst_dir, target_is_directory=True)
                    linked += 1
                except OSError:
                    pass
    return linked


def main():
    args = parse_args()
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config_from_sources(ckpt, args)
    datasets_root, dataset_name, split_dir, image_size, batch_size = resolve_data_cfg(cfg, args)

    split_root = Path(datasets_root) / dataset_name / split_dir
    if not split_root.exists():
        raise FileNotFoundError(f"Split dir not found: {split_root}")

    generator = build_generator(cfg, ckpt, device)
    if args.sigma_fuse is not None:
        generator.sigma_fuse = float(args.sigma_fuse)
    gates = {"g_res": float(args.g_res)} if args.g_res is not None else None
    transform = build_transform(image_size)

    adapter = BOPDatasetAdapter(
        datasets_root=datasets_root,
        dataset_name=dataset_name,
        split_dir=split_dir,
        domain="syn",
        val_ratio=0.0,
        seed=cfg.get("data", {}).get("bop", {}).get("val", {}).get("seed", 42),
        cache_enable=True,
        mode="all",
    )
    dataset = BOPTranslationDataset(adapter, transform, split_root=split_root)
    scene_indices, selected_scene_ids = select_scene_subset(adapter, args.scene_limit)
    if scene_indices is not None:
        dataset = Subset(dataset, scene_indices)
        scene_list = ", ".join(f"{sid:06d}" for sid in selected_scene_ids)
        print(f"[convert] Restricting to first {len(selected_scene_ids)} scene(s): {scene_list}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_fn,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Translating", dynamic_ncols=True):
            images = batch["image"].to(device, non_blocking=True)
            out = generator(images, gates=gates)
            if hasattr(out, "y_final") and out.y_final is not None:
                fake = out.y_final
            elif hasattr(out, "y"):
                fake = out.y
            else:
                fake = out[0]
            fake = denormalize(fake).clamp(0.0, 1.0).cpu()
            bsz = fake.shape[0]

            for i in range(bsz):
                rel_path = Path(batch["relative_path"][i])
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img = fake[i]
                if args.preserve_resolution:
                    img = maybe_resize_to_orig(img, batch["orig_size"][i])
                save_image(img, out_path)
                total_saved += 1
    scene_summary = "" if not selected_scene_ids else f" from {len(selected_scene_ids)} scene(s)"
    print(f"[convert] Wrote {total_saved} images to {output_dir}{scene_summary}")
    if not args.no_symlink_aux:
        linked = symlink_auxiliary(split_root, output_dir)
        if linked:
            print(f"[convert] Symlinked {linked} auxiliary files/dirs into {output_dir}")


if __name__ == "__main__":
    main()
