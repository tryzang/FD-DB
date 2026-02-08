#!/usr/bin/env python3
"""Summarize evaluation results across variants."""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List

import yaml
from tabulate import tabulate


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def summarize(cfg: Dict[str, Any], variants: List[str]) -> None:
    runs_root = Path(cfg["output"]["runs_root"])
    rows = []
    for v in variants:
        metrics_path = runs_root / v / "metrics.json"
        if not metrics_path.exists():
            print(f"[warn] {v} missing metrics.json, skipped")
            continue
        m = load_metrics(metrics_path)
        rows.append([v, m.get("split"), m.get("miou"), m.get("dice"), m.get("num_images")])

    if not rows:
        print("No metric files found")
        return

    headers = ["variant", "split", "mIoU", "Dice", "num_images"]
    print(tabulate(rows, headers=headers, floatfmt=".4f"))

    out_dir = runs_root / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
    print(f"Summary saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize metrics across variants")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--variants", nargs="*", help="Optional: explicit variant list, otherwise use summary.variants from config")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    variants = args.variants or cfg.get("summary", {}).get("variants") or list(cfg["dataset"]["variants"].keys())
    summarize(cfg, variants)


if __name__ == "__main__":
    main()
