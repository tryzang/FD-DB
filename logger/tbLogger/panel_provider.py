import random
from typing import Dict, List, Optional

import torch


class PanelDataProvider:
    """
    Deterministic sampler for fixed (A, B) pairs used in TensorBoard panels.
    """

    def __init__(
        self,
        syn_adapter,
        real_adapter,
        transform,
        num_samples: int = 2,
        seed: int = 42,
    ):
        self.syn_adapter = syn_adapter
        self.real_adapter = real_adapter
        self.transform = transform
        self.num_samples = max(int(num_samples), 1)
        self.seed = int(seed)
        self._fixed: Optional[Dict] = None

    def _select_indices(self, length: int, rng: random.Random) -> List[int]:
        if length <= 0:
            return []
        if length >= self.num_samples:
            return rng.sample(range(length), self.num_samples)
        indices = list(range(length))
        while len(indices) < self.num_samples:
            indices.append(rng.choice(range(length)))
        return indices

    def set_fixed_samples(self):
        rng = random.Random(self.seed)
        a_indices = self._select_indices(len(self.syn_adapter), rng)
        b_indices = self._select_indices(len(self.real_adapter), rng)

        a_tensors = []
        b_tensors = []
        a_meta = []
        b_meta = []
        a_ann = []
        b_ann = []

        for idx in a_indices:
            sample = self.syn_adapter[idx]
            a_tensors.append(self.transform(sample["image"]))
            a_meta.append(sample.get("meta", {}))
            a_ann.append(sample.get("ann", {}))

        for idx in b_indices:
            sample = self.real_adapter[idx]
            b_tensors.append(self.transform(sample["image"]))
            b_meta.append(sample.get("meta", {}))
            b_ann.append(sample.get("ann", {}))

        if not a_tensors or not b_tensors:
            self._fixed = None
            return

        self._fixed = {
            "A": torch.stack(a_tensors, dim=0),
            "B": torch.stack(b_tensors, dim=0),
            "A_meta": a_meta,
            "B_meta": b_meta,
            "A_ann": a_ann,
            "B_ann": b_ann,
        }

    def get_fixed_samples(self) -> Optional[Dict]:
        return self._fixed
