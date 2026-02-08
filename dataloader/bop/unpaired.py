import random
from typing import Callable, Optional

from .adapter import BOPDatasetAdapter


class UnpairedTranslationDataset:
    """
    Combine two adapters (syn/real) for unpaired translation training.
    """

    def __init__(
        self,
        syn_adapter: BOPDatasetAdapter,
        real_adapter: BOPDatasetAdapter,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_meta: bool = True,
    ):
        self.syn_adapter = syn_adapter
        self.real_adapter = real_adapter
        self.transform = transform
        self.target_transform = target_transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return max(len(self.syn_adapter), len(self.real_adapter))

    def __getitem__(self, idx: int) -> dict:
        syn_idx = idx % len(self.syn_adapter)
        real_idx = idx % len(self.real_adapter)

        # Introduce slight randomness for the target domain to avoid strict cycling.
        if len(self.real_adapter) > 0 and random.random() < 0.5:
            real_idx = random.randint(0, len(self.real_adapter) - 1)

        syn_sample = self.syn_adapter[syn_idx]
        real_sample = self.real_adapter[real_idx]

        a_img = syn_sample["image"]
        b_img = real_sample["image"]
        if self.transform is not None:
            a_img = self.transform(a_img)
        if self.target_transform is not None:
            b_img = self.target_transform(b_img)

        output = {
            "A": a_img,
            "B": b_img,
            "A_paths": syn_sample["meta"]["rgb_path"],
            "B_paths": real_sample["meta"]["rgb_path"],
        }
        if self.return_meta:
            output.update({"A_meta": syn_sample["meta"], "B_meta": real_sample["meta"]})
        return output

