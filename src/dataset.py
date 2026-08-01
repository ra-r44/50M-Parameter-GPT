"""Memory-mapped token dataset for language model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBinDataset(Dataset):
    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.block_size = block_size
        self.data = np.memmap(Path(bin_path), dtype=np.uint16, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(
                f"Dataset at {bin_path} is too small ({len(self.data)} tokens) "
                f"for block_size={block_size}"
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y
