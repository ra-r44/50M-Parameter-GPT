"""Validation perplexity evaluation."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import TokenBinDataset
from model import GPT, GPTConfig


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            tokens = y.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return math.exp(total_loss / max(total_tokens, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation perplexity")
    parser.add_argument("--config", type=str, default="config/model_50m_colab.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / args.config)
    meta_path = project_root / config["data"]["meta_pkl"]
    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = config["training"].get("dtype", "float16")
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16

    model_cfg = GPTConfig(**config["model"])
    model = GPT(model_cfg).to(device)
    checkpoint = torch.load(project_root / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    val_ds = TokenBinDataset(project_root / config["data"]["val_bin"], model_cfg.block_size)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    ppl = evaluate(model, val_loader, device, dtype)
    print(f"Validation perplexity: {ppl:.2f}")
    print(f"Vocab size: {meta['vocab_size']:,}")


if __name__ == "__main__":
    main()
