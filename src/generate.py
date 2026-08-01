"""Text generation from a trained checkpoint."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import yaml

from model import GPT, GPTConfig


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode_text(text: str, meta: dict) -> list[int]:
    if meta.get("encoding_name"):
        import tiktoken

        enc = tiktoken.get_encoding(meta["encoding_name"])
        return enc.encode(text)
    raise ValueError("Tokenizer metadata missing encoding_name")


def decode_tokens(tokens: list[int], meta: dict) -> str:
    import tiktoken

    enc = tiktoken.get_encoding(meta["encoding_name"])
    return enc.decode(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint")
    parser.add_argument("--config", type=str, default="config/model_50m_colab.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / args.config)
    meta_path = project_root / config["data"]["meta_pkl"]
    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = GPTConfig(**config["model"])
    model = GPT(model_cfg).to(device)
    checkpoint = torch.load(project_root / args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    start_ids = encode_text(args.prompt, meta)
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    output = model.generate(idx, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
    text = decode_tokens(output[0].tolist(), meta)
    print("\n--- Generated Text ---\n")
    print(text)
    print("\n----------------------\n")


if __name__ == "__main__":
    main()
