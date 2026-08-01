"""Training loop for the 50M GPT foundation model."""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import TokenBinDataset
from model import GPT, GPTConfig, build_model


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def setup_cache_dirs(project_root: Path, cache_dir: str) -> None:
    cache_path = project_root / cache_dir
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_path / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "huggingface")
    os.environ["TORCH_HOME"] = str(cache_path / "torch")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lr(step: int, cfg: dict) -> float:
    base_lr = cfg["learning_rate"]
    warmup = cfg["warmup_steps"]
    decay_steps = cfg["lr_decay_steps"]
    min_lr = cfg["min_lr"]

    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if step >= decay_steps:
        return min_lr
    decay_ratio = (step - warmup) / max(decay_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    latest = checkpoint_dir / "latest.pt"
    return latest if latest.exists() else None


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    best_val_loss: float,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    step_path = checkpoint_dir / f"step_{step:07d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    print(f"Saved checkpoint: {step_path}")


def evaluate(model: GPT, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            tokens = y.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    model.train()
    return total_loss / max(total_tokens, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 50M GPT foundation model")
    parser.add_argument("--config", type=str, default="config/model_50m_colab.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint or 'auto'")
    args = parser.parse_args()

    project_root = get_project_root()
    config = load_config(project_root / args.config)
    setup_cache_dirs(project_root, config["paths"]["cache_dir"])

    train_cfg = config["training"]
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = train_cfg.get("dtype", "float16")
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16
    print(f"Using device: {device}")

    model_cfg = GPTConfig(**config["model"])
    model = build_model(model_cfg).to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    train_ds = TokenBinDataset(project_root / config["data"]["train_bin"], model_cfg.block_size)
    val_ds = TokenBinDataset(project_root / config["data"]["val_bin"], model_cfg.block_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        weight_decay=train_cfg["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and dtype == torch.float16)

    checkpoint_dir = project_root / config["paths"]["checkpoint_dir"]
    start_step = 0
    best_val_loss = float("inf")

    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(checkpoint_dir)
    elif args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path

    if resume_path and resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])

    model.train()
    train_iter = iter(train_loader)
    tokens_per_step = (
        train_cfg["batch_size"] * model_cfg.block_size * train_cfg["gradient_accumulation_steps"]
    )
    print(f"Tokens per optimizer step: {tokens_per_step:,}")

    for step in range(start_step, train_cfg["max_steps"]):
        lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_cfg["gradient_accumulation_steps"]):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / train_cfg["gradient_accumulation_steps"]
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % train_cfg["log_interval"] == 0:
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e}")

        if step > 0 and step % train_cfg["eval_interval"] == 0:
            val_loss = evaluate(model, val_loader, device, dtype)
            val_ppl = math.exp(val_loss)
            print(f"step {step:6d} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if step > 0 and step % train_cfg["checkpoint_interval"] == 0:
            save_checkpoint(checkpoint_dir, step, model, optimizer, scaler, best_val_loss)

    save_checkpoint(checkpoint_dir, train_cfg["max_steps"] - 1, model, optimizer, scaler, best_val_loss)
    print("Training complete.")


if __name__ == "__main__":
    main()
