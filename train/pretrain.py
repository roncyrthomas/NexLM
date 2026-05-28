"""Pretrain entrypoint for the Frankenstein SLM.

M1a smoke usage (TinyStories, 30M model):
    python train/pretrain.py --config configs/smoke_30m.yaml

Designed to work identically on Windows native (pure-PyTorch Mamba2 fallback)
and Linux cloud (CUDA fast path), without code changes.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import yaml

from data.streaming import build_tinystories_loader
from model.backbone import Frankenstein
from model.config import ModelConfig


def get_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    decay_progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, decay_progress)))


def make_optimizer(model: torch.nn.Module, weight_decay: float, lr: float):
    """AdamW with weight decay only on 2D parameters (no decay on norms/biases)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def estimate_val_loss(model, val_loader, steps: int = 50) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        x, y = next(val_loader)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg_yaml = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] CUDA not available; running on CPU (will be slow)")

    # build model
    shape = cfg_yaml["model"]["shape"]
    if shape == "smoke_30m":
        model_cfg = ModelConfig.smoke_30m()
    elif shape == "production_700m":
        model_cfg = ModelConfig.production_700m()
    else:
        raise ValueError(f"unknown shape {shape}")
    model = Frankenstein(model_cfg).to(device)
    dtype = torch.bfloat16 if cfg_yaml["training"].get("bf16") and device == "cuda" else torch.float32
    model = model.to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] model: {shape}, {n_params:,} params, dtype={dtype}, device={device}")

    if cfg_yaml["training"].get("compile") and device == "cuda":
        print("[init] torch.compile enabled")
        model = torch.compile(model, mode="max-autotune")

    # data
    bs = cfg_yaml["training"]["batch_size"]
    seq = model_cfg.max_seq_len
    train_loader = build_tinystories_loader("train", seq, bs, seed=0, device=device)
    val_loader = build_tinystories_loader("validation", seq, bs, seed=1, device=device)

    # optimizer
    base_lr = float(cfg_yaml["training"]["lr"])
    weight_decay = float(cfg_yaml["training"]["weight_decay"])
    opt = make_optimizer(model, weight_decay, base_lr)

    # logging
    if args.wandb:
        import wandb

        wandb.init(
            project=cfg_yaml["logging"]["wandb_project"],
            config={"model": shape, "params": n_params, **cfg_yaml["training"]},
        )

    # state
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        print(f"[resume] step={step}")

    out_dir = Path(cfg_yaml["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = cfg_yaml["training"]["max_steps"]
    warmup = cfg_yaml["training"]["warmup_steps"]
    grad_clip = float(cfg_yaml["training"]["grad_clip"])
    val_every = cfg_yaml["eval"]["val_every"]
    save_every = cfg_yaml["eval"]["save_every"]
    log_every = cfg_yaml["logging"]["log_every"]
    target_val_loss = float(cfg_yaml["eval"]["target_val_loss"])

    print(f"[train] starting from step {step}/{max_steps}")
    model.train()
    t0 = time.time()
    running_loss = 0.0

    while step < max_steps:
        x, y = next(train_loader)
        lr = get_lr(step, max_steps, warmup, base_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        running_loss += loss.item()
        step += 1

        if step % log_every == 0:
            avg = running_loss / log_every
            running_loss = 0.0
            elapsed = time.time() - t0
            toks = step * bs * seq
            tps = toks / elapsed
            print(f"step {step}/{max_steps} | loss {avg:.4f} | lr {lr:.2e} | {tps:.0f} tok/s | {elapsed:.0f}s")
            if args.wandb:
                wandb.log({"train/loss": avg, "train/lr": lr, "train/tokens_per_sec": tps}, step=step)

        if step % val_every == 0:
            vl = estimate_val_loss(model, val_loader, cfg_yaml["eval"]["val_steps"])
            print(f"[val] step {step} | val_loss {vl:.4f} | target {target_val_loss}")
            if args.wandb:
                wandb.log({"val/loss": vl}, step=step)
            if vl < target_val_loss:
                print(f"[exit] target val loss {target_val_loss} reached at step {step}")
                save_ckpt(model, opt, step, out_dir / "final.pt", model_cfg)
                return

        if step % save_every == 0:
            save_ckpt(model, opt, step, out_dir / f"ckpt_{step}.pt", model_cfg)

    save_ckpt(model, opt, step, out_dir / "final.pt", model_cfg)
    print(f"[done] {max_steps} steps in {(time.time()-t0)/60:.1f} min")


def save_ckpt(model, opt, step, path, cfg):
    state = {
        "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
        "optim": opt.state_dict(),
        "step": step,
        "config": cfg.__dict__,
    }
    torch.save(state, path)
    print(f"[save] {path}")


if __name__ == "__main__":
    main()
