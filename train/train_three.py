"""Unified trainer for the three-way comparison.

Trains Vanilla, Frank v1, Frank v2 on the SAME data with the SAME
hyperparameters, then saves each LoRA adapter (+ tier state for v1/v2)
into separate directories. The eval runner then loads them for comparison.

Usage:
    python train/train_three.py --base smollm2 --steps 500
    python train/train_three.py --base phi3 --steps 1000 --wandb
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agent.config import AgentConfig
from agent.wrapper import NexAgent
from train.sft import SFTConfig, run_sft


BASE_PRESETS = {
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi3":    "microsoft/Phi-3-mini-4k-instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def _build_agent(variant: str, base_name: str) -> NexAgent:
    base = BASE_PRESETS.get(base_name, base_name)
    if variant == "vanilla":
        cfg = AgentConfig.vanilla(base=base)
    elif variant == "frank_v1":
        cfg = AgentConfig.frank_v1(base=base)
    elif variant == "frank_v2":
        cfg = AgentConfig.frank_v2(base=base)
    else:
        raise ValueError(f"unknown variant: {variant}")
    return NexAgent(cfg).cuda()


def train_one(
    variant: str,
    base_name: str,
    sft_cfg: SFTConfig,
    out_root: Path,
    use_wandb: bool = False,
) -> dict:
    """Train one of {vanilla, frank_v1, frank_v2} and save its state."""
    print(f"\n{'=' * 60}\n[train_three] variant: {variant}\n{'=' * 60}")
    agent = _build_agent(variant, base_name)
    total, trainable = agent.count_params()
    print(f"[init] total={total:,} trainable={trainable:,}")

    # Override SFT output dir per variant
    sft_cfg.output_dir = str(out_root / variant)
    sft_cfg.wandb_project = f"nexlm-{variant}"
    Path(sft_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run_sft(agent, sft_cfg, use_wandb=use_wandb)
    elapsed = time.time() - t0

    # Save final agent state (LoRA + tier states)
    agent.save_state(out_root / variant / "final_state")

    return {
        "variant": variant,
        "trainable_params": trainable,
        "wall_seconds": elapsed,
        "output_dir": sft_cfg.output_dir,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="smollm2", choices=list(BASE_PRESETS) + ["custom"])
    p.add_argument("--variants", default="vanilla,frank_v1,frank_v2",
                   help="comma-separated subset of {vanilla, frank_v1, frank_v2}")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--dataset_split", default="train_sft")
    p.add_argument("--text_field", default="messages")
    p.add_argument("--out_root", default="out/three_way")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    sft_cfg = SFTConfig(
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        text_field=args.text_field,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_steps=args.steps,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_log = []
    for variant in args.variants.split(","):
        variant = variant.strip()
        try:
            info = train_one(variant, args.base, sft_cfg, out_root, use_wandb=args.wandb)
        except Exception as e:
            info = {"variant": variant, "error": str(e)}
            print(f"[train_three] {variant} FAILED: {e}")
        run_log.append(info)

    with open(out_root / "training_summary.json", "w") as f:
        json.dump(run_log, f, indent=2)
    print(f"\n[train_three] all done. Summary: {out_root / 'training_summary.json'}")


if __name__ == "__main__":
    main()
