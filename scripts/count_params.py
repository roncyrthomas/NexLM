"""Print parameter count for a config. Usage:
    python scripts/count_params.py --config configs/smoke_30m.yaml
"""

import argparse
from pathlib import Path

import yaml

from model.backbone import Frankenstein
from model.config import ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    shape = cfg["model"]["shape"]
    model_cfg = getattr(ModelConfig, shape)()
    model = Frankenstein(model_cfg)
    n = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"shape: {shape}")
    print(f"total params:     {n:,}")
    print(f"trainable params: {n_trainable:,}")
    # breakdown
    by_module = {}
    for name, p in model.named_parameters():
        mod = name.split(".")[0]
        by_module[mod] = by_module.get(mod, 0) + p.numel()
    print("\nby top-level module:")
    for mod, cnt in sorted(by_module.items(), key=lambda x: -x[1]):
        print(f"  {mod:20s} {cnt:>12,}  ({100*cnt/n:.1f}%)")


if __name__ == "__main__":
    main()
