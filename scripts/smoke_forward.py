"""Smoke test a trained checkpoint by generating a continuation.

Usage:
    python scripts/smoke_forward.py --ckpt out/smoke_30m/final.pt --prompt "Once upon a time"
"""

import argparse

import torch
from transformers import GPT2TokenizerFast

from model.backbone import Frankenstein
from model.config import ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = Frankenstein(cfg).to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok.encode(args.prompt, return_tensors="pt").to(device)
    out = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k)
    text = tok.decode(out[0].tolist())
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
