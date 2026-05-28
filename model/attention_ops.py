"""Generic attention math primitives shared by DiffAttn (and future cross-attn).

Split out of diff_attn_block.py per the blueprint pre-plan.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def scaled_softmax_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Standard scaled-dot-product attention with optional causal mask.

    q, k, v: (B, H, T, D). Returns (B, H, T, D).
    """
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = (q @ k.transpose(-1, -2)) * scale
    if causal:
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=q.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask
    weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
    return weights @ v


def gqa_repeat(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match Q head count for GQA. x: (B, n_kv_heads, T, D)."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)
