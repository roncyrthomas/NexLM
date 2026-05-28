"""Rotary Positional Embedding (RoPE) primitives.

Split out of diff_attn_block.py per the blueprint pre-plan so the attention
file stays focused on the differential-attention algorithm.

Reference: Su et al., RoFormer (arXiv:2104.09864).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim by 90deg in two-element pairs: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a tensor of shape (..., seq, head_dim).

    `cos` and `sin` are broadcast-compatible with `x`'s last two dims.
    """
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """Caches cos/sin tables sized for `max_seq_len`. Slice at forward time."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, head_dim/2)
        # duplicate to (T, head_dim) so it matches the rotate_half pairing
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        assert seq_len <= self.max_seq_len, (
            f"seq_len {seq_len} exceeds RoPE cache size {self.max_seq_len}"
        )
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        # broadcast shape: (1, 1, seq_len, head_dim) for (B, H, T, D) inputs
        return cos[None, None, :, :], sin[None, None, :, :]
