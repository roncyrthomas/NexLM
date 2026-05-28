"""Differential attention block (Ye et al., arXiv:2410.05258).

Two attention maps with separate Q/K projections; their difference cancels
attention noise. We use GQA for the V projection and RoPE on Q/K.

Critical order fix from adversarial review (⟂ REVIEW H1):
    RoPE is applied to q1, q2, k1, k2 *individually* BEFORE the GQA expansion
    of k1/k2 to n_heads. Applying RoPE after expansion only to half the keys
    leaves k2 without positional information and breaks the differential map.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.attention_ops import gqa_repeat, scaled_softmax_attn
from model.config import ModelConfig
from model.mamba2_block import SwiGLU
from model.rope import RotaryEmbedding, apply_rope


class DiffAttnBlock(nn.Module):
    """Single differential-attention block: pre-norm, diff-attn, pre-norm, FFN."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        # pre-attention norm
        self.attn_norm = nn.RMSNorm(cfg.d_model)

        # projections: Q is doubled (q1 || q2), K is doubled (k1 || k2), V is single
        self.q_proj = nn.Linear(cfg.d_model, 2 * cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # lambda parameters per the paper; lambda_init depends on depth
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_init = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_q1 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(cfg.head_dim) * 0.1)

        # output sub-layer norm (paper appendix A)
        self.subln = nn.RMSNorm(cfg.n_heads * cfg.head_dim)

        # rope
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, base=cfg.rope_base)

        # ffn
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        d_ff = int(cfg.d_model * cfg.ffn_expansion)
        d_ff = ((d_ff + 7) // 8) * 8
        self.ffn = SwiGLU(cfg.d_model, d_ff)

    def _compute_lambda(self) -> torch.Tensor:
        return (
            torch.exp((self.lambda_q1 * self.lambda_k1).sum())
            - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
            + self.lambda_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.attn_norm(x)

        # project and split into q1/q2/k1/k2 BEFORE RoPE+GQA  ⟂ REVIEW H1
        q = self.q_proj(h).view(B, T, 2 * self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 2 * self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q1, q2 = q[:, : self.n_heads], q[:, self.n_heads :]
        k1_kv, k2_kv = k[:, : self.n_kv_heads], k[:, self.n_kv_heads :]

        # rope on ALL four (q1, q2, k1, k2) before GQA expansion
        cos, sin = self.rope(T, x.device, x.dtype)
        q1 = apply_rope(q1, cos, sin)
        q2 = apply_rope(q2, cos, sin)
        k1_kv = apply_rope(k1_kv, cos, sin)
        k2_kv = apply_rope(k2_kv, cos, sin)

        # GQA expand keys (and values) up to n_heads
        k1 = gqa_repeat(k1_kv, self.n_rep)
        k2 = gqa_repeat(k2_kv, self.n_rep)
        v = gqa_repeat(v, self.n_rep)

        # two causal softmax attention maps
        a1 = scaled_softmax_attn(q1, k1, v, causal=True)
        a2 = scaled_softmax_attn(q2, k2, v, causal=True)

        lam = self._compute_lambda()
        attn = a1 - lam * a2
        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        attn = self.subln(attn)
        x = x + self.o_proj(attn)

        # ffn
        x = x + self.ffn(self.ffn_norm(x))
        return x
