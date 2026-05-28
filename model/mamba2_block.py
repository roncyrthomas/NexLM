"""Mamba2 block — wraps HuggingFace's Mamba2Mixer with our pre-norm + SwiGLU FFN.

The mixer itself runs in pure-PyTorch fallback on Windows (no mamba_ssm),
and automatically activates the fast CUDA kernel on Linux when mamba_ssm
is installed. Identical interface either way — our wrapper is platform-agnostic.

Architecture per layer (pre-norm residual):
    x -> RMSNorm -> Mamba2Mixer -> + residual -> RMSNorm -> SwiGLU FFN -> + residual
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mamba2.configuration_mamba2 import Mamba2Config as HFMamba2Config
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

from model.config import ModelConfig


def _build_hf_mamba2_config(cfg: ModelConfig) -> HFMamba2Config:
    """Translate our ModelConfig fields into the HuggingFace Mamba2Config shape."""
    d_inner = cfg.d_model * cfg.mamba_expansion
    assert d_inner % cfg.mamba_head_dim == 0, (
        f"d_inner ({d_inner}) must be divisible by mamba_head_dim ({cfg.mamba_head_dim})"
    )
    num_heads = d_inner // cfg.mamba_head_dim
    return HFMamba2Config(
        num_heads=num_heads,
        head_dim=cfg.mamba_head_dim,
        hidden_size=cfg.d_model,
        state_size=cfg.mamba_state_dim,
        expand=cfg.mamba_expansion,
        conv_kernel=cfg.mamba_conv_kernel,
        n_groups=1,  # small models work better with n_groups=1
        chunk_size=64,
        use_cache=False,
        rms_norm=True,
        residual_in_fp32=True,
    )


class SwiGLU(nn.Module):
    """SwiGLU FFN: y = W3(silu(W1 x) * W2 x)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Mamba2Block(nn.Module):
    """Single Mamba2 block: pre-norm, mixer, pre-norm, FFN, residuals."""

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        hf_cfg = _build_hf_mamba2_config(cfg)
        self.mixer_norm = nn.RMSNorm(cfg.d_model)
        self.mixer = Mamba2Mixer(hf_cfg, layer_idx=layer_idx)
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        d_ff = int(cfg.d_model * cfg.ffn_expansion)
        # round d_ff to a multiple of 8 for hardware-friendliness
        d_ff = ((d_ff + 7) // 8) * 8
        self.ffn = SwiGLU(cfg.d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mixer path
        h = self.mixer_norm(x)
        mixer_out = self.mixer(h)
        if isinstance(mixer_out, tuple):
            mixer_out = mixer_out[0]
        x = x + mixer_out
        # ffn path
        x = x + self.ffn(self.ffn_norm(x))
        return x
