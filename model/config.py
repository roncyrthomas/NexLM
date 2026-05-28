"""Model configuration for the Frankenstein SLM.

Two shapes share the same code path:
- `smoke_30m` — M1a validation model on TinyStories.
- `production_700m` — M1b production model on FineWeb-Edu + friends.

Attention layer positions are 1-indexed (so position 4 = the 4th layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # Backbone
    d_model: int = 1536
    n_layers: int = 24
    attn_layer_positions: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24]
    )

    # Attention
    n_heads: int = 12
    head_dim: int = 128
    n_kv_heads: int = 4
    rope_base: float = 1_000_000.0

    # Mamba2
    mamba_state_dim: int = 128
    mamba_expansion: int = 2
    mamba_conv_kernel: int = 4
    mamba_head_dim: int = 64  # mamba2 internal head_dim

    # FFN
    ffn_expansion: float = 2.67

    # Tokenizer / sequence
    vocab_size: int = 32016  # Phi-3 + 16 specials
    max_seq_len: int = 4096

    # Titans MAG (stubbed in M1a, real in M6)
    titans_enabled: bool = False
    titans_hidden: int = 2048

    @classmethod
    def smoke_30m(cls) -> "ModelConfig":
        return cls(
            d_model=256,
            n_layers=6,
            attn_layer_positions=[3, 6],
            n_heads=4,
            head_dim=64,
            n_kv_heads=2,
            rope_base=10_000.0,
            mamba_state_dim=64,
            mamba_head_dim=32,
            vocab_size=50257,  # GPT-2 BPE for TinyStories
            max_seq_len=512,
            titans_enabled=False,
        )

    @classmethod
    def production_700m(cls) -> "ModelConfig":
        return cls()  # defaults already match the production target
