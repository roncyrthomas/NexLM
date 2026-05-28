"""Titans MAG (Memory-as-Gating) module — STUB version for M1a.

This stub establishes the wiring: WK/WV projections, MLP, layer embedding, gate.
In M1a the module always returns zeros (titans_enabled=False) so it is a
no-op during smoke training. M6 replaces the forward path with the real
surprise-gated test-time update logic.

The non-identity associative objective (WK/WV present) is locked in here per
the adversarial review (⟂ REVIEW C2): when M6 turns on the inner update, the
surprise metric will be `||MLP(WK·x) − WV·x||²` rather than `||MLP(x) − x||²`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.config import ModelConfig


class TitansMAG(nn.Module):
    """Tier 2 memory module. M1a stub: always returns zeros."""

    LAYER_EMBED_DIM = 16

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # learned key/value projections — make the objective associative, not identity
        self.WK = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.WV = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model + self.LAYER_EMBED_DIM, cfg.titans_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.titans_hidden, cfg.d_model, bias=False),
        )
        self.layer_embed = nn.Embedding(cfg.n_layers, self.LAYER_EMBED_DIM)
        # gate is a learned per-channel sigmoid; initialized to zero so memory contributes nothing
        self.gate = nn.Parameter(torch.zeros(cfg.d_model))
        # inner-loop hyperparams (used only when M6 enables the real path)
        self.tau_surprise = 0.5
        self.eta_titans = 1e-3

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        allow_inner_update: bool = False,
    ) -> torch.Tensor:
        """Stub forward — returns zeros when titans_enabled is False.

        When enabled and gate is non-zero, returns gate * MLP([WK x, layer_embed]).
        """
        if not self.cfg.titans_enabled:
            return torch.zeros_like(x)
        B, T, _ = x.shape
        k = self.WK(x)
        le = self.layer_embed(torch.tensor(layer_idx, device=x.device)).to(x.dtype)
        le = le.expand(B, T, -1)
        pred = self.mlp(torch.cat([k, le], dim=-1))
        # gate starts at zero -> sigmoid(0)=0.5 ... wait, we want 0 contribution at start.
        # Solution: pass gate through (sigmoid(g) - 0.5) so init contributes nothing, then
        # let learning push it up if memory is useful.
        gate = torch.sigmoid(self.gate) - 0.5
        return gate * pred
