"""Tier M — Metaplastic.

Per-pair Hebbian learning rate. Tracks whether updates to a feature pair
historically led to positively-reinforced outcomes; raises η for reliable
pairs, lowers η for noisy pairs.

Implementation: a second matrix `eta[i, j]` same shape as the Hebbian H.
After each update, the MetaPlastic controller watches the next K outcomes
and credits the pairs that were updated. Win → η bumps up. Loss → η bumps down.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MetaPlastic:
    eta_base: float = 0.05
    eta_min: float = 0.001
    eta_max: float = 0.5
    alpha_up: float = 0.05      # multiplicative gain on positive outcome
    alpha_down: float = 0.03    # multiplicative shrink on negative outcome
    credit_window: int = 5      # turns over which we credit a recent update

    n_features: int = 850
    eta: Optional[torch.Tensor] = field(default=None, repr=False)
    # Queue of (turn_idx, active_features) pairs to be retro-credited
    pending: deque = field(default_factory=deque)
    turn_counter: int = 0

    def __post_init__(self):
        if self.eta is None:
            self.eta = torch.full(
                (self.n_features, self.n_features), self.eta_base, dtype=torch.float32
            )

    def get_eta(self, i: int, j: int) -> float:
        """Look up per-pair learning rate."""
        return float(self.eta[i, j].item())

    def record_update(self, active_features: list[int]) -> None:
        """Remember that an update occurred for later credit assignment."""
        self.turn_counter += 1
        self.pending.append((self.turn_counter, list(active_features)))
        # Drop entries older than credit_window
        while self.pending and self.turn_counter - self.pending[0][0] > self.credit_window:
            self.pending.popleft()

    def credit_outcome(self, outcome_reward: float) -> int:
        """Update η for every pair in the recent credit window.

        Returns number of pairs adjusted (for diagnostics).
        """
        if not self.pending:
            return 0
        adjusted = 0
        for _, features in self.pending:
            for i_idx, i in enumerate(features):
                for j in features[i_idx + 1 :]:
                    if outcome_reward > 0:
                        self.eta[i, j] = torch.clamp(
                            self.eta[i, j] * (1 + self.alpha_up * outcome_reward),
                            max=self.eta_max,
                        )
                    elif outcome_reward < 0:
                        self.eta[i, j] = torch.clamp(
                            self.eta[i, j] * (1 - self.alpha_down * abs(outcome_reward)),
                            min=self.eta_min,
                        )
                    self.eta[j, i] = self.eta[i, j]
                    adjusted += 1
        return adjusted

    def stats(self) -> dict:
        eta = self.eta
        return {
            "eta_min": float(eta.min()),
            "eta_max": float(eta.max()),
            "eta_mean": float(eta.mean()),
            "n_above_base": int((eta > self.eta_base * 1.1).sum() // 2),
            "n_below_base": int((eta < self.eta_base * 0.9).sum() // 2),
        }
