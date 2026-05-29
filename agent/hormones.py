"""Tier 0a — Hormone scalars.

Five EMA-tracked scalars define the agent's affective state:
  joy, frustration, confidence, fatigue, boredom.

They are updated from interaction outcomes (user-confirmed positive/negative,
retry signals, prediction entropy, surprise variance) and in turn modulate:
  - LoRA learning rate (Tier 3)
  - Hebbian update strength (Tier 0b)
  - Sampling temperature at inference
  - Titans surprise threshold (Tier 2)
  - Snapshot/rollback triggers
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HormoneState:
    """EMA-based affect tracker. All values clamped to [0, 1]."""

    joy: float = 0.0
    frustration: float = 0.0
    confidence: float = 0.0
    fatigue: float = 0.0
    boredom: float = 0.0

    # EMA smoothing factor: higher = more reactive, lower = more stable
    alpha: float = 0.1

    # Thresholds for derived behaviors
    tau_panic: float = 0.7
    tau_joy: float = 0.3
    tau_frust: float = 0.3

    # Snapshot bookkeeping
    updates_since_snapshot: int = 0
    updates_per_fatigue_unit: int = 1000

    # ─── Updating ─────────────────────────────────────────────────────────
    def update(
        self,
        reward: float = 0.0,
        retry_signal: float = 0.0,
        prediction_entropy: float = 0.5,
        surprise_variance: float = 1.0,
    ) -> None:
        """Single-tick update from one interaction.

        reward            : in [-1, 1]. +1 = user accepted, -1 = user rejected.
        retry_signal      : in [0, 1]. number of retries this turn / max_retries.
        prediction_entropy: in [0, ~log_vocab]. normalized to [0, 1].
        surprise_variance : in [0, inf). higher = model is "surprised" more.
        """
        a = self.alpha
        self.joy = (1 - a) * self.joy + a * max(0.0, reward)
        self.frustration = (
            (1 - a) * self.frustration
            + a * max(0.0, -reward)
            + a * retry_signal
        )
        self.confidence = (1 - a) * self.confidence + a * max(0.0, (1 - prediction_entropy) * reward)
        self.fatigue = min(1.0, self.updates_since_snapshot / self.updates_per_fatigue_unit)
        self.boredom = (1 - a) * self.boredom + a * (1.0 / (surprise_variance + 1e-3))

        # clamp
        for k in ("joy", "frustration", "confidence", "fatigue", "boredom"):
            setattr(self, k, max(0.0, min(1.0, getattr(self, k))))

        self.updates_since_snapshot += 1

    def snapshot_taken(self) -> None:
        """Call when the system has just taken a LoRA snapshot."""
        self.updates_since_snapshot = 0
        self.fatigue = 0.0

    # ─── Derived signals consumers use ───────────────────────────────────
    def lora_lr_multiplier(self, base_lr: float, alpha_joy: float = 1.5, beta_frust: float = 1.0) -> float:
        """Modulate the LoRA learning rate.
        joy ↑ → LR ↑ (reinforce good)
        frustration ↑ → LR ↓ (don't reinforce bad)
        """
        return base_lr * (1.0 + alpha_joy * self.joy - beta_frust * self.frustration)

    def sampling_temperature(self, base_T: float = 0.7, alpha_frust: float = 0.4, beta_conf: float = 0.2) -> float:
        """Frustration → explore more. Confidence → exploit more."""
        T = base_T * (1.0 + alpha_frust * self.frustration - beta_conf * self.confidence)
        return max(0.1, min(2.0, T))

    def titans_surprise_threshold(self, base_tau: float = 0.5, beta_boredom: float = 0.4) -> float:
        """Boredom → lower threshold so Titans updates more eagerly."""
        return max(0.05, base_tau * (1.0 - beta_boredom * self.boredom))

    def should_panic_rollback(self) -> bool:
        """Trigger LoRA rollback when frustration and fatigue both high."""
        return self.frustration > self.tau_panic and self.fatigue > 0.5

    def should_sleep_train(self) -> bool:
        """Trigger sleep-training when fatigue saturates."""
        return self.fatigue >= 0.95

    def update_was_positive(self) -> bool:
        """True iff this turn looks like a clean win → permit LoRA write."""
        return self.joy > self.tau_joy and self.frustration < self.tau_frust

    # ─── Persistence ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "joy": self.joy,
            "frustration": self.frustration,
            "confidence": self.confidence,
            "fatigue": self.fatigue,
            "boredom": self.boredom,
            "updates_since_snapshot": self.updates_since_snapshot,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HormoneState":
        h = cls()
        for k, v in d.items():
            setattr(h, k, v)
        return h
