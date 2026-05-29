"""Tier P — Predictive coding.

The agent predicts the next user input; observing the actual user message
yields a surprise signal that drives self-derived learning. This is the
v2 load-bearing piece — every other v2 tier benefits from a real surprise
signal that doesn't require external reward.

Mechanism:
  1. After the agent's response at turn t, generate a prediction distribution
     for what the user is likely to say at turn t+1, conditioned on the
     conversation so far.
  2. When the user actually responds, compute -log p(actual_first_token | pred).
  3. Normalize via EMA z-score → sigmoid → [0, 1].
  4. Surprise is consumed by HormoneState (via the wrapper's observe_feedback).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PredictiveCoder:
    """Maintains the agent's last next-user-input prediction + EMA stats."""

    ema_alpha: float = 0.1
    surprise_ema_mean: float = 0.0
    surprise_ema_var: float = 1.0

    _last_prediction: Optional[torch.Tensor] = field(default=None, repr=False)
    _last_actual_first_token: Optional[int] = field(default=None, repr=False)

    @torch.no_grad()
    def predict_from_logits(self, next_token_logits: torch.Tensor) -> None:
        """Cache a prediction distribution from already-computed logits.

        next_token_logits: 1-D tensor (vocab,) — logits for the next token.
        """
        self._last_prediction = torch.softmax(next_token_logits.float(), dim=-1).cpu()

    def observe_first_token(self, actual_token_id: int) -> float:
        """Compute normalized surprise of the observed first user token."""
        if self._last_prediction is None:
            return 0.5  # no prior — neutral surprise
        p = float(self._last_prediction[actual_token_id].item())
        raw_surprise = -math.log(max(p, 1e-9))

        # Update EMA stats
        delta = raw_surprise - self.surprise_ema_mean
        self.surprise_ema_mean += self.ema_alpha * delta
        self.surprise_ema_var = (
            (1 - self.ema_alpha) * self.surprise_ema_var
            + self.ema_alpha * delta * delta
        )

        # Z-score → sigmoid for [0, 1]
        z = delta / (math.sqrt(self.surprise_ema_var) + 1e-6)
        self._last_actual_first_token = actual_token_id
        return 1.0 / (1.0 + math.exp(-z))

    def observe_text(self, text: str, tokenizer) -> float:
        """Convenience: tokenize and observe the first token of a text."""
        if not text:
            return 0.5
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.numel() == 0:
            return 0.5
        return self.observe_first_token(int(ids[0, 0].item()))

    def reset(self) -> None:
        """Clear the cached prediction (e.g., at start of new conversation)."""
        self._last_prediction = None
        self._last_actual_first_token = None


def predict_next_user_input(
    base_model,
    tokenizer,
    conversation: list[dict],
    device: Optional[str] = None,
) -> torch.Tensor:
    """Run the base model on a prompt that primes a "User:" turn and return
    the next-token logits.

    `conversation` is a list of {"role": "user"|"assistant", "content": str}.
    Returns 1-D (vocab,) logits tensor.
    """
    parts = []
    for m in conversation:
        role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            m.get("role", "user"), "User"
        )
        parts.append(f"{role}: {m.get('content', '')}")
    parts.append("User: ")
    prompt = "\n".join(parts)

    device = device or next(base_model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = base_model(ids)
    return out.logits[0, -1].detach()
