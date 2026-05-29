"""Tier 0b — Hebbian co-firing matrix H.

Sparse-feature dense matrix H[i, j] strengthened when features i and j co-fire
in a positively-reinforced interaction. Provides fast, interpretable,
instantly-reversible learning of associations (e.g., task_intent ↔ tool_used).

At inference, biases tool-selection logits proportional to the row
H[active_task_intent, :].

Storage: dense float32, |F|×|F|. For |F|=850 that's 2.7 MB — trivial.
PyTorch's sparse_coo_tensor doesn't support in-place indexing so we use dense.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class HebbianH:
    n_features: int = 850
    eta: float = 0.05       # base update rate
    decay: float = 0.005    # frustration-driven decay
    cap: float = 5.0        # absolute cap to prevent rut behaviors
    lam: float = 0.1        # logit-bias scaling at inference

    H: torch.Tensor = field(default=None, repr=False)

    def __post_init__(self):
        if self.H is None:
            self.H = torch.zeros(self.n_features, self.n_features, dtype=torch.float32)

    # ─── Updating ─────────────────────────────────────────────────────────
    def update(self, active_features: list[int], joy: float, frustration: float, tau_joy: float = 0.3, tau_frust: float = 0.3) -> int:
        """Three-factor Hebbian rule: pre × post × neuromodulator.

        Returns the number of (i, j) pairs modified (for diagnostics).
        """
        n_changed = 0
        if joy > tau_joy and len(active_features) >= 2:
            delta = self.eta * joy
            for i, j in itertools.combinations(active_features, 2):
                self.H[i, j] = torch.clamp(self.H[i, j] + delta, max=self.cap)
                self.H[j, i] = self.H[i, j]  # keep symmetric
                n_changed += 1
        if frustration > tau_frust:
            self.H *= 1.0 - self.decay * frustration
        return n_changed

    # ─── Inference-time bias ──────────────────────────────────────────────
    def bias_logits(self, task_intent_id: int, tool_embeddings: torch.Tensor) -> torch.Tensor:
        """Add Hebbian bias to tool-selection logits.

        Args:
            task_intent_id: cluster id of the user's current intent (0..n_features).
            tool_embeddings: (n_tools, d) — vectors representing each tool.

        Returns:
            (n_tools,) additive bias to add to a tool-selection logits.
        """
        row = self.H[task_intent_id]  # (n_features,)
        # If the embedding matrix is smaller than n_features, slice the row to match.
        n_tools = tool_embeddings.shape[0]
        if row.shape[0] >= n_tools:
            row = row[:n_tools]
        else:
            # pad with zeros if features < tools
            pad = torch.zeros(n_tools - row.shape[0], dtype=row.dtype, device=row.device)
            row = torch.cat([row, pad])
        # Lightweight version: directly use Hebbian strength as logit boost (no embedding projection)
        return self.lam * row

    # ─── Inspection ───────────────────────────────────────────────────────
    def top_associations(self, feature_id: int, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k features most strongly associated with `feature_id`."""
        row = self.H[feature_id]
        values, indices = torch.topk(row, k=min(k, row.shape[0]))
        return [(int(i), float(v)) for i, v in zip(indices, values) if v > 0]

    def stats(self) -> dict:
        return {
            "n_features": self.n_features,
            "nonzero": int((self.H > 0).sum()),
            "max": float(self.H.max()),
            "mean_active": float(self.H[self.H > 0].mean()) if (self.H > 0).any() else 0.0,
        }

    # ─── Persistence ─────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "H": self.H,
                "n_features": self.n_features,
                "eta": self.eta,
                "decay": self.decay,
                "cap": self.cap,
                "lam": self.lam,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HebbianH":
        d = torch.load(str(path), weights_only=False)
        h = cls(
            n_features=d["n_features"],
            eta=d["eta"],
            decay=d["decay"],
            cap=d["cap"],
            lam=d["lam"],
        )
        h.H = d["H"]
        return h


# ─── Feature space convention (used to pack/unpack feature ids) ───────────
# Default feature space (|F| = 850):
#   task_intent : ids 0..511    (k-means cluster id over query embedding)
#   tools_used  : ids 512..575  (64 tool slots)
#   retrieval   : ids 576..831  (256 HippoRAG community ids)
#   response    : ids 832..847  (16 response-shape categories)
TASK_INTENT_RANGE = (0, 512)
TOOLS_USED_RANGE = (512, 576)
RETRIEVAL_RANGE = (576, 832)
RESPONSE_RANGE = (832, 848)


def encode_features(
    task_intent_id: int | None = None,
    tools_used_ids: list[int] | None = None,
    retrieval_ids: list[int] | None = None,
    response_shape_id: int | None = None,
) -> list[int]:
    """Pack a turn's features into the |F|=850 id space."""
    out: list[int] = []
    if task_intent_id is not None:
        out.append(TASK_INTENT_RANGE[0] + task_intent_id)
    if tools_used_ids:
        out.extend(TOOLS_USED_RANGE[0] + i for i in tools_used_ids)
    if retrieval_ids:
        out.extend(RETRIEVAL_RANGE[0] + i for i in retrieval_ids)
    if response_shape_id is not None:
        out.append(RESPONSE_RANGE[0] + response_shape_id)
    return out
