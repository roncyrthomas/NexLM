"""Tier 2 — Titans MAG over a frozen base model's hidden states.

Design (pivoted from the original 700M-from-scratch plan):
  - A small MLP runs in parallel with the base model.
  - Captures the base's hidden states via forward hook at a chosen layer.
  - Surprise-gated test-time updates: when ||MLP(WK·h) - WV·h||² > τ,
    take one gradient step on the MLP's own weights (NOT the base).
  - A learned sigmoid gate combines MLP output with the base's hidden state
    before feeding to the lm_head.

Critical implementation notes from adversarial review:
  - Use torch.autograd.grad (not .backward()) to keep the inner update's
    graph isolated from any outer training step.
  - Maintain fp32 master copies of the MLP weights for the inner Adam step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TitansMAG(nn.Module):
    """Standalone Tier 2 memory; instantiated per-NexAgent."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 2048,
        eta: float = 1e-3,
        tau_surprise: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        # Learned associative key/value projections so the objective is not the identity
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model, bias=False),
        )
        # Gate starts at zero so memory contributes nothing initially
        self.gate = nn.Parameter(torch.zeros(d_model))
        # Inner-loop hyperparams (hormones can modulate these at runtime)
        self.eta = eta
        self.tau_surprise = tau_surprise

    # ─── Forward ──────────────────────────────────────────────────────────
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Standard forward: return gated memory output.

        h: (B, T, d_model) hidden states from the base.
        Returns: (B, T, d_model) memory contribution to add to h.
        """
        k = self.WK(h)
        out = self.mlp(k)
        # Center the sigmoid gate at 0 so initial contribution is zero
        gate = torch.sigmoid(self.gate) - 0.5
        return gate * out

    # ─── Surprise-gated test-time update ──────────────────────────────────
    @torch.enable_grad()
    def inner_update(self, h: torch.Tensor) -> dict:
        """Take one surprise-gated update of the MLP weights.

        Safe to call inside a no-grad outer context; we build a tiny isolated
        autograd graph here and only touch self.mlp.parameters().

        Returns diagnostics: {surprise, updated: bool, post_surprise}.
        """
        h_d = h.detach()
        k_d = self.WK(h_d).detach()
        v_d = self.WV(h_d).detach()
        pred = self.mlp(k_d)
        loss = ((pred - v_d) ** 2).mean()
        surprise = float(loss.item())

        if surprise <= self.tau_surprise:
            return {"surprise": surprise, "updated": False, "post_surprise": surprise}

        # Inner gradient — isolated, not connected to any outer loss
        grads = torch.autograd.grad(
            loss,
            list(self.mlp.parameters()),
            retain_graph=False,
            create_graph=False,
        )
        with torch.no_grad():
            for p, g in zip(self.mlp.parameters(), grads):
                # fp32 master update path; cast back to original dtype
                p_fp32 = p.float() - self.eta * g.float()
                p.copy_(p_fp32.to(p.dtype))

        # Re-evaluate surprise after update (diagnostic)
        with torch.no_grad():
            pred_post = self.mlp(k_d)
            post = float(((pred_post - v_d) ** 2).mean().item())
        return {"surprise": surprise, "updated": True, "post_surprise": post}

    # ─── Hooking into a base model ────────────────────────────────────────
    def attach_to_base(self, base_model: nn.Module, layer_index: int = -1) -> "TitansHook":
        """Register a forward hook on a transformer block of the base model.

        After the hook is installed, Titans is automatically applied to the
        layer's output. Returns a handle you can use to remove it.
        """
        # Heuristic: navigate HF causal LM structure to find the transformer layers
        layers = _find_transformer_layers(base_model)
        if layers is None or len(layers) == 0:
            raise ValueError("Could not locate transformer layers on this base model")
        layer = layers[layer_index]

        return TitansHook(self, layer)


class TitansHook:
    """Manages a single forward hook applying TitansMAG to a base model layer."""

    def __init__(self, titans: TitansMAG, layer: nn.Module):
        self.titans = titans
        self.layer = layer
        self.last_inner_update_stats: dict | None = None

        def _hook(module, args, output):
            # output of a transformer block is usually (hidden_states, ...) tuple
            if isinstance(output, tuple):
                h = output[0]
                mem = self.titans(h)
                new = (h + mem,) + output[1:]
                return new
            else:
                return output + self.titans(output)

        self.handle = layer.register_forward_hook(_hook)

    def detach(self) -> None:
        self.handle.remove()

    def inner_update(self, h: torch.Tensor) -> dict:
        stats = self.titans.inner_update(h)
        self.last_inner_update_stats = stats
        return stats


def _find_transformer_layers(model: nn.Module) -> list[nn.Module] | None:
    """Best-effort search for the `nn.ModuleList` of transformer blocks.

    Works for: Llama / SmolLM / Phi-3 / Qwen / Mistral families.
    """
    for path in ("model.layers", "transformer.h", "base_model.model.model.layers"):
        node = model
        ok = True
        for attr in path.split("."):
            if hasattr(node, attr):
                node = getattr(node, attr)
            else:
                ok = False
                break
        if ok and isinstance(node, (nn.ModuleList, list)) and len(node) > 0:
            return list(node)
    return None
