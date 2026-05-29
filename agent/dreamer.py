"""Tier D — Dreamer.

Idle-time consolidation. When the agent is fatigued (high updates-since-snapshot),
it enters a dream cycle: sample past situations from episodic memory, replay
them through the agent layer, compute predictive error, and use that error
as a training signal for the LoRA adapters.

This is wake-sleep style consolidation — abstracting patterns from the day's
experience without new external data. Drives the TRACE-BWT claim in the paper.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Dreamer:
    """Wake-sleep consolidation over episodic memory."""

    n_samples_per_dream: int = 64
    inner_lr: float = 1e-4
    max_grad_norm: float = 1.0

    last_dream_stats: dict = None

    def should_dream(self, hormones, threshold: float = 0.7) -> bool:
        return hormones.fatigue >= threshold

    def dream(
        self,
        agent,  # NexAgent
        n_samples: Optional[int] = None,
    ) -> dict:
        """Run a single dream cycle.

        Returns stats: {n_dreamed, avg_predictive_error, lora_grad_norm}.
        """
        if agent.episodic is None or not agent.episodic.episodes:
            return {"n_dreamed": 0, "skipped": "no_episodic_memory"}

        if agent.predictive is None:
            return {"n_dreamed": 0, "skipped": "no_predictive"}

        n = n_samples or self.n_samples_per_dream
        n = min(n, len(agent.episodic.episodes))
        samples = random.sample(agent.episodic.episodes, n)

        # Find trainable params (LoRA adapters)
        trainable = [p for p in agent.parameters() if p.requires_grad]
        if not trainable:
            return {"n_dreamed": 0, "skipped": "no_trainable_params"}

        opt = torch.optim.AdamW(trainable, lr=self.inner_lr)
        device = next(agent.parameters()).device
        total_loss = 0.0

        for ep in samples:
            # Construct an input: a faux conversation ending in the user query
            text = f"User: {ep.query}\nAssistant: {ep.response}"
            ids = agent.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            labels = ids.clone()
            # Only train on the assistant continuation portion
            user_part = agent.tokenizer(
                f"User: {ep.query}\nAssistant: ",
                return_tensors="pt", truncation=True, max_length=512,
            ).input_ids
            user_len = min(user_part.shape[1], labels.shape[1])
            labels[:, :user_len] = -100

            out = agent.base(input_ids=ids, labels=labels)
            (out.loss).backward()
            total_loss += float(out.loss.item())

        torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

        grad_norm = math.sqrt(
            sum(float((p.grad if p.grad is not None else torch.zeros_like(p)).pow(2).sum())
                for p in trainable)
        ) if any(p.grad is not None for p in trainable) else 0.0

        stats = {
            "n_dreamed": n,
            "avg_loss": total_loss / max(1, n),
            "lora_grad_norm": grad_norm,
        }
        # Snapshot the hormone state since we just consolidated
        if agent.hormones is not None:
            agent.hormones.snapshot_taken()
        self.last_dream_stats = stats
        return stats
