"""Tests for Tier D — Dreamer.

Most of these are unit tests on the gate logic; the real LoRA-update test
is in the slow category since it needs the base model loaded.
"""

import pytest

from agent.dreamer import Dreamer
from agent.hormones import HormoneState


def test_should_dream_when_fatigued():
    d = Dreamer()
    h = HormoneState()
    h.fatigue = 0.9
    assert d.should_dream(h) is True


def test_should_not_dream_when_rested():
    d = Dreamer()
    h = HormoneState()
    h.fatigue = 0.1
    assert d.should_dream(h) is False


def test_dream_skips_without_episodic_memory():
    """No tracked episodes → dream is a no-op."""
    d = Dreamer()

    # Minimal stand-in: agent has the right attributes set to None
    class _A:
        episodic = None
        predictive = None
    out = d.dream(_A())
    assert out["n_dreamed"] == 0


@pytest.mark.slow
def test_dream_actually_steps_lora():
    """Requires base model load. Verifies dream cycle changes LoRA weights."""
    import torch
    from agent.config import AgentConfig
    from agent.wrapper import NexAgent

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    cfg = AgentConfig.frank_v2()
    agent = NexAgent(cfg).cuda()
    # Add some episodes
    for i in range(8):
        agent.episodic.remember(f"q{i}", f"r{i}", reward=1.0)

    before = next(p.detach().clone() for p in agent.parameters() if p.requires_grad)
    stats = agent.dreamer.dream(agent, n_samples=4)
    after = next(p for p in agent.parameters() if p.requires_grad)
    assert stats["n_dreamed"] == 4
    assert (before - after).abs().max().item() > 0
