"""Tests for Tier 2 Titans MAG."""

import torch

from agent.titans import TitansMAG


def test_titans_forward_shape():
    t = TitansMAG(d_model=128, d_hidden=256)
    h = torch.randn(2, 16, 128)
    out = t(h)
    assert out.shape == h.shape


def test_titans_zero_gate_zero_output():
    """Default gate = 0 means sigmoid(0) - 0.5 = 0, so memory contributes nothing."""
    t = TitansMAG(d_model=128, d_hidden=256)
    h = torch.randn(2, 16, 128)
    out = t(h)
    assert out.abs().max().item() < 1e-6


def test_titans_inner_update_drops_surprise():
    """After a single inner update, re-evaluating on the same input should have lower loss."""
    t = TitansMAG(d_model=64, d_hidden=128, eta=1e-2, tau_surprise=0.0)
    h = torch.randn(2, 8, 64) * 5  # high-variance input so surprise > 0
    stats = t.inner_update(h)
    assert stats["updated"] is True
    assert stats["post_surprise"] <= stats["surprise"]


def test_titans_below_threshold_skips_update():
    t = TitansMAG(d_model=32, d_hidden=64, tau_surprise=1e9)
    h = torch.randn(2, 4, 32)
    stats = t.inner_update(h)
    assert stats["updated"] is False
    assert stats["surprise"] == stats["post_surprise"]


def test_titans_gradient_through_gate():
    t = TitansMAG(d_model=32, d_hidden=64)
    with torch.no_grad():
        t.gate.fill_(2.0)  # open the gate manually
    h = torch.randn(2, 4, 32, requires_grad=True)
    out = t(h)
    out.sum().backward()
    assert t.gate.grad is not None
    assert t.gate.grad.abs().sum().item() > 0
