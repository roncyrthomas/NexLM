"""Tests for the Titans MAG stub."""

import torch

from model.config import ModelConfig
from model.titans_mag import TitansMAG


def test_titans_disabled_returns_zeros():
    cfg = ModelConfig.smoke_30m()
    assert cfg.titans_enabled is False
    mem = TitansMAG(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = mem(x, layer_idx=0)
    assert out.shape == x.shape
    assert out.abs().max().item() == 0.0  # exactly zero


def test_titans_enabled_initial_contribution_is_zero():
    """Even when enabled, untrained gate (sigmoid(0)-0.5=0) should not affect output."""
    cfg = ModelConfig.smoke_30m()
    cfg.titans_enabled = True
    mem = TitansMAG(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    out = mem(x, layer_idx=0)
    assert out.abs().max().item() < 1e-6


def test_titans_enabled_grad_flows_through_gate():
    """If we manually open the gate, gradient must flow through it."""
    cfg = ModelConfig.smoke_30m()
    cfg.titans_enabled = True
    mem = TitansMAG(cfg)
    # open the gate manually
    with torch.no_grad():
        mem.gate.fill_(2.0)  # sigmoid(2) - 0.5 ~= 0.38
    x = torch.randn(2, 16, cfg.d_model, requires_grad=True)
    out = mem(x, layer_idx=0)
    out.sum().backward()
    assert mem.gate.grad is not None
    assert mem.gate.grad.abs().sum().item() > 0
    assert x.grad is not None


def test_titans_has_associative_projections():
    """⟂ REVIEW C2: surprise objective must be associative (WK, WV present)."""
    cfg = ModelConfig.smoke_30m()
    mem = TitansMAG(cfg)
    assert hasattr(mem, "WK")
    assert hasattr(mem, "WV")
    assert isinstance(mem.WK, torch.nn.Linear)
    assert isinstance(mem.WV, torch.nn.Linear)
