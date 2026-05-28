"""Tests for the differential-attention block."""

import math

import pytest
import torch

from model.config import ModelConfig
from model.diff_attn_block import DiffAttnBlock


def test_diff_attn_forward_shape_cpu():
    cfg = ModelConfig.smoke_30m()
    block = DiffAttnBlock(cfg, layer_idx=2)
    x = torch.randn(2, 32, cfg.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_diff_attn_grad_flows_cpu():
    cfg = ModelConfig.smoke_30m()
    block = DiffAttnBlock(cfg, layer_idx=2)
    x = torch.randn(2, 32, cfg.d_model, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_diff_attn_lambda_depth_dependent():
    cfg = ModelConfig.smoke_30m()
    b0 = DiffAttnBlock(cfg, layer_idx=0)
    b5 = DiffAttnBlock(cfg, layer_idx=5)
    expected_0 = 0.8 - 0.6 * math.exp(-0.3 * 0)
    expected_5 = 0.8 - 0.6 * math.exp(-0.3 * 5)
    assert abs(b0.lambda_init.item() - expected_0) < 1e-5
    assert abs(b5.lambda_init.item() - expected_5) < 1e-5
    assert b0.lambda_init.item() != b5.lambda_init.item()


def test_diff_attn_causal_mask_enforced():
    """Future tokens cannot influence past tokens (causal property)."""
    cfg = ModelConfig.smoke_30m()
    block = DiffAttnBlock(cfg, layer_idx=2).eval()
    x = torch.randn(1, 16, cfg.d_model)
    # perturb only the last position
    x2 = x.clone()
    x2[:, -1] += 5.0
    with torch.no_grad():
        out1 = block(x)
        out2 = block(x2)
    # output at positions [0..-2] should be identical between the two runs
    diff = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
    assert diff < 1e-4, f"causal mask broken: max diff at non-final positions = {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diff_attn_bf16_cuda():
    cfg = ModelConfig.smoke_30m()
    block = DiffAttnBlock(cfg, layer_idx=2).cuda().to(torch.bfloat16)
    x = torch.randn(2, 32, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16
