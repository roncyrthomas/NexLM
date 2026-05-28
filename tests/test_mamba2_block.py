"""Tests for the Mamba2 block: forward shape, gradient flow, bf16 on CUDA."""

import pytest
import torch

from model.config import ModelConfig
from model.mamba2_block import Mamba2Block


def test_mamba2_forward_shape_cpu():
    cfg = ModelConfig.smoke_30m()
    block = Mamba2Block(cfg, layer_idx=0)
    x = torch.randn(2, 32, cfg.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_mamba2_grad_flows_cpu():
    cfg = ModelConfig.smoke_30m()
    block = Mamba2Block(cfg, layer_idx=0)
    x = torch.randn(2, 32, cfg.d_model, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mamba2_bf16_cuda():
    cfg = ModelConfig.smoke_30m()
    block = Mamba2Block(cfg, layer_idx=0).cuda().to(torch.bfloat16)
    x = torch.randn(2, 32, cfg.d_model, device="cuda", dtype=torch.bfloat16)
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


def test_mamba2_param_count_reasonable():
    cfg = ModelConfig.smoke_30m()
    block = Mamba2Block(cfg, layer_idx=0)
    n = sum(p.numel() for p in block.parameters())
    # rough sanity: ~1-2M params for d_model=256 mamba2 + swiglu
    assert 500_000 < n < 3_000_000, f"params={n:,}"
