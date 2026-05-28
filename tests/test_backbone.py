"""End-to-end backbone tests: param count, forward shape, loss, generation."""

import pytest
import torch

from model.backbone import Frankenstein
from model.config import ModelConfig
from model.diff_attn_block import DiffAttnBlock
from model.mamba2_block import Mamba2Block


def test_smoke_30m_param_count():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    n = sum(p.numel() for p in model.parameters())
    # tied embed/lm_head; M1a "30M" is approximate; actual is dominated by 50257*256
    # embed (~13M) + 6 layers (~7M). Tolerate 15-35M.
    assert 15_000_000 < n < 35_000_000, f"param count out of range: {n:,}"


def test_smoke_layer_pattern():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    # positions 3 and 6 should be DiffAttn (1-indexed); others Mamba2
    expected = [Mamba2Block, Mamba2Block, DiffAttnBlock,
                Mamba2Block, Mamba2Block, DiffAttnBlock]
    for layer, expected_cls in zip(model.layers, expected):
        assert isinstance(layer, expected_cls), (
            f"layer type mismatch: got {type(layer).__name__}, expected {expected_cls.__name__}"
        )


def test_forward_logits_shape():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_forward_with_targets_returns_loss():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(ids, targets=targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    # untrained -log(1/vocab) = log(50257) ~= 10.8; should be in that ballpark
    import math
    expected_initial_loss = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected_initial_loss) < 1.5, (
        f"untrained loss {loss.item():.2f} far from expected {expected_initial_loss:.2f}"
    )


def test_tied_embeddings():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_generate_runs():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=8, temperature=1.0, top_k=10)
    assert out.shape == (1, 12)
    assert out[0, :4].tolist() == ids[0].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backbone_bf16_cuda_full_forward_backward():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg).cuda().to(torch.bfloat16)
    ids = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
    targets = torch.randint(0, cfg.vocab_size, (2, 32), device="cuda")
    _, loss = model(ids, targets=targets)
    loss.backward()
    # at least one param must have non-zero gradient
    total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert total_grad > 0


# ----------------------------------------------------------------------
# M1b — production_700m shape tests
# ----------------------------------------------------------------------

def test_production_700m_layer_pattern():
    """24 layers, Samba 3:1: DiffAttn at positions 4, 8, 12, 16, 20, 24."""
    cfg = ModelConfig.production_700m()
    model = Frankenstein(cfg)
    attn_positions = {4, 8, 12, 16, 20, 24}
    for i, layer in enumerate(model.layers):
        pos = i + 1
        expected = DiffAttnBlock if pos in attn_positions else Mamba2Block
        assert isinstance(layer, expected), (
            f"layer {pos}: got {type(layer).__name__}, expected {expected.__name__}"
        )


def test_production_700m_param_count():
    """Spec called it 700M; actual is ~830M with SwiGLU expansion 2.67x.

    Still well within "small LM" territory (Phi-1.5 = 1.3B, SmolLM2 = 1.7B).
    Accepting 700M-900M as the production shape. If we want closer to 700M,
    we can drop ffn_expansion to 2.0 in M3.
    """
    cfg = ModelConfig.production_700m()
    model = Frankenstein(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 700_000_000 < n < 900_000_000, f"production shape out of range: {n:,}"


def test_production_700m_forward_cpu_tiny_batch():
    """Just verify it forward-passes on CPU with tiny input.

    Actual perf testing happens on GPU; this is just a shape sanity.
    """
    cfg = ModelConfig.production_700m()
    model = Frankenstein(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits = model(ids)
    assert logits.shape == (1, 4, cfg.vocab_size)
