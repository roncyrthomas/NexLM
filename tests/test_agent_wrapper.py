"""Tests for the NexAgent wrapper.

Two tiers:
- Fast (default) — config + dataclass behavior, no model download.
- Slow (`-m slow`) — full model load + LoRA attach + 4-token generation.
  Marked slow because first run downloads SmolLM2-1.7B (~3.4 GB) and runs forward.
"""

import pytest
import torch

from agent.config import AgentConfig
from agent.wrapper import NexAgent


# ────────────────────────────────────────────────────────────────────────
# Fast tests — config defaults
# ────────────────────────────────────────────────────────────────────────
def test_default_is_smollm2_local():
    cfg = AgentConfig()
    assert cfg.base_model_name == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert cfg.base_dtype == "bfloat16"
    assert cfg.quantization is None
    assert cfg.lora_r == 16


def test_smollm2_local_preset():
    cfg = AgentConfig.smollm2_local()
    assert cfg.base_model_name.endswith("SmolLM2-1.7B-Instruct")
    assert "q_proj" in cfg.lora_target_modules
    assert "o_proj" in cfg.lora_target_modules


def test_phi3_cloud_preset():
    cfg = AgentConfig.phi3_cloud()
    assert cfg.base_model_name == "microsoft/Phi-3-mini-4k-instruct"
    assert cfg.trust_remote_code is True
    # Phi-3 has fused QKV, so target module names differ
    assert "qkv_proj" in cfg.lora_target_modules


def test_tinyllama_smoke_preset():
    cfg = AgentConfig.tinyllama_smoke()
    assert cfg.base_model_name.endswith("TinyLlama-1.1B-Chat-v1.0")


def test_all_tier_flags_default_off():
    """In P1, no agent-layer tiers are enabled. They turn on incrementally in P2+."""
    cfg = AgentConfig()
    assert cfg.enable_hormones is False
    assert cfg.enable_hebbian is False
    assert cfg.enable_hipporag is False
    assert cfg.enable_titans is False
    assert cfg.train_lora_online is False


# ────────────────────────────────────────────────────────────────────────
# Slow tests — actually load the model
# ────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU recommended; CPU too slow")
def test_agent_loads_smollm2_and_attaches_lora():
    """End-to-end: build agent, verify LoRA-only trainable params, run 4-token generation."""
    cfg = AgentConfig.smollm2_local()
    agent = NexAgent(cfg).cuda()

    total, trainable = agent.count_params()
    # SmolLM2-1.7B ~ 1.7B params total; LoRA r=16 on q/k/v/o adds ~3-5M
    assert total > 1_500_000_000, f"base load looks wrong: {total:,}"
    assert 1_000_000 < trainable < 30_000_000, f"trainable should be LoRA-only: {trainable:,}"
    # Trainable should be a tiny fraction of total
    assert trainable / total < 0.02, f"trainable fraction too high: {trainable/total:.4f}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU recommended")
def test_agent_generation_smoke():
    """Generate a few tokens to confirm the full pipeline works."""
    cfg = AgentConfig.smollm2_local()
    cfg.max_new_tokens = 8
    agent = NexAgent(cfg).cuda()
    out = agent.generate("Hello, how are", max_new_tokens=8, do_sample=False)
    assert isinstance(out, str)
    assert len(out) > len("Hello, how are")
