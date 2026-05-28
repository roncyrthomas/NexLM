"""Tests for model.config — both smoke and production shapes."""

from model.config import ModelConfig


def test_smoke_30m_shape():
    cfg = ModelConfig.smoke_30m()
    assert cfg.d_model == 256
    assert cfg.n_layers == 6
    assert cfg.attn_layer_positions == [3, 6]
    assert cfg.n_heads == 4
    assert cfg.head_dim == 64
    assert cfg.n_kv_heads == 2
    assert cfg.vocab_size == 50257
    assert cfg.max_seq_len == 512
    assert cfg.titans_enabled is False


def test_smoke_attn_layers_count():
    cfg = ModelConfig.smoke_30m()
    # 6 layers, 2 attn => 4 mamba
    assert len(cfg.attn_layer_positions) == 2
    assert cfg.n_layers - len(cfg.attn_layer_positions) == 4


def test_production_700m_shape():
    cfg = ModelConfig.production_700m()
    assert cfg.d_model == 1536
    assert cfg.n_layers == 24
    assert cfg.attn_layer_positions == [4, 8, 12, 16, 20, 24]
    assert cfg.n_heads == 12
    assert cfg.head_dim == 128
    assert cfg.n_kv_heads == 4
    assert cfg.rope_base == 1_000_000.0


def test_production_3to1_ratio():
    cfg = ModelConfig.production_700m()
    # Samba-style 3:1 SSM:Attn => 18 Mamba2 + 6 DiffAttn
    assert len(cfg.attn_layer_positions) == 6
    assert cfg.n_layers - len(cfg.attn_layer_positions) == 18
