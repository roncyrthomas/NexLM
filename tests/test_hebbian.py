"""Tests for Tier 0b Hebbian co-firing matrix."""

import torch

from agent.hebbian import HebbianH, encode_features, TASK_INTENT_RANGE, TOOLS_USED_RANGE


def test_default_zero_matrix():
    h = HebbianH(n_features=10)
    assert h.H.shape == (10, 10)
    assert h.H.sum().item() == 0.0


def test_joy_strengthens_pair():
    h = HebbianH(n_features=10, eta=0.1)
    n_changed = h.update(active_features=[0, 1], joy=1.0, frustration=0.0)
    assert n_changed == 1
    assert h.H[0, 1].item() > 0
    assert h.H[1, 0].item() > 0
    assert h.H[0, 1].item() == h.H[1, 0].item()  # symmetric


def test_frustration_decays_matrix():
    h = HebbianH(n_features=5, eta=1.0, decay=0.1)
    h.update(active_features=[0, 1, 2], joy=1.0, frustration=0.0)
    before = h.H[0, 1].item()
    h.update(active_features=[], joy=0.0, frustration=1.0)
    after = h.H[0, 1].item()
    assert after < before


def test_cap_prevents_runaway():
    h = HebbianH(n_features=5, eta=10.0, cap=2.0)
    for _ in range(20):
        h.update(active_features=[0, 1], joy=1.0, frustration=0.0)
    assert h.H[0, 1].item() <= 2.0


def test_bias_logits_shape():
    h = HebbianH(n_features=10)
    h.update(active_features=[0, 1], joy=1.0, frustration=0.0)
    tool_embs = torch.zeros(5, 4)
    bias = h.bias_logits(task_intent_id=0, tool_embeddings=tool_embs)
    assert bias.shape == (5,)


def test_top_associations_after_training():
    h = HebbianH(n_features=10, eta=1.0)
    h.update([0, 5], joy=1.0, frustration=0.0)
    h.update([0, 5], joy=1.0, frustration=0.0)
    top = h.top_associations(0, k=3)
    assert top[0][0] == 5
    assert top[0][1] > 0


def test_feature_encoding_ranges():
    fids = encode_features(task_intent_id=3, tools_used_ids=[1, 2], response_shape_id=5)
    assert fids[0] == TASK_INTENT_RANGE[0] + 3
    assert TOOLS_USED_RANGE[0] + 1 in fids
    assert TOOLS_USED_RANGE[0] + 2 in fids


def test_save_load_roundtrip(tmp_path):
    h = HebbianH(n_features=8)
    h.update([0, 1, 2], joy=1.0, frustration=0.0)
    path = tmp_path / "hebbian.pt"
    h.save(path)
    h2 = HebbianH.load(path)
    assert torch.allclose(h.H, h2.H)
    assert h.n_features == h2.n_features
