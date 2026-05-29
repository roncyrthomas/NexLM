"""Tests for Tier M — Metaplastic."""

from agent.metaplastic import MetaPlastic


def test_default_eta_uniform():
    mp = MetaPlastic(n_features=10, eta_base=0.05)
    assert abs(mp.get_eta(0, 1) - 0.05) < 1e-6


def test_positive_outcome_raises_eta():
    mp = MetaPlastic(n_features=10, eta_base=0.05, alpha_up=0.5)
    mp.record_update([0, 1])
    mp.credit_outcome(1.0)
    assert mp.get_eta(0, 1) > 0.05


def test_negative_outcome_lowers_eta():
    mp = MetaPlastic(n_features=10, eta_base=0.05, alpha_down=0.5)
    mp.record_update([0, 1])
    mp.credit_outcome(-1.0)
    assert mp.get_eta(0, 1) < 0.05


def test_clamp_bounds_eta():
    mp = MetaPlastic(n_features=5, eta_base=0.1, eta_min=0.05, eta_max=0.2, alpha_up=10.0)
    mp.record_update([0, 1])
    for _ in range(50):
        mp.record_update([0, 1])
        mp.credit_outcome(1.0)
    assert mp.get_eta(0, 1) <= 0.2 + 1e-6


def test_credit_window_drops_old_updates():
    mp = MetaPlastic(n_features=5, credit_window=2)
    mp.record_update([0, 1])
    mp.record_update([])
    mp.record_update([])
    mp.record_update([])  # turn 4 — initial update at turn 1 should be dropped
    initial_eta = mp.get_eta(0, 1)
    mp.credit_outcome(1.0)
    # Should NOT have adjusted (0,1) because that update is out of window
    assert abs(mp.get_eta(0, 1) - initial_eta) < 1e-6


def test_stats_report():
    # Use a larger alpha_up so a single credit clearly pushes eta above the
    # `> base*1.1` threshold used in stats().
    mp = MetaPlastic(n_features=5, alpha_up=0.5)
    mp.record_update([0, 1])
    mp.credit_outcome(1.0)
    s = mp.stats()
    assert "eta_min" in s
    assert "eta_max" in s
    assert s["n_above_base"] >= 1
