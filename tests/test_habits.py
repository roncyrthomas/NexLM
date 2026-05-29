"""Tests for Tier H — Habits."""

from agent.habits import HabitsCache


def test_no_habits_initially():
    h = HabitsCache()
    assert h.maybe_bypass(intent=1) is None


def test_compiles_after_threshold():
    h = HabitsCache(compile_threshold=3, reward_threshold=0.5)
    for _ in range(3):
        h.observe(intent=1, tool=2, shape=0, reward=1.0, cached_response="cached!")
    rec = h.maybe_bypass(intent=1)
    assert rec is not None
    assert rec.compiled
    assert rec.cached_response == "cached!"


def test_does_not_compile_under_threshold():
    h = HabitsCache(compile_threshold=5)
    for _ in range(2):
        h.observe(intent=1, tool=2, shape=0, reward=1.0)
    assert h.maybe_bypass(intent=1) is None


def test_low_reward_blocks_compilation():
    h = HabitsCache(compile_threshold=3, reward_threshold=0.8)
    for _ in range(5):
        h.observe(intent=1, tool=2, shape=0, reward=0.1)
    assert h.maybe_bypass(intent=1) is None


def test_frustration_inhibits_firing():
    h = HabitsCache(compile_threshold=2, reward_threshold=0.5, frustration_inhibit=0.5)
    h.observe(intent=1, tool=2, shape=0, reward=1.0, cached_response="cached")
    h.observe(intent=1, tool=2, shape=0, reward=1.0, cached_response="cached")
    assert h.maybe_bypass(intent=1, frustration=0.0) is not None
    assert h.maybe_bypass(intent=1, frustration=0.8) is None


def test_decay_demotes_unused():
    h = HabitsCache(compile_threshold=2, decay_max_unused=3)
    for _ in range(2):
        h.observe(intent=1, tool=2, shape=0, reward=1.0, cached_response="cached")
    assert h.maybe_bypass(intent=1) is not None
    for _ in range(5):
        h.tick()
    assert h.maybe_bypass(intent=1) is None


def test_stats_count_compiled():
    h = HabitsCache(compile_threshold=2)
    for _ in range(2):
        h.observe(intent=1, tool=2, shape=0, reward=1.0)
    s = h.stats()
    assert s["compiled"] == 1
    assert s["compiled_intents"] == 1
