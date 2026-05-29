"""Tests for the three-way compare runner + new AgentConfig presets."""

from agent.config import AgentConfig
from evals.runner import BENCHMARK_REGISTRY, compare_three


def test_vanilla_preset_all_tiers_off():
    cfg = AgentConfig.vanilla()
    assert not cfg.enable_hormones
    assert not cfg.enable_hebbian
    assert not cfg.enable_hipporag
    assert not cfg.enable_titans
    assert not cfg.enable_predictive
    assert not cfg.enable_episodic
    assert not cfg.enable_habits
    assert not cfg.enable_dreamer
    assert not cfg.enable_metaplastic


def test_frank_v1_preset_v1_on_v2_off():
    cfg = AgentConfig.frank_v1()
    # v1 tiers ON
    assert cfg.enable_hormones
    assert cfg.enable_hebbian
    assert cfg.enable_hipporag
    assert cfg.enable_titans
    assert cfg.train_lora_online
    # v2 tiers OFF
    assert not cfg.enable_predictive
    assert not cfg.enable_episodic
    assert not cfg.enable_habits
    assert not cfg.enable_dreamer
    assert not cfg.enable_metaplastic


def test_frank_v2_preset_all_tiers_on():
    cfg = AgentConfig.frank_v2()
    for flag in (
        "enable_hormones", "enable_hebbian", "enable_hipporag", "enable_titans",
        "enable_predictive", "enable_episodic", "enable_habits",
        "enable_dreamer", "enable_metaplastic",
    ):
        assert getattr(cfg, flag), f"{flag} should be True in frank_v2 preset"


def test_preset_base_override():
    cfg = AgentConfig.frank_v2(base="microsoft/Phi-3-mini-4k-instruct")
    assert cfg.base_model_name == "microsoft/Phi-3-mini-4k-instruct"
    assert cfg.enable_predictive  # still v2


def test_compare_three_with_mock_agents():
    """Compare runner works structurally even with stub agents and inline fallbacks."""

    # Register a tiny fake benchmark that doesn't need a real model
    @register_mock("mock_bench")
    def _bench(agent, n):
        return {"score": 0.5 + getattr(agent, "_test_marker", 0.0)}

    class FakeAgent:
        def __init__(self, marker):
            self._test_marker = marker
            self.tools = type("R", (), {"clear": lambda self: None, "tools": {}})()

    a, b, c = FakeAgent(0.0), FakeAgent(0.1), FakeAgent(0.2)
    res = compare_three(a, b, c, benchmarks=["mock_bench"], max_examples=1)
    assert "delta_v2_vs_v1" in res
    assert "delta_v1_vs_vanilla" in res
    assert abs(res["delta_v1_vs_vanilla"]["mock_bench"]["score"] - 0.1) < 1e-6
    assert abs(res["delta_v2_vs_v1"]["mock_bench"]["score"] - 0.1) < 1e-6

    # cleanup the mock registration
    del BENCHMARK_REGISTRY["mock_bench"]


def register_mock(name):
    """Tiny helper to register a mock benchmark without using the decorator import."""
    def deco(fn):
        BENCHMARK_REGISTRY[name] = fn
        return fn
    return deco
