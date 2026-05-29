"""Tests for the new AgentConfig Titans per-layer fields."""

from agent.config import AgentConfig


def test_default_titans_layer_indices():
    cfg = AgentConfig()
    assert cfg.titans_layer_indices == [-1]


def test_titans_multi_layer_config():
    cfg = AgentConfig(titans_layer_indices=[0, 8, 16, 24])
    assert cfg.titans_layer_indices == [0, 8, 16, 24]


def test_titans_hyperparams_defaults():
    cfg = AgentConfig()
    assert cfg.titans_d_hidden == 2048
    assert cfg.titans_eta == 1e-3
    assert cfg.titans_tau_surprise == 0.5


def test_titans_hyperparams_overridable():
    cfg = AgentConfig(titans_d_hidden=4096, titans_eta=1e-4, titans_tau_surprise=0.3)
    assert cfg.titans_d_hidden == 4096
    assert cfg.titans_eta == 1e-4
    assert cfg.titans_tau_surprise == 0.3


def test_tools_clear_method_via_registry():
    """Sanity-check on the new ToolRegistry.clear() method."""
    from agent.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register("a", lambda: 1, "desc")
    reg.register("b", lambda: 2, "desc")
    assert len(reg.tools) == 2
    reg.clear()
    assert reg.tools == {}
    assert reg.specs == {}


def test_register_many():
    from agent.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_many([
        {"name": "x", "description": "x", "parameters": {"type": "object", "properties": {}}},
        {"name": "y", "description": "y", "parameters": {"type": "object", "properties": {}}},
    ])
    assert "x" in reg.tools
    assert "y" in reg.tools
