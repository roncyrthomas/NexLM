"""Tests for the BFCL → ToolRegistry adapter and helpers."""

from agent.tools import ToolRegistry
from evals.bfcl import _extract_expected_tool, bfcl_to_registry


def test_bfcl_plain_format():
    reg = ToolRegistry()
    bfcl = [{"name": "get_weather", "description": "weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    bfcl_to_registry(bfcl, reg)
    assert "get_weather" in reg.tools
    assert reg.specs["get_weather"]["description"] == "weather"


def test_bfcl_openai_function_format():
    reg = ToolRegistry()
    bfcl = [{"type": "function",
             "function": {"name": "calc", "description": "math",
                          "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}}]
    bfcl_to_registry(bfcl, reg)
    assert "calc" in reg.tools
    assert reg.specs["calc"]["description"] == "math"


def test_bfcl_shorthand_typed_params():
    """Some user-provided BFCL examples use {arg: type_string} shorthand."""
    reg = ToolRegistry()
    bfcl = [{"name": "send", "description": "send msg",
             "parameters": {"to": "string", "body": "string"}}]
    bfcl_to_registry(bfcl, reg)
    assert "send" in reg.tools
    props = reg.specs["send"]["parameters"]["properties"]
    assert "to" in props
    assert props["to"]["type"] == "string"


def test_bfcl_registry_clears_between_examples():
    """Critical: prior example's tools must not leak into the next."""
    reg = ToolRegistry()
    bfcl_to_registry([{"name": "a", "description": "", "parameters": {}}], reg)
    assert "a" in reg.tools
    bfcl_to_registry([{"name": "b", "description": "", "parameters": {}}], reg)
    assert "a" not in reg.tools
    assert "b" in reg.tools


def test_extract_expected_tool_v3_groundtruth_list():
    item = {"ground_truth": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}
    assert _extract_expected_tool(item) == "get_weather"


def test_extract_expected_tool_string_form():
    item = {"ground_truth": "get_weather(city='Tokyo')"}
    assert _extract_expected_tool(item) == "get_weather"


def test_extract_expected_tool_v2_answer():
    item = {"answer": {"name": "calc"}}
    assert _extract_expected_tool(item) == "calc"


def test_extract_expected_tool_our_fallback():
    item = {"expected_tool": "send_email"}
    assert _extract_expected_tool(item) == "send_email"


def test_extract_expected_tool_returns_none_when_missing():
    assert _extract_expected_tool({"query": "no gt here"}) is None
