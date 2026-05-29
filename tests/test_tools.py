"""Tests for tool calling format."""

from agent.tools import (
    ToolCall,
    ToolRegistry,
    format_tool_descriptions,
    parse_tool_calls,
)


def test_parse_single_tool_call():
    text = 'I will use a tool. <tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Tokyo"}


def test_parse_two_tool_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call> '
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "a"
    assert calls[1].name == "b"


def test_parse_ignores_malformed_json():
    text = '<tool_call>not json</tool_call>'
    assert parse_tool_calls(text) == []


def test_registry_execute_unknown_tool():
    reg = ToolRegistry()
    result = reg.execute(ToolCall(name="missing", arguments={}))
    assert result.error is not None
    assert "unknown" in result.error.lower()


def test_registry_execute_real_tool():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, description="add two ints")
    result = reg.execute(ToolCall(name="add", arguments={"a": 2, "b": 3}))
    assert result.result == 5
    assert result.error is None


def test_registry_execute_error_caught():
    reg = ToolRegistry()
    reg.register("boom", lambda: 1 / 0, description="fails")
    result = reg.execute(ToolCall(name="boom", arguments={}))
    assert result.error is not None
    assert "ZeroDivisionError" in result.error


def test_render_descriptions_includes_tools():
    reg = ToolRegistry()
    reg.register("foo", lambda: None, description="does foo")
    reg.register("bar", lambda: None, description="does bar")
    s = reg.render_descriptions()
    assert "foo" in s
    assert "bar" in s
    assert "<tool_call>" in s
