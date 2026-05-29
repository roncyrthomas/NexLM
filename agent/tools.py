"""Granite-style tool-calling helpers.

Format adopted from IBM Granite-3:
  <tool_call>{"name": "tool_name", "arguments": {"arg1": ...}}</tool_call>
  ... model executes the tool, gets back ...
  <tool_response>{"result": ...}</tool_response>

Parsing is permissive: regex-driven, tolerates whitespace and minor noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class ToolResponse:
    result: Any
    error: str | None = None

    def to_block(self) -> str:
        payload = {"result": self.result}
        if self.error:
            payload["error"] = self.error
        return f"<tool_response>{json.dumps(payload)}</tool_response>"


def parse_tool_calls(model_output: str) -> list[ToolCall]:
    """Extract all <tool_call> blocks from a model output."""
    calls: list[ToolCall] = []
    for m in TOOL_CALL_RE.finditer(model_output):
        try:
            data = json.loads(m.group(1))
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", {}), raw=m.group(0)))
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


def format_tool_descriptions(tools: dict[str, dict]) -> str:
    """Render the tool catalogue as a system-prompt block.

    `tools` maps name → {"description": str, "parameters": {...JSON-schema-like...}}.
    """
    parts = ["You have access to the following tools:"]
    for name, spec in tools.items():
        parts.append(f"  - {name}: {spec.get('description', '')}")
        params = spec.get("parameters", {}).get("properties", {})
        if params:
            parts.append(f"    parameters: {json.dumps(params)}")
    parts.append(
        "To call a tool, output: "
        '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>'
    )
    return "\n".join(parts)


@dataclass
class ToolRegistry:
    """Holds Python callables, exposes them by name with execution + error handling."""

    tools: dict[str, Callable] = None
    specs: dict[str, dict] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = {}
        if self.specs is None:
            self.specs = {}

    def register(self, name: str, fn: Callable, description: str = "", parameters: dict | None = None) -> None:
        self.tools[name] = fn
        self.specs[name] = {
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        }

    def clear(self) -> None:
        """Drop all registered tools. Call between benchmark examples that use
        different tool catalogues so prior turns' tools don't leak into the
        current decision."""
        self.tools.clear()
        self.specs.clear()

    def register_many(self, specs: list[dict], stub: bool = True) -> None:
        """Bulk-register from a list of {name, description, parameters} dicts.

        Used by benchmark adapters. `stub=True` registers a no-op callable —
        appropriate for evals where we only score whether the model picks the
        right tool name, not whether the tool runs.
        """
        for spec in specs:
            name = spec.get("name") or spec.get("function_name") or ""
            if not name:
                continue
            params = spec.get("parameters")
            if params is None:
                params = spec.get("function", {}).get("parameters") if isinstance(spec.get("function"), dict) else None
            if params is None:
                params = {"type": "object", "properties": {}}
            self.register(
                name,
                (lambda **kw: {"stub": True}) if stub else (lambda **kw: None),
                description=spec.get("description", ""),
                parameters=params if isinstance(params, dict) else {"type": "object", "properties": {}},
            )

    def execute(self, call: ToolCall) -> ToolResponse:
        if call.name not in self.tools:
            return ToolResponse(result=None, error=f"unknown tool: {call.name}")
        try:
            result = self.tools[call.name](**call.arguments)
            return ToolResponse(result=result)
        except Exception as e:
            return ToolResponse(result=None, error=f"{type(e).__name__}: {e}")

    def render_descriptions(self) -> str:
        return format_tool_descriptions(self.specs)
