"""BFCL v3 — Berkeley Function Calling Leaderboard.

Tests the agent's ability to:
  - Choose the right tool from a catalogue
  - Produce well-formed JSON arguments

We implement two scoring metrics:
  - format_score: fraction of outputs that successfully parse as a tool call
  - tool_match : fraction of outputs that picked the correct tool name

This is a simplified harness; the full BFCL v3 also has multi-turn and
executable tracks. Our P5 paper claim uses the simple track.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.tools import parse_tool_calls
from agent.wrapper import NexAgent
from evals.runner import register


# Minimal self-contained eval set — for the real BFCL load from HF
_FALLBACK_EVALSET = [
    {
        "tools": [
            {"name": "get_weather", "description": "Get current weather", "parameters": {"city": "string"}},
            {"name": "send_email", "description": "Send an email", "parameters": {"to": "string", "body": "string"}},
        ],
        "query": "What's the weather like in Tokyo right now?",
        "expected_tool": "get_weather",
    },
    {
        "tools": [
            {"name": "get_weather", "description": "Get current weather", "parameters": {"city": "string"}},
            {"name": "send_email", "description": "Send an email", "parameters": {"to": "string", "body": "string"}},
        ],
        "query": "Send Alice an email saying I'll be late.",
        "expected_tool": "send_email",
    },
    {
        "tools": [
            {"name": "calc", "description": "Evaluate a math expression", "parameters": {"expr": "string"}},
        ],
        "query": "What is 17 times 23?",
        "expected_tool": "calc",
    },
]


def _load_eval_set(max_examples: int) -> list[dict]:
    """Try HF first; fall back to the inline set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train", streaming=True)
        out = []
        for i, row in enumerate(ds):
            if i >= max_examples:
                break
            out.append(row)
        return out if out else _FALLBACK_EVALSET
    except Exception:
        return _FALLBACK_EVALSET[:max_examples]


@register("bfcl")
def evaluate(agent: NexAgent, max_examples: int = 100) -> dict:
    items = _load_eval_set(max_examples)
    n = len(items)
    format_ok = 0
    tool_ok = 0
    for item in items:
        # Build the prompt: register tools then ask
        for tool_spec in item.get("tools", []):
            agent.tools.register(
                tool_spec["name"],
                lambda **kw: {"stub": True},
                description=tool_spec.get("description", ""),
                parameters={"type": "object", "properties": tool_spec.get("parameters", {})},
            )
        query = item.get("query") or item.get("question") or item.get("text", "")
        response = agent.turn(query, retrieve=False)["response"]

        calls = parse_tool_calls(response)
        if calls:
            format_ok += 1
            expected = item.get("expected_tool") or item.get("ground_truth", "")
            if calls[0].name == expected:
                tool_ok += 1

    return {
        "n_examples": n,
        "format_score": format_ok / max(1, n),
        "tool_match": tool_ok / max(1, n),
    }
