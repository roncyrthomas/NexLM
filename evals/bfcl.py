"""BFCL v3 — Berkeley Function Calling Leaderboard.

Tests the agent's ability to:
  - Choose the right tool from a catalogue
  - Produce well-formed JSON arguments

Two scoring metrics:
  - format_score: fraction of outputs that successfully parse as a tool call
  - tool_match : fraction of outputs that picked the correct tool name

Real BFCL has 2000+ examples across categories (simple, parallel, multiple,
multi-turn, executable, irrelevance, live). We score on the "simple" track.

Dataset loading tries several known HF paths because the BFCL repo has
moved/renamed a few times; falls back to a small inline set for unit tests.
"""

from __future__ import annotations

from typing import Optional

from agent.tools import parse_tool_calls
from agent.wrapper import NexAgent
from evals.runner import register


# ─── BFCL → ToolRegistry adapter ────────────────────────────────────────────
def bfcl_to_registry(bfcl_tools: list[dict], registry) -> None:
    """Convert a BFCL example's tool catalogue into entries on a ToolRegistry.

    BFCL tools come in several formats across releases. We accept the ones
    seen in v2 and v3:

      a) Plain JSON-schema-like:
         {"name": ..., "description": ..., "parameters": {"type": ..., "properties": {...}}}

      b) OpenAI-style function spec:
         {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}

      c) String-typed shorthand (some user examples):
         {"name": ..., "parameters": {"arg1": "string", "arg2": "integer"}}

    The adapter normalizes all three into our internal {description, parameters}
    shape and registers no-op stubs so the model can score on tool selection
    without us implementing the actual tool.
    """
    registry.clear()
    normalized = []
    for spec in bfcl_tools:
        if isinstance(spec, dict) and "function" in spec and isinstance(spec["function"], dict):
            inner = spec["function"]
            normalized.append({
                "name": inner.get("name", ""),
                "description": inner.get("description", ""),
                "parameters": inner.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            params = spec.get("parameters", {})
            # if params is the shorthand {"name": "type"} form, wrap in JSON-schema
            if isinstance(params, dict) and "type" not in params and "properties" not in params:
                params = {"type": "object", "properties": {k: {"type": v} for k, v in params.items()}}
            normalized.append({
                "name": spec.get("name") or spec.get("function_name", ""),
                "description": spec.get("description", ""),
                "parameters": params,
            })
    registry.register_many(normalized, stub=True)


def _extract_expected_tool(item: dict) -> Optional[str]:
    """Extract the ground-truth tool name from a BFCL example.

    Tries the fields BFCL has used across its history. Returns None if it
    can't find one — caller should skip those for the tool_match score.
    """
    # v3 simple track uses `ground_truth` as list of {name, arguments}
    gt = item.get("ground_truth")
    if isinstance(gt, list) and gt and isinstance(gt[0], dict):
        return gt[0].get("name")
    if isinstance(gt, dict):
        return gt.get("name")
    if isinstance(gt, str):
        # sometimes a single-tool name string, sometimes "tool_name(arg=val,...)"
        return gt.split("(")[0].strip()
    # our fallback set + some older BFCL dumps
    if "expected_tool" in item:
        return item["expected_tool"]
    # v2 used "answer"
    ans = item.get("answer")
    if isinstance(ans, dict):
        return ans.get("name") or ans.get("function_name")
    return None


# ─── Fallback inline eval set for unit tests + offline use ──────────────────
_FALLBACK_EVALSET = [
    {
        "tools": [
            {"name": "get_weather", "description": "Get current weather",
             "parameters": {"city": "string"}},
            {"name": "send_email", "description": "Send an email",
             "parameters": {"to": "string", "body": "string"}},
        ],
        "query": "What's the weather like in Tokyo right now?",
        "expected_tool": "get_weather",
    },
    {
        "tools": [
            {"name": "get_weather", "description": "Get current weather",
             "parameters": {"city": "string"}},
            {"name": "send_email", "description": "Send an email",
             "parameters": {"to": "string", "body": "string"}},
        ],
        "query": "Send Alice an email saying I'll be late.",
        "expected_tool": "send_email",
    },
    {
        "tools": [
            {"name": "calc", "description": "Evaluate a math expression",
             "parameters": {"expr": "string"}},
        ],
        "query": "What is 17 times 23?",
        "expected_tool": "calc",
    },
    {
        "tools": [
            {"name": "search_web", "description": "Search the web",
             "parameters": {"query": "string"}},
            {"name": "summarize", "description": "Summarize text",
             "parameters": {"text": "string"}},
        ],
        "query": "Look up the latest news about renewable energy in Germany.",
        "expected_tool": "search_web",
    },
]


# Dataset path candidates, tried in order. BFCL has been mirrored several places.
_HF_PATHS = [
    ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", None, "train"),
    ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "simple", "train"),
    ("gorilla-llm/BFCL-v3", None, "train"),
    ("Salesforce/xlam-function-calling-60k", None, "train"),
]


def _load_eval_set(max_examples: int) -> tuple[list[dict], str]:
    """Try HF paths in order; fall back to inline. Returns (items, source_label)."""
    try:
        from datasets import load_dataset
        for path, name, split in _HF_PATHS:
            try:
                kwargs = {"split": split, "streaming": True}
                if name:
                    kwargs["name"] = name
                ds = load_dataset(path, **kwargs)
                out = []
                for i, row in enumerate(ds):
                    if i >= max_examples:
                        break
                    out.append(row)
                if out:
                    return out, f"hf:{path}"
            except Exception:
                continue
    except ImportError:
        pass
    return _FALLBACK_EVALSET[:max_examples], "inline_fallback"


@register("bfcl")
def evaluate(agent: NexAgent, max_examples: int = 100) -> dict:
    items, source = _load_eval_set(max_examples)
    n = len(items)
    format_ok = 0
    tool_ok = 0
    matchable = 0  # examples where we could extract a ground-truth name

    for item in items:
        tools_field = item.get("tools") or item.get("functions") or item.get("function_specs") or []
        bfcl_to_registry(tools_field, agent.tools)

        query = item.get("query") or item.get("question") or item.get("text") or item.get("user_query", "")
        response = agent.turn(query, retrieve=False)["response"]

        calls = parse_tool_calls(response)
        if calls:
            format_ok += 1
            expected = _extract_expected_tool(item)
            if expected is not None:
                matchable += 1
                if calls[0].name == expected:
                    tool_ok += 1

    return {
        "n_examples": n,
        "source": source,
        "format_score": format_ok / max(1, n),
        "tool_match": tool_ok / max(1, matchable),
        "n_matchable": matchable,
    }
