"""Unified eval runner.

All benchmarks implement a common interface:
    score = bench.run(agent, max_examples=N) -> dict[str, float]

This file routes the benchmark name to the implementation and aggregates
results into a single comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from agent.wrapper import NexAgent


BENCHMARK_REGISTRY: dict[str, Callable[[NexAgent, int], dict]] = {}


def register(name: str):
    def decorator(fn: Callable[[NexAgent, int], dict]) -> Callable:
        BENCHMARK_REGISTRY[name] = fn
        return fn
    return decorator


def run_one(agent: NexAgent, benchmark: str, max_examples: int = 100) -> dict:
    if benchmark not in BENCHMARK_REGISTRY:
        raise KeyError(f"unknown benchmark: {benchmark}. Available: {list(BENCHMARK_REGISTRY)}")
    return BENCHMARK_REGISTRY[benchmark](agent, max_examples)


def run_all(agent: NexAgent, max_examples: int = 100, output_dir: str = "evals/results") -> dict:
    """Run every registered benchmark on this agent."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in BENCHMARK_REGISTRY.items():
        print(f"\n[eval] running {name}...")
        try:
            r = fn(agent, max_examples)
            results[name] = r
            print(f"[eval] {name}: {r}")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"[eval] {name} FAILED: {e}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def compare(agent_a: NexAgent, agent_b: NexAgent, max_examples: int = 100) -> dict:
    """A/B comparison — the bread and butter of the pivoted paper.

    Returns per-benchmark deltas: agent_b - agent_a.
    """
    print("[compare] running benchmarks on agent_a (baseline)...")
    a = {name: fn(agent_a, max_examples) for name, fn in BENCHMARK_REGISTRY.items()}
    print("[compare] running benchmarks on agent_b (treatment)...")
    b = {name: fn(agent_b, max_examples) for name, fn in BENCHMARK_REGISTRY.items()}

    delta = {}
    for name in BENCHMARK_REGISTRY:
        delta[name] = {}
        for metric in a[name]:
            if isinstance(a[name][metric], (int, float)):
                delta[name][metric] = b[name][metric] - a[name][metric]
    return {"agent_a": a, "agent_b": b, "delta": delta}


# Auto-import concrete benchmarks so their @register decorators fire
from evals import bfcl  # noqa: F401, E402
from evals import musique  # noqa: F401, E402
from evals import trace  # noqa: F401, E402
