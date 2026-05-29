"""Unified eval runner with three-way comparison.

All benchmarks implement a common interface:
    score = bench.run(agent, max_examples=N) -> dict[str, float]

This file routes the benchmark name to the implementation and aggregates
results. The headline function is compare_three() — runs Vanilla / Frank v1
/ Frank v2 on the same benchmarks and reports per-metric deltas.
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
    """Pairwise A/B comparison."""
    a = {name: fn(agent_a, max_examples) for name, fn in BENCHMARK_REGISTRY.items()}
    b = {name: fn(agent_b, max_examples) for name, fn in BENCHMARK_REGISTRY.items()}
    delta = {}
    for name in BENCHMARK_REGISTRY:
        delta[name] = {}
        for metric in a[name]:
            if isinstance(a[name][metric], (int, float)):
                delta[name][metric] = b[name][metric] - a[name][metric]
    return {"agent_a": a, "agent_b": b, "delta": delta}


def compare_three(
    vanilla: NexAgent,
    frank_v1: NexAgent,
    frank_v2: NexAgent,
    benchmarks: list[str] | None = None,
    max_examples: int = 100,
    output_dir: str = "evals/results",
) -> dict:
    """Three-way comparison — the paper's load-bearing experiment.

    Returns a structured dict:
        {
            "vanilla":   {bench_name: metrics, ...},
            "frank_v1":  {bench_name: metrics, ...},
            "frank_v2":  {bench_name: metrics, ...},
            "delta_v1_vs_vanilla": {bench_name: {metric: float}, ...},
            "delta_v2_vs_v1":      {bench_name: {metric: float}, ...},
            "delta_v2_vs_vanilla": {bench_name: {metric: float}, ...},
        }
    """
    benches = benchmarks or list(BENCHMARK_REGISTRY.keys())
    benches = [b for b in benches if b in BENCHMARK_REGISTRY]

    def _run(agent, label):
        out = {}
        for name in benches:
            print(f"\n[compare_three] {label} :: {name}")
            try:
                out[name] = BENCHMARK_REGISTRY[name](agent, max_examples)
            except Exception as e:
                out[name] = {"error": str(e)}
        return out

    a = _run(vanilla, "vanilla")
    b = _run(frank_v1, "frank_v1")
    c = _run(frank_v2, "frank_v2")

    def _delta(x, y):
        d = {}
        for name in benches:
            d[name] = {}
            xn, yn = x.get(name, {}), y.get(name, {})
            for metric in xn:
                if isinstance(xn[metric], (int, float)) and metric in yn and isinstance(yn[metric], (int, float)):
                    d[name][metric] = yn[metric] - xn[metric]
        return d

    results = {
        "vanilla": a,
        "frank_v1": b,
        "frank_v2": c,
        "delta_v1_vs_vanilla": _delta(a, b),
        "delta_v2_vs_v1": _delta(b, c),
        "delta_v2_vs_vanilla": _delta(a, c),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "compare_three.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# Auto-import concrete benchmarks so their @register decorators fire
from evals import bfcl  # noqa: F401, E402
from evals import musique  # noqa: F401, E402
from evals import trace  # noqa: F401, E402
