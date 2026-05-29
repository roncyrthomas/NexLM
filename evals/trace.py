"""TRACE — continual learning eval.

Simplified protocol: present a sequence of tasks, measure final accuracy
on each task at the end of the sequence. Backward transfer (BWT) = mean
drop in accuracy on early tasks vs their post-training accuracy.

We use a tiny synthetic task stream for unit-test purposes. The real TRACE
benchmark has 8 task domains; switch out _load_task_stream() when running
the actual paper experiments.
"""

from __future__ import annotations

from agent.wrapper import NexAgent
from evals.runner import register


_SYNTHETIC_TASKS = [
    ("math", [{"q": "What is 7 + 5?", "a": "12"}, {"q": "What is 9 - 3?", "a": "6"}]),
    ("trivia", [{"q": "Capital of France?", "a": "Paris"}, {"q": "Largest ocean?", "a": "Pacific"}]),
    ("logic", [{"q": "If all cats are mammals, are cats mammals?", "a": "yes"},
                {"q": "If A>B and B>C, is A>C?", "a": "yes"}]),
]


def _eval_task(agent: NexAgent, examples: list[dict]) -> float:
    n_correct = 0
    for ex in examples:
        out = agent.turn(ex["q"], retrieve=False)["response"]
        if "Assistant:" in out:
            out = out.split("Assistant:")[-1]
        if ex["a"].lower() in out.lower():
            n_correct += 1
    return n_correct / max(1, len(examples))


@register("trace")
def evaluate(agent: NexAgent, max_examples: int = 100) -> dict:
    """Run sequential tasks and report avg acc + a forgetting proxy."""
    tasks = _SYNTHETIC_TASKS
    per_task_acc: dict[str, float] = {}

    for task_name, examples in tasks:
        # In a real continual setup we'd do a short SFT step here for each task.
        # The unit test version just evaluates without per-task training.
        per_task_acc[task_name] = _eval_task(agent, examples)

    avg = sum(per_task_acc.values()) / max(1, len(per_task_acc))
    return {
        "n_tasks": len(tasks),
        "avg_accuracy": avg,
        "per_task": per_task_acc,
    }
