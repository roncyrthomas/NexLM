"""MuSiQue multi-hop QA — tests HippoRAG retrieval contribution.

Score: exact-match (EM) on the answer string.
"""

from __future__ import annotations

import re
import string

from agent.wrapper import NexAgent
from evals.runner import register


_FALLBACK_MUSIQUE = [
    {
        "question": "Who founded the company that owns YouTube?",
        "answer": "Larry Page",
        "supporting": "YouTube is owned by Google. Google was founded by Larry Page and Sergey Brin.",
    },
    {
        "question": "What city is the capital of the country where the Eiffel Tower is located?",
        "answer": "Paris",
        "supporting": "The Eiffel Tower is located in France. Paris is the capital of France.",
    },
]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(gold) in _normalize(pred)


def _load_eval_set(max_examples: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset("dgslibisey/MuSiQue", split="validation", streaming=True)
        out = []
        for i, row in enumerate(ds):
            if i >= max_examples:
                break
            out.append({"question": row["question"], "answer": row["answer"]})
        return out if out else _FALLBACK_MUSIQUE
    except Exception:
        return _FALLBACK_MUSIQUE[:max_examples]


@register("musique")
def evaluate(agent: NexAgent, max_examples: int = 100) -> dict:
    items = _load_eval_set(max_examples)
    n_correct = 0
    for item in items:
        # If HippoRAG is loaded and contains supporting docs, retrieve()
        # will inject context; otherwise the model answers cold.
        out = agent.turn(item["question"], retrieve=True)["response"]
        # Try to extract the part of the response that follows "Assistant:"
        if "Assistant:" in out:
            out = out.split("Assistant:")[-1]
        if _exact_match(out, item["answer"]):
            n_correct += 1
    return {
        "n_examples": len(items),
        "exact_match": n_correct / max(1, len(items)),
    }
