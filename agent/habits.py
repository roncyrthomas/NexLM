"""Tier H — Habits.

Compiled `(task_intent_id, tool_id, response_shape_id)` triples that have
fired enough times with positive reward get marked as habits. When a
matching intent recurs, the cached response is returned directly,
bypassing the base model's generation loop.

Critical safety: habits never fire when hormones.frustration > τ.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HabitRecord:
    intent: int
    tool: int
    shape: int
    count: int = 0
    reward_sum: float = 0.0
    cached_response: str = ""
    compiled: bool = False
    turns_since_used: int = 0


@dataclass
class HabitsCache:
    compile_threshold: int = 10        # how many positive turns before compiling
    reward_threshold: float = 0.7      # avg reward must exceed this
    decay_max_unused: int = 50         # habit demoted if not used in N turns
    frustration_inhibit: float = 0.5   # block habits when hormones.frustration above this

    records: dict[tuple[int, int, int], HabitRecord] = field(default_factory=dict)
    by_intent: dict[int, list[HabitRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def observe(
        self,
        intent: int,
        tool: int,
        shape: int,
        reward: float,
        cached_response: Optional[str] = None,
    ) -> None:
        key = (intent, tool, shape)
        rec = self.records.get(key)
        if rec is None:
            rec = HabitRecord(intent=intent, tool=tool, shape=shape)
            self.records[key] = rec
            self.by_intent[intent].append(rec)
        rec.count += 1
        rec.reward_sum += reward
        rec.turns_since_used = 0
        if cached_response:
            rec.cached_response = cached_response
        # Eligibility for compilation
        if (
            not rec.compiled
            and rec.count >= self.compile_threshold
            and rec.reward_sum / rec.count >= self.reward_threshold
        ):
            rec.compiled = True

    def maybe_bypass(
        self,
        intent: int,
        frustration: float = 0.0,
    ) -> Optional[HabitRecord]:
        """Return the compiled habit for this intent, or None."""
        if frustration > self.frustration_inhibit:
            return None
        candidates = [r for r in self.by_intent.get(intent, []) if r.compiled]
        if not candidates:
            return None
        # Prefer the highest average reward
        best = max(candidates, key=lambda r: r.reward_sum / max(1, r.count))
        return best

    def tick(self) -> int:
        """Increment turns_since_used and demote anything stale.

        Returns number demoted."""
        demoted = 0
        for rec in self.records.values():
            rec.turns_since_used += 1
            if rec.compiled and rec.turns_since_used > self.decay_max_unused:
                rec.compiled = False
                demoted += 1
        return demoted

    def stats(self) -> dict:
        compiled_count = sum(1 for r in self.records.values() if r.compiled)
        return {
            "total_records": len(self.records),
            "compiled": compiled_count,
            "compiled_intents": len({r.intent for r in self.records.values() if r.compiled}),
        }
