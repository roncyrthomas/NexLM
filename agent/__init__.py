"""NexLM Agent — tri-memory neuromodulation layer on top of a frozen base model.

This is the pivoted P1+ direction (see docs/superpowers/specs/2026-05-29-agent-layer-pivot.md).
We wrap a pretrained small LM and add the agent layer:
  - Tier 0a: Hormone scalars
  - Tier 0b: Hebbian co-firing matrix
  - Tier 1: HippoRAG cross-attn adapter
  - Tier 2: Titans MAG MLP (test-time updates)
  - Tier 3: Runtime LoRA + snapshot ring

Only the agent layer is trained; the base model stays frozen.
"""

from agent.config import AgentConfig
from agent.wrapper import NexAgent

__all__ = ["AgentConfig", "NexAgent"]
