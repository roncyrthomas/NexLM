# Frank v2: Self-Learning, Self-Updating, Intuitive Agent

**Date:** 2026-05-29
**Status:** Design spec — implementation-ready
**Supersedes:** None. Complements `2026-05-29-agent-layer-pivot.md` as the *next-generation* successor agent.
**Companion model lineage:** Vanilla base (LoRA only) → **Frank v1** (P1–P4: hormones+Hebbian+HippoRAG+Titans) → **Frank v2** (this spec: + predictive+episodic+habits+dreamer+metaplastic).

---

## 1. Why a v2 exists

**v1's structural limit:** every learning signal in Frank v1 traces back to *external reward* (user accept/reject). It is therefore reactive — it only learns when humans react. This places a ceiling on how much it can learn, on what timescale, and at what cost.

**v2's thesis:** by adding **self-derived learning signals** (predictive error), **fast associative recall** (episodic cache), **compiled procedural memory** (habits), **offline consolidation** (dreamer), and **meta-plasticity** (Hebbian-on-Hebbian), the agent transitions from craftsman (high-quality recombination on demand) to something closer to intuitive (anticipates, recalls, decides without re-deriving).

**Three-way cross-comparison** lets us isolate the contribution of each generation:

| Model | Layers active | What it tests |
|---|---|---|
| **Vanilla** | Base model + LoRA only | Phi-3-mini baseline |
| **Frank v1** | + Hormones, Hebbian, HippoRAG, Titans, LoRA-online | Does our agent layer add value? |
| **Frank v2** | + Predictive, Episodic, Habits, Dreamer, Metaplastic | Does self-learning beyond external reward add value? |

The paper claim shifts from "our agent layer helps" (v1) to "self-derived learning signals are the load-bearing piece" (v2 vs v1 delta).

---

## 2. Architecture overview

```
┌────────────────────────────────────────────────────────────────────┐
│  Frozen base: SmolLM2-1.7B / Phi-3-mini                            │
└─────────┬──────────────────────────────────────┬───────────────────┘
          │ hidden states                        │ logits
          │                                      │
┌─────────▼──────────────────────────────────────▼───────────────────┐
│                   FRANK v1 AGENT LAYER (unchanged)                 │
│  Tier 0a: Hormones    Tier 0b: Hebbian    Tier 1: HippoRAG         │
│  Tier 2:  Titans MAG  Tier 3: Runtime LoRA + snapshot ring         │
└─────────┬──────────────────────────────────────┬───────────────────┘
          │                                      │
┌─────────▼──────────────────────────────────────▼───────────────────┐
│                   FRANK v2 NEW TIERS                               │
│                                                                    │
│  Tier P (Predictive):   predict next user input, derive surprise   │
│  Tier E (Episodic):     embedding cache + KNN of past situations   │
│  Tier H (Habits):       compiled (task,tool,shape) fast paths      │
│  Tier D (Dreamer):      idle-time consolidation via world model    │
│  Tier M (Metaplastic):  meta-Hebbian — per-pair learning rates     │
│                                                                    │
│  Self-learning loop: Tier P generates internal reward → Tier 0a    │
│  hormones update without external feedback → all v1 tiers adapt.   │
└────────────────────────────────────────────────────────────────────┘
```

**Key architectural shift:** v1 had reward flow **inward** (user → hormones → tiers). v2 adds reward flow **inward from inside** (predictive error → hormones → tiers). The agent gets a pulse independent of conversation.

---

## 3. The five new tiers

### Tier P — Predictive coding (`agent/predictive.py`)

**Purpose:** Generate internal learning signal from prediction error, severing dependency on external reward.

**Mechanism:**
1. At each turn, run the base model to predict a distribution over likely next user inputs given conversation context. (Sample K candidate continuations, weight by base model logprob.)
2. At turn t+1 when the user actually responds, compute the log-probability of their actual message under the prediction distribution.
3. `surprise = -log p(actual | predicted)`. Normalize to [0, 1] via running stats.
4. **Feed surprise into the hormone system** as a new input alongside reward and retry_signal.

**Effect on hormones:** Surprise replaces boredom's update source. High surprise → low boredom → Titans `tau_surprise` lowers → memory updates more eagerly. The system now self-modulates its plasticity based on its own predictive accuracy.

**Interface:**
```python
class Predictive:
    def predict(self, conversation: list[dict]) -> Distribution: ...
    def observe(self, actual_user_message: str) -> float:
        """Returns surprise in [0, 1]."""
    def attach_to_hormones(self, hormones: HormoneState) -> None:
        """Wire surprise stream into hormone update path."""
```

**Tests:**
- Predictable user message → low surprise
- Random user message → high surprise
- Repeated identical turns → surprise decays as model "expects" them

**Param cost:** zero new params; reuses base + LoRA.

**Why it's the single biggest piece:** every other v2 tier benefits from a real surprise signal. Without Tier P, Tier D has nothing to dream about, Tier M has nothing to recalibrate, Tier H has no consistent reward proxy.

---

### Tier E — Episodic memory (`agent/episodic.py`)

**Purpose:** Fast associative recall — "have I seen something like this before?" — that biases generation toward proven good responses.

**Mechanism:**
1. Every turn embeds `(user_query, response_summary, outcome)` into a vector using the base model's penultimate layer mean-pooled.
2. Store in an in-memory FAISS or `scipy.spatial.cKDTree` index (FAISS optional, scipy default to avoid heavy dep).
3. At inference: KNN over current query embedding. If `max_similarity > τ_episodic` AND past outcomes were positive (joy in observed feedback), retrieve those past responses as exemplars and inject as few-shot context.

**Pruning policy:** When buffer exceeds N entries, drop the lowest-utility entry (utility = recency × success_rate × distance-from-other-entries to preserve diversity).

**Interface:**
```python
class EpisodicMemory:
    def remember(self, query: str, response: str, reward: float) -> None: ...
    def recall(self, query: str, k: int = 5) -> list[Episode]: ...
    def prune(self, max_size: int = 10_000) -> None: ...
```

**Tests:**
- Recall returns nothing on empty buffer
- After 5 stores with same query, recall returns them at top of ranking
- Pruning keeps high-utility entries

**Param cost:** zero. Just storage. ~100MB for 10k episodes.

---

### Tier H — Habits (`agent/habits.py`)

**Purpose:** Compile repeated `(task_intent, tool, response_shape)` triples into bypass-the-generation fast paths. Most cognition is habit, not deliberation.

**Mechanism:**
1. Counter on each `(task_intent_id, tool_id, shape_id)` triple seen with positive reward.
2. When `count > N AND consistent_reward > τ_habit`, mark the triple as **compiled**.
3. At inference: if task_intent matches a compiled triple, **return the cached response directly without running the base model**. Decision is microseconds, not seconds.
4. Decay: compiled habits that aren't fired within K turns get demoted back to non-compiled status (use-it-or-lose-it).

**Interface:**
```python
class HabitsCache:
    def observe(self, intent: int, tool: int, shape: int, reward: float) -> None: ...
    def maybe_bypass(self, intent: int) -> Optional[CachedResponse]: ...
    def decay_unused(self) -> int:
        """Returns number demoted."""
```

**Tests:**
- After 10 (intent=5, tool=2, shape=1, reward=1.0), the triple is compiled
- maybe_bypass returns the cached response on intent=5
- After K turns without firing, the habit is demoted

**Param cost:** zero. Sparse dict.

**Critical safety:** habits NEVER fire if hormones.frustration > τ_habit_inhibit. Frustration means "things are going wrong, don't autopilot."

---

### Tier D — Dreamer (`agent/dreamer.py`)

**Purpose:** Idle-time consolidation. Train the LoRA on synthetic interactions drawn from the agent's own memory, without needing new external data.

**Mechanism:**
1. When `hormones.fatigue > τ_sleep`, enter dream mode.
2. Sample K (situation, response) pairs from episodic memory + HippoRAG.
3. For each: run the agent layer to generate a new response in the same situation. Compute predictive error against the past response.
4. Use that error as a training signal for LoRA (gradient step with hormone-modulated LR).
5. Take a snapshot after the dream session.

**Why this works:** the agent rehearses what worked, refines what didn't, without new data costing money or time. The same trick Hinton's wake-sleep algorithm used; the same trick biological brains do.

**Interface:**
```python
class Dreamer:
    def should_dream(self, hormones: HormoneState) -> bool: ...
    def dream(self, n_samples: int = 100) -> dict:
        """Returns stats: {samples, avg_pred_error, loss_delta}."""
```

**Tests:**
- Dreams reduce per-sample predictive error on the rehearsal set
- Does not fire when fatigue is low
- LoRA weights change before vs after dream session

**Param cost:** zero new params; uses existing LoRA + Predictive.

---

### Tier M — Metaplastic (`agent/metaplastic.py`)

**Purpose:** Make the Hebbian learning rate itself adaptive per-feature-pair. Stable pairs learn faster; noisy pairs learn slower.

**Mechanism:**
1. Track per-pair `eta[i,j]` matrix (same shape as Hebbian H).
2. After each Hebbian update, retrospectively track whether the strengthened association led to a positively-reinforced outcome within Δ turns.
3. If yes → `eta[i,j] *= 1 + α`. If no → `eta[i,j] *= 1 - β`. Clamp.
4. Hebbian update rule now uses `eta[i,j]` instead of global `eta`.

**Interface:**
```python
class MetaPlastic:
    def attach_to_hebbian(self, hebbian: HebbianH) -> None: ...
    def update_after_outcome(self, active_features: list[int], outcome_reward: float) -> None: ...
```

**Tests:**
- After 10 successful updates on pair (0,1), eta[0,1] > eta_default
- After 10 failed updates on pair (2,3), eta[2,3] < eta_default
- Clamping prevents runaway

**Param cost:** ~3MB (850×850 float32 second matrix).

---

## 4. Integration into `NexAgent`

Add to `AgentConfig`:
```python
# Frank v2 tier toggles
enable_predictive: bool = False
enable_episodic: bool = False
enable_habits: bool = False
enable_dreamer: bool = False
enable_metaplastic: bool = False

# Frank v2 hyperparams
episodic_buffer_size: int = 10_000
episodic_similarity_threshold: float = 0.85
habits_compile_threshold: int = 10
habits_reward_threshold: float = 0.7
dream_n_samples: int = 100
metaplastic_alpha: float = 0.05
metaplastic_beta: float = 0.03
```

Add three new convenience presets to `AgentConfig`:

```python
@classmethod
def vanilla(cls):
    """Baseline — LoRA only, no agent layer."""
    return cls()  # current default

@classmethod
def frank_v1(cls):
    """Full Frank v1 — external-reward-driven agent layer."""
    return cls(
        enable_hormones=True, enable_hebbian=True,
        enable_hipporag=True, enable_titans=True,
        train_lora_online=True,
    )

@classmethod
def frank_v2(cls):
    """Full Frank v2 — adds self-derived learning."""
    cfg = cls.frank_v1()
    cfg.enable_predictive = True
    cfg.enable_episodic = True
    cfg.enable_habits = True
    cfg.enable_dreamer = True
    cfg.enable_metaplastic = True
    return cfg
```

Update `NexAgent.__init__` to instantiate each v2 tier when its flag is on, mirroring how v1 tiers are handled.

Update `NexAgent.turn()` pipeline order:
1. **Habit check** (Tier H) — if compiled triple matches, bypass everything else, return cached response.
2. **Episodic recall** (Tier E) — KNN lookup past similar situations as few-shot context.
3. **HippoRAG retrieve** (Tier 1, unchanged) — KG retrieval.
4. **Predictive forecast** (Tier P) — predict likely next user inputs (cached for next turn's surprise computation).
5. **Generate** with hormone-modulated temperature.
6. **Parse tools, return response.**

Update `NexAgent.observe_feedback()` to:
1. Compute predictive surprise from the actual user message at turn t+1 vs the prediction at turn t.
2. Inject surprise into `HormoneState.update(...)`.
3. Store the episode in `EpisodicMemory`.
4. Update `HabitsCache`.
5. Update `MetaPlastic` with retrospective utility.
6. If `Dreamer.should_dream()` → schedule a dream cycle.

---

## 5. Three-way comparison protocol

```python
from agent import AgentConfig, NexAgent
from evals.runner import compare_three

results = compare_three(
    vanilla=NexAgent(AgentConfig.vanilla()),
    frank_v1=NexAgent(AgentConfig.frank_v1()),
    frank_v2=NexAgent(AgentConfig.frank_v2()),
    benchmarks=["bfcl", "musique", "trace"],
    max_examples_per_benchmark=200,
    seeds=[0, 1, 2],
)
```

Returns a delta matrix: each benchmark × each metric × each (v1−vanilla), (v2−v1), (v2−vanilla).

The **(v2 − v1) deltas are the paper's load-bearing evidence**: do self-derived signals add value over external-reward-only signals?

---

## 6. Eval predictions (what we expect to see if v2 works)

| Benchmark | (v1 − vanilla) expected | (v2 − v1) expected | Why |
|---|---|---|---|
| BFCL tool_match | +5 to +15 pp | +3 to +8 pp | Habits compile common (intent → tool) routings |
| MuSiQue EM | +3 to +10 pp | +1 to +4 pp | Mostly retrieval; episodic adds modest recall |
| TRACE avg_acc | +5 to +15 pp | +5 to +15 pp | Dreamer prevents forgetting across task switches; predictive sharpens it |
| TRACE BWT (forgetting) | small reduction | **large reduction** | Dreamer rehearsal is the right tool for forgetting |
| **First-token latency** | +5–10% slower | **−40% faster** on compiled tasks | Habits bypass generation entirely |

If we don't see TRACE-BWT improvement specifically, the dreamer doesn't work as designed; that's our most diagnostic single metric.

---

## 7. Implementation milestones

| Step | File(s) | Effort | Critical? |
|---|---|---|---|
| V2.1 | `agent/predictive.py` + tests | 1 day | **YES — load-bearing** |
| V2.2 | `agent/episodic.py` + tests | 1 day | YES |
| V2.3 | `agent/habits.py` + tests | half day | nice-to-have |
| V2.4 | `agent/dreamer.py` + tests | 2 days (hardest) | YES — drives BWT claim |
| V2.5 | `agent/metaplastic.py` + tests | half day | nice-to-have |
| V2.6 | wire into `NexAgent` + new presets | 1 day | YES |
| V2.7 | `evals.runner.compare_three()` | half day | YES |
| V2.8 | three-way experiments on cloud | ~$100 | YES |
| V2.9 | paper draft updated to v2 thesis | 1 week | YES |

**Total: ~2 weeks engineering + ~$100 compute.** Bookended by the agent-layer-pivot's M5–M9 budget; total project envelope still ~$250.

---

## 8. Anti-patterns to avoid

| Anti-pattern | Why bad | What to do instead |
|---|---|---|
| Letting habits fire under high frustration | Compiled mistakes accumulate fast | Frustration > τ blocks all habit firing |
| Dreaming on raw external rewards only | Reinforces user-pleasing, not correctness | Dream uses predictive error, not stored rewards |
| Episodic buffer with no pruning | OOM at scale, dilutes recall quality | Mandatory utility-based pruning at N=10k |
| Metaplastic without bounds | Runaway η on lucky pairs | Hard clamp on `eta[i,j]` ∈ [η_min, η_max] |
| Predictive that reinforces echo-chamber | Model only predicts what user already said | Conversation context must include retrieved/external content, not just history |

---

## 9. What's still missing (honest)

Even Frank v2 is **still a craftsman**, just a craftsman with intuition and habits. The real artist threshold from the prior conversation requires:
- Self-modifying *goals* (not just self-modifying weights). v2 has none of this.
- Embodied feedback. v2 has none.
- Counter-extrapolation as a primitive. v2 has none.
- Compositionality from invented primitives. v2 has none.

A **Frank v3** would need at least one of these. Right now we have no clean engineering recipe for any of them. v2 is the last fully-tractable step on this path; v3 would be a research project, not engineering.

---

## 10. Why this is publishable

The paper claim sharpens further:

> **"Self-derived learning signals (predictive coding + episodic recall + habits + dreamer) are the load-bearing piece of an adaptive agent layer. Removing them — but keeping the v1 external-reward-driven tiers — costs Y% on TRACE-BWT and Z% on long-horizon BFCL."**

This is a much stronger thesis than v1's. It directly tests *whether self-learning matters* in a controlled three-way comparison on standard benchmarks. That's the kind of claim NeurIPS or workshop reviewers can engage with.

---

## 11. Immediate next action

Start V2.1 — `agent/predictive.py`. It's the single highest-leverage module. Everything else is composable on top.

Implementation sketch:

```python
class PredictiveCoder:
    def __init__(self, base_model, tokenizer, ema_alpha=0.1):
        self.base = base_model
        self.tokenizer = tokenizer
        self.surprise_ema_mean = 0.0
        self.surprise_ema_var = 1.0
        self.alpha = ema_alpha
        self._last_prediction: torch.Tensor | None = None
        self._last_context: list[dict] | None = None

    @torch.no_grad()
    def predict(self, conversation: list[dict]) -> None:
        """Cache a prediction distribution over likely next user inputs."""
        # Build prompt expecting a USER turn next
        prompt = render_for_user_turn_prediction(conversation)
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.base.device)
        out = self.base(ids)
        self._last_prediction = out.logits[0, -1].softmax(dim=-1)  # next-token distrib
        self._last_context = conversation

    def observe(self, actual_user_message: str) -> float:
        """Compute normalized surprise of the observed message."""
        if self._last_prediction is None:
            return 0.5
        ids = self.tokenizer(actual_user_message, return_tensors="pt").input_ids[0]
        first_tok = ids[0].item()
        p = self._last_prediction[first_tok].item()
        raw_surprise = -math.log(max(p, 1e-9))
        # Normalize via EMA z-score → sigmoid
        delta = raw_surprise - self.surprise_ema_mean
        self.surprise_ema_mean += self.alpha * delta
        self.surprise_ema_var = (1 - self.alpha) * self.surprise_ema_var + self.alpha * delta * delta
        z = delta / (math.sqrt(self.surprise_ema_var) + 1e-6)
        return 1.0 / (1.0 + math.exp(-z))  # sigmoid → [0, 1]
```

That's the seed. Around it grows the rest of v2.

---

*End of Frank v2 spec. Next: implement V2.1, commit to GitHub, plan v2-aware cloud experiments.*
