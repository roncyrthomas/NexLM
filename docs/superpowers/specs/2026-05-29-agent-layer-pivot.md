# Pivot: Tri-Memory Neuromodulation as a Base-Model-Agnostic Agent Layer

**Date:** 2026-05-29
**Status:** Primary spec (supersedes 2026-05-28-frankenstein-slm-design.md as the *product* direction)
**Author:** Vaisakh
**Why this exists:** The original spec required $5,000+ to validate (pretrain from scratch + ablations). The actual novel contribution — *reward-gated three-factor Hebbian co-firing unified with affect-modulated multi-timescale memory* — is base-model-agnostic and can be validated for $50–200 on Phi-3-mini.

---

## 1. What changed and why

**Original spec ambition:** train a 700M Frankenstein from scratch, prove every component contributes via ablations on 60B-token pretrains.

**Brutal honest realization (from session 2026-05-29):**
- $15 cloud budget cannot deliver competitive 700M weights (Phi-3-mini saw 3.3T tokens; we'd have ~50M).
- Reviewers will demand component-by-component ablations vs same-size baselines → $5k–10k.
- The genuinely original contribution isn't the backbone — it's the *agent layer*.

**New thesis:**
> A 50–80M parameter **agent layer** (hormones + Hebbian co-firing matrix + runtime LoRA with snapshot rollback + HippoRAG retrieval interface) attached to any pretrained small base model improves performance on agentic benchmarks (tool calling, multi-hop QA, continual learning) by a measurable, reproducible amount, with the unifying neuromodulatory schedule as the architectural contribution.

**Why this is more defensible:**
1. Tight A/B comparison: same base model with/without our agent layer
2. Each Tier ablatable cheaply (LoRA finetune is $5–10, not $500)
3. Real benchmarks (BFCL, TRACE, MuSiQue) instead of synthetic toys
4. Product-viable: deploys on top of off-the-shelf Phi-3-mini
5. No claims about base model competitiveness — sidesteps the entire ablation-cost problem

---

## 2. Preserved vs replaced from the original spec

| Component | Original spec | Pivoted spec |
|---|---|---|
| 24-layer Mamba2+DiffAttn hybrid backbone | The whole model | **Research demo only** (M1a smoke as proof-of-concept) |
| Titans MAG memory | Internal layer | **External adapter**, attached to base model's hidden states at select layers via hooks |
| HippoRAG | Inject at attention layers | **Same, via cross-attention adapter on Phi-3-mini's attention layers** |
| Hormones (Tier 0a) | Modulate everything | **Same** — modulate LoRA LR, sampling temp, retrieval intensity |
| Hebbian H (Tier 0b) | Co-firing matrix | **Same** — sparse |F|×|F| float32 matrix biasing tool-selection logits |
| Runtime LoRA (Tier 3) | On our base model | **LoRA adapters on Phi-3-mini's Q/K/V/O projections** |
| Granite-style tool calling | SFT from scratch | **Fine-tune Phi-3-mini via LoRA** to use our `<tool_call>` format |
| 60B-token pretrain (M3) | Centerpiece | **Dropped** — research note: "future work, requires industrial budget" |
| FrankenBench (M8) | Custom + ablation matrix | **Reduced** — focus on 3 benchmarks, factorial ablation of the *agent layer* only |
| COLM 2026 paper | Main track | **More realistically:** workshop track or NeurIPS systems track. Or post on arXiv + ship product. |

**M1a smoke run lives forever as Figure 1** (proof the novel hybrid architecture is implementable and stable).
**M1b code is preserved as future-work appendix** (architecture available for those with industrial resources).

---

## 3. New architecture: the Agent Layer

```
┌─────────────────────────────────────────────────────────┐
│  Frozen base model: microsoft/Phi-3-mini-4k-instruct     │
│  (3.8B params, MIT-licensed, ~3.3T pretraining tokens)   │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           │ hidden states (read)         │ logits (write before sampling)
           │                              │
┌──────────▼──────────────────────────────▼───────────────┐
│           AGENT LAYER (~50–80M trainable params)        │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Tier 0a: Hormone scalars (5 EMAs, ~0 params)     │   │
│  │   joy, frustration, confidence, fatigue, boredom │   │
│  │   modulates everything below                     │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Tier 0b: Hebbian H matrix (~3 MB, frozen weights)│   │
│  │   biases tool-selection logits at inference      │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Tier 1: HippoRAG cross-attn adapter (~5M params) │   │
│  │   retrieves chunks; injects at Phi-3 attn layers │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Tier 2: Titans MAG MLP (~15M params, mutable)    │   │
│  │   parallel branch with frozen-base hidden state  │   │
│  │   test-time surprise-gated inner updates         │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Tier 3: Runtime LoRA (rank 16, ~30M params)      │   │
│  │   adapters on Phi-3's Q/K/V/O                    │   │
│  │   snapshot ring + canary eval + rollback         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Total trainable: ~50–80M params.** Base Phi-3-mini stays frozen.
**Memory footprint at inference: ~3.8B base + 80M layer = small.**

---

## 4. Experiments and benchmarks (the actual paper)

**Three claim-evidence pairs:**

### Claim 1: Hebbian co-firing improves tool-selection efficiency
- **Setup:** Phi-3-mini + LoRA adapters, fine-tuned on Glaive-FC-v2 (Apache-2.0).
- **Conditions:** (A) baseline LoRA-only, (B) +Hebbian H, (C) +Hebbian H +hormone-modulated H learning rate.
- **Benchmark:** BFCL v3 (Berkeley Function Calling Leaderboard).
- **Cost:** ~3 × $10 LoRA finetunes + ~$20 BFCL eval = **$50**.
- **Predicted win:** B and C improve few-shot tool-selection precision over A, with C > B due to faster adaptation on user-specific tools (post-hoc fine-tuning eval).

### Claim 2: Hormone-modulated runtime LoRA reduces catastrophic forgetting
- **Setup:** Phi-3-mini + LoRA, sequential task stream from TRACE benchmark.
- **Conditions:** (A) vanilla online LoRA, (B) +snapshot rollback, (C) +hormones+rollback.
- **Benchmark:** TRACE — measure BWT (backward transfer / forgetting).
- **Cost:** ~$50 fine-tunes + benchmark.
- **Predicted win:** C achieves higher avg-accuracy with less negative BWT than A or B.

### Claim 3: HippoRAG + Titans MAG together beat either alone on multi-hop QA
- **Setup:** Phi-3-mini with our three retrieval/memory variants on MuSiQue dev split.
- **Conditions:** (A) Phi-3 alone, (B) +HippoRAG, (C) +Titans MAG, (D) +both.
- **Benchmark:** MuSiQue EM/F1, 2WikiMultiHopQA, HotpotQA.
- **Cost:** essentially free (retrieval-time, no training) + Titans needs ~$30 inner-loop validation.
- **Predicted win:** D > B > C > A; D's gain over B is the tri-memory contribution.

**Total experimental budget: ~$150.** Versus $5,000+ for the original plan.

---

## 5. Revised milestone plan

```
M1a   Smoke model               DONE (2026-05-28). Research demo only.
M1b   700M code + tests         DONE (2026-05-29). Research demo only.
M1b'  Cloud sanity              DROPPED. Documented as future-work.

NEW MILESTONES:
P1    Base wrapper + adapters   Local. Wire Phi-3-mini into our codebase via
                                transformers AutoModelForCausalLM with LoRA on
                                Q/K/V/O. ~1 week, $0.
P2    Hormone scalars + Hebbian Local. Re-purpose M7 code we already designed.
                                ~1 week, $0.
P3    HippoRAG ingest+retrieve  Local CPU. Phi-3-mini does the IE locally
                                (CPU or 8GB VRAM fine). ~1 week, $0.
P4    Titans MAG over Phi-3     Local + tiny cloud. Tier 2 as external MLP
                                cross-attending Phi-3 hidden states. ~1 week, $10.
P5    Claim 1 experiments       Cloud. Glaive-FC-v2 finetune + BFCL eval. ~$50.
P6    Claim 2 experiments       Cloud. TRACE continual learning. ~$50.
P7    Claim 3 experiments       Mostly local. MuSiQue + 2WikiMultiHop. ~$30.
P8    Paper draft               Local. ~1 week.
P9    arXiv submission +        Local. Workshop/COLM-systems submission.
      product packaging         Productize as installable Python library.
```

**Total: ~6 weeks, $150 cloud cost.** Compatible with $15 remaining budget plus AWS credits.

---

## 6. Product side (the "proprietary agent")

Once P1–P4 are done, you have a **deployable package**:

```python
from nexlm import Frankenstein

agent = Frankenstein.from_base("microsoft/Phi-3-mini-4k-instruct")
agent.ingest_documents("./my_docs/")          # HippoRAG graph
agent.register_tools([analyze_churn, make_chart, ...])
agent.train_lora_online = True                 # Tier 3 enabled
agent.enable_hormones = True                   # Tier 0a enabled
agent.enable_hebbian = True                    # Tier 0b enabled
agent.start_server(port=8000)                  # FastAPI on localhost
```

That's the proprietary agent. Built on a proven base, with your novel adaptive layer doing the unique work.

---

## 7. What the paper actually claims (much narrower, much more defensible)

**Title (working):**
> *Tri-Memory Neuromodulation as a Base-Agnostic Agent Layer: Reward-Gated Hebbian Co-Firing and Affect-Modulated Multi-Timescale Memory for Small Pretrained Language Models*

**Abstract structure:**
1. Pretrained small LMs lack mechanisms for online task adaptation without catastrophic forgetting.
2. We introduce an *agent layer* (~80M trainable params) that wraps any frozen base model with three memory systems at three timescales, unified by a scalar affect signal.
3. On Phi-3-mini, our layer improves BFCL v3 tool calling by X%, reduces TRACE forgetting by Y%, and boosts MuSiQue EM by Z% over LoRA-only baselines.
4. Component ablations show the hormone signal is load-bearing (without it, online LoRA collapses within N updates).

**Venue fit:**
- **Best:** ACL/EMNLP workshop on memory or continual learning
- **Stretch:** NeurIPS systems track (because of the modular agent-layer engineering)
- **Realistic backup:** arXiv + tech blog + library on PyPI

**This is honest, defensible, novel, and producible on $150.**

---

## 8. What dies, what survives

**Dies:**
- 700M from-scratch pretrain (M3)
- 60B-token data mix
- Full FrankenBench ablation matrix
- COLM 2026 main-track ambition
- "We beat Phi-3-mini" claims

**Survives:**
- Hybrid Mamba2 + DiffAttn architecture (research demo M1a, future-work mention)
- All five memory tiers (now external to base model)
- The novel unifying neuromodulatory schedule (the actual paper contribution)
- The HippoRAG + Titans + Hebbian + LoRA tri-memory architecture
- The tool-calling + RAG + continual-learning evaluation suite (just narrower scope)
- The product (now real, not vaporware)

---

## 9. Immediate next actions

1. **Stop and terminate the M1b cloud run.** Save $80.
2. **Commit this spec to GitHub** — replaces design.md as primary direction.
3. **Update memory file** to reflect pivot.
4. **Start P1:** clone Phi-3-mini, wrap in our codebase, write LoRA adapter integration. Local, free.
5. **Plan P5/P6/P7 budgets** against AWS credits when they arrive.

---

*Honest pivot, smaller scope, real contribution. Path B from the 2026-05-29 conversation, formalized.*
