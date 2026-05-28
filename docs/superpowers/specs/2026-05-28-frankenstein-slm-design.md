# Frankenstein SLM — Design Spec

**Date:** 2026-05-28
**Author:** Vaisakh
**Status:** Draft for review
**Type:** Learning-vehicle SLM with paper-worthy thesis

---

## 1. One-line thesis

A 700M-parameter hybrid Mamba2 + differential-attention small language model with **three independent memory timescales** (symbolic graph, neural test-time memory, Hebbian fast weights + LoRA) unified by a **neuromodulatory affect signal** that gates writes across all three.

Working paper title:
> *Tri-Memory Neuromodulation: Unifying Symbolic Graph Retrieval, Test-Time Neural Memory, and Hebbian Skill Plasticity in a 700M Hybrid SSM via Affect-Gated Learning Rates*

---

## 2. Goals and non-goals

**Primary goal:** Learning vehicle. Build it to understand every piece by wiring them together. Each component must do something measurable on its own ablation.

**Secondary goal:** Publishable systems paper at COLM 2026 (or NeurIPS / ICLR / workshop fallback).

**Tertiary goal:** A genuinely usable local agent for domain tasks (churn analysis, document Q&A, tool-driven workflows).

**Non-goals:**
- Beating Llama-3-8B or any frontier model at general chat.
- Multimodal anything.
- Training infrastructure novelty (use existing kernels).
- Anything that needs >$1500 cloud spend.

---

## 3. Architecture

### 3.1 Backbone — hybrid Mamba2 + differential attention

```
Layer pattern (Samba-style 3:1, 24 layers total)

  L1   Mamba2
  L2   Mamba2
  L3   Mamba2
  L4   DiffAttn   ← precision checkpoint #1
  L5   Mamba2
  L6   Mamba2
  L7   Mamba2
  L8   DiffAttn   ← #2
  L9   Mamba2
  L10  Mamba2
  L11  Mamba2
  L12  DiffAttn   ← #3   (mid-stack — HippoRAG injection point)
  L13  Mamba2
  L14  Mamba2
  L15  Mamba2
  L16  DiffAttn   ← #4
  L17  Mamba2
  L18  Mamba2
  L19  Mamba2
  L20  DiffAttn   ← #5
  L21  Mamba2
  L22  Mamba2
  L23  Mamba2
  L24  DiffAttn   ← #6   (top — tool-call structured output)
```

**Hyperparameters:**

| | Value |
|---|---|
| d_model | 1536 |
| layers | 24 (18 Mamba2 + 6 DiffAttn) |
| heads (attn) | 12, head_dim 128, GQA with 4 kv-heads |
| Mamba2 state dim | 128 |
| Mamba2 expansion | 2× → d_inner 3072 |
| FFN | SwiGLU, expansion ~2.67× → d_ff 4096 |
| Norm | RMSNorm, pre-norm |
| Position | RoPE on diff-attn layers; Mamba2 inherently positional |
| Vocab | 32k (Phi-3 tokenizer + 16 special tokens) |
| Train ctx | 4k → extended to 32k post-pretrain |
| Total params | ~700M (±50M) |

**Why differential attention specifically:** denoised attention maps measurably improve precise recall (Microsoft DIFF Transformer, ICLR 2025). The 6 attention layers carry every precision-critical job: tool-call structure, retrieved-chunk attendance, Titans memory queries, structured output generation. Cheap upgrade over MHA.

### 3.2 Memory hierarchy

```
TIER 0a — Hormones (scalars)        global affect, modulates everything
TIER 0b — Hebbian H matrix (sparse)  "fires together, wires together"
TIER 1  — HippoRAG graph            external, symbolic, exact recall
TIER 2  — Titans MAG memory MLP     internal, compressed, surprise-gated
TIER 3  — Runtime LoRA              behavioral drift, snapshot-protected
```

**Tier 1 — HippoRAG (external, symbolic)**
- Ingest pipeline: chunked docs → Phi-3-mini OpenIE → (subject, predicate, object) triples → NetworkX graph.
- Retrieval: Personalized PageRank seeded by query entity matches; top-k chunks.
- Wiring: retrieved chunks prepended as context at the **6 diff-attn layers only** via cross-attention adapter (avoids Mamba layers wasting compute on retrieval tokens).
- Update cadence: per-ingest (minutes to hours).
- Storage: SQLite + NetworkX pickle. <500MB for personal-doc scale.

**Tier 2 — Titans memory in MAG configuration**
- A 2-layer MLP, d_in=1536, d_hidden=2048, d_out=1536. ~15M params.
- At every layer, runs *in parallel* with the backbone block (Mamba2 or DiffAttn). Outputs combined via learned sigmoid gate.
- Test-time update: when surprise = ‖MLP(k_t) − v_t‖² > τ_surprise, do one gradient step on MLP weights with learning rate η_titans.
- η_titans is modulated by hormones (see Tier 0).
- One memory module shared across all 24 layers (keeps params low; layer index added to key as positional bias).

**Tier 3 — Runtime LoRA**
- Rank-16 LoRA on Q, K, V, O projections of the 6 diff-attn layers + the in/out projections of all Mamba2 blocks. ~30M params.
- Updated by **two distinct triggers**:
  1. **Online (live)**: surprise-gated, hormone-modulated LR, single-step Adam updates on confirmed-positive interactions only.
  2. **Sleep training** (nightly / on idle): full epoch over the day's interaction buffer with standard LR, then snapshot.
- Snapshot ring buffer of 8 versions. Eval canary set (100 frozen prompts: tool format, RAG faithfulness, reasoning) runs every 100 online updates. Any metric drop >5% → auto-rollback to last good snapshot.

**Tier 0a — Hormone scalars (the "mood" system)**

Five EMA-updated state variables per session/agent:

```
joy          ↑ on user-accept, successful tool exec, low-retry-rate
frustration  ↑ on user-correction, tool-fail, repeated bad outputs
confidence   ↑ when prediction entropy low AND outcomes good
fatigue      ↑ with updates_since_last_snapshot
boredom      ↑ when surprise variance stays low (system not learning)
```

Modulation effects:

| Signal | Effect on system |
|---|---|
| joy↑ | LoRA LR × (1 + α·joy) on recent traces → reinforce |
| frustration↑ | LoRA LR × (1 − β·frustration); raise sampling temperature; consult HippoRAG more; if > τ_panic → rollback last snapshot, flag for user |
| confidence↑ | Skip retrieval (saves latency); lower sampling temperature |
| fatigue↑ | Force snapshot + scheduled sleep-training run |
| boredom↑ | Lower τ_surprise → Titans updates more eagerly |

**Tier 0b — Hebbian co-firing matrix H**

Tracks which features fire together in *positively-reinforced* interactions, to short-circuit slow LoRA drift.

```
Features tracked per turn (sparse one-hot or embedding ids):
  - task_intent cluster id        (k-means over query embeddings, k=512)
  - tools invoked                  (~64 tool types in v1)
  - retrieval source clusters      (k=256 HippoRAG community ids)
  - response shape                 (table / chart / prose / code, ~16 ids)

H is a sparse matrix, |F| × |F| where |F| ≈ 850.
Memory: <100MB even fully populated.

Update rule (three-factor learning):
  if joy_t > τ_joy:
    for each active pair (i,j):
       H[i,j] += η_H · joy_t · co_act(i,j)
  if frustration_t > τ_frust:
       H[i,j] *= (1 − decay · frustration_t)

H biases the tool-selection logits at inference:
  logits_tool += λ · H[task_intent_id, :] @ tool_embeddings
```

**Concrete example (the "churn → chart" loop):**
1. User: "Run a churn analysis on this CSV."
2. System invokes `analyze_churn` tool + `make_chart` tool. User accepts result.
3. `joy ↑`. `H[churn_intent_id, make_chart_id] += η·joy`.
4. After 3–4 such happy interactions, H[churn → chart] is strong.
5. Fifth churn query: tool-selection logits now bias toward `make_chart` *before* any LoRA delta has propagated. The bias is interpretable (you can `print(H[churn_id])`) and instantly reversible (set entry to zero).

This is the "fast" learning loop. LoRA is the "slow" loop. Together they implement complementary learning systems (McClelland et al. 1995).

---

## 4. Data flow at inference (one turn)

```
1.  User query arrives.
2.  Embedding model classifies task_intent (cluster id).
3.  HippoRAG retrieves top-k chunks (PPR over graph).
4.  Hebbian H[task_intent, tools] biases tool-selection prior.
5.  Forward pass through 24 hybrid layers:
       at every layer:
         backbone_out = Mamba2(x) or DiffAttn(x)
         memory_out   = TitansMLP(x)
         x_next       = backbone_out + gate ⊙ memory_out
       at the 6 DiffAttn layers:
         cross-attend to retrieved HippoRAG chunks
6.  Sampling temperature modulated by confidence/frustration.
7.  Generate response (possibly with tool calls).
8.  Tool execution; user reaction observed.
9.  Hormone scalars updated.
10. If surprise high → Titans MLP gradient step.
11. If joy high → Hebbian H update + LoRA online step.
12. Every 100 updates → run canary eval; rollback if regressed.
13. Buffer interaction for nightly sleep-training.
```

---

## 5. Training plan

### 5.1 Stages

| Stage | Tokens | Data | Compute | Cost |
|---|---|---|---|---|
| 1. Pretrain | 60B | mix below | 8×H100 spot, ~10 days | $800–1100 |
| 2. Long-ctx extension | 5B | ProLong-64K + LongAlign | 8×H100, 2 days | $150 |
| 3. SFT (general) | 1B | SmolTalk + Tulu-3 selective | 4×H100, 1 day | $80 |
| 4. SFT (tool) | 200M | Glaive-FC-v2 + Hermes-FC + APIGen | 4×H100, half day | $40 |
| 5. SFT (RAG) | 200M | RAG-Instruct + ChatQA + Self-RAG + RAGTruth | 4×H100, half day | $40 |
| 6. DPO/preference | 50M | UltraFeedback + Nectar subset | 4×H100, half day | $40 |
| 7. Hebbian / hormone bootstrap | — | UltraFeedback chosen-rejected as proxy reward | CPU + 1×GPU eval | $20 |

**Total budget: ~$1170–1470.** Tight but within the user's intent.

### 5.2 Pretraining data mix (60B tokens)

| % | Dataset | HF path | License |
|---|---|---|---|
| 40 | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ODC-By |
| 20 | DCLM-baseline | `mlfoundations/dclm-baseline-1.0` | CC-BY-4 |
| 15 | Cosmopedia-v2 | `HuggingFaceTB/cosmopedia` | Apache-2 |
| 10 | OpenWebMath | `open-web-math/open-web-math` | ODC-By |
| 8 | Stack-v2 (filtered) | `bigcode/the-stack-v2-dedup` | various permissive |
| 5 | OpenCoder corpus | `OpenCoder-LLM/opc-fineweb-code-corpus` | Apache-2 |
| 2 | Wiki + books | `wikimedia/wikipedia`, `Skylion007/openwebtext` | CC-BY-SA |

SLM-tuned ratio: heavier code/math/synthetic than Chinchilla LLMs (matches Phi-1.5, SmolLM2 findings).

### 5.3 Tool-calling data

- **Glaive-FC-v2** (113k examples, Apache-2.0) — primary.
- **Hermes-Function-Calling-V1** (NousResearch, Apache-2.0) — secondary.
- **xLAM-60k** *only* if non-commercial use is acceptable (CC-BY-NC).
- Reformatted to Granite-3 `<tool_call>{json}</tool_call>` schema with `<tool_response>...</tool_response>` returns.

### 5.4 Hormone/Hebbian bootstrap

No public dataset has paired (correction, retry, satisfaction) trajectories. Bootstrap strategy:
- Use **UltraFeedback** `chosen` examples as joy=+1, `rejected` as frustration=+1.
- Synthesize ~10k multi-turn corrections from existing SFT data via teacher (claude-haiku) labelling.
- Real signal collected post-deployment from actual usage. v1 ships with weak Hebbian H; it strengthens through use.

---

## 6. Evaluation

### 6.1 Standard benchmarks (the "minimum defensible suite")

| Component | Benchmark | Metric |
|---|---|---|
| Backbone PPL | C4, SlimPajama val | nats/byte |
| Common-sense | lm-eval-harness 7-task (LAMBADA, HellaSwag, PIQA, ARC-e/c, WinoGrande, OpenBookQA) | acc / acc-norm |
| Recall | MQAR (Zoology) | accuracy at various seq lengths |
| Long context | RULER, NIAH at 4k/16k/32k | retrieval accuracy |
| Titans claim | BABILong (`RMT-team/babilong`), S-NIAH 64k | accuracy |
| HippoRAG claim | MuSiQue, 2WikiMultiHopQA, HotpotQA | EM / F1, recall@2 |
| Tool claim | BFCL v3 | AST + exec accuracy |
| Continual learning | TRACE (8 sequential tasks) | BWT, FWT, avg-acc |
| End-to-end agent | GAIA Level-1 | task success rate |

### 6.2 Custom probes (paper-original, must invent)

1. **MAG memory-attention probe** — needle planted in Titans memory state (not context). Measure recall after k forward passes since write.
2. **Hormone gating eval** — stability-plasticity dilemma probe: alternate a "stable" task with a "novel" task; measure forgetting on stable while learning novel. Compare hormones-on vs hormones-off.
3. **Joint retrieval-writeback eval** — stream new docs into HippoRAG; query the implied facts. Measures KG mutation under load.
4. **Online-LoRA tool-schema retention** — BFCL × TRACE composition: do continual updates on miscellaneous tasks corrupt tool-calling accuracy?

### 6.3 Required ablations (for paper credibility)

Factorial across memory tiers:
- Backbone only
- + HippoRAG
- + Titans
- + HippoRAG + Titans
- + Tier 0 (hormones + Hebbian) on top of full

Factorial across modulators (2³ = 8 runs):
- Hormones {on, off} × Hebbian {on, off} × LoRA-rollback {on, off}

Plus compute-matched baselines:
- Samba-700M (pure hybrid, no memory)
- Llama-3.2-1B + RAG (off-the-shelf comparison)
- Titans-MAG-700M on pure-attention backbone (isolates hybrid-backbone contribution)

**Total: ~30 training runs of varying size.** Most are short fine-tunes from the same pretrained checkpoint. Realistic on the budget.

### 6.4 Eval harness — "FrankenBench"

Wrap lm-eval-harness + RULER + BFCL + TRACE + the 4 custom probes behind one runner. Single command: `python frankenbench.py --ckpt path/to/model --tier full`. Logs to W&B.

---

## 7. Risks and how each is mitigated

| Risk | Mitigation |
|---|---|
| Pure SSM hurts precise recall | Diff-attn layers carry recall load; ablation will show this |
| MAG memory destabilizes training | Gate initialized to 0 (memory off at start); ramp up over 5B tokens |
| Online LoRA collapses model | Frozen base + snapshot ring + canary eval + rollback |
| Reward-hacking the hormone signal | joy only from *user-confirmed* outcomes, not self-evaluations |
| HippoRAG graph extraction noisy | Use Phi-3-mini (good IE), filter triples by confidence, de-dupe via embeddings |
| Hebbian H biases too strongly | Cap λ; decay via frustration; H is interpretable so we can debug |
| Compute overrun | Stage gating: don't proceed to long-context until pretrain PPL hits target |
| Reviewer-novelty doubt | Tri-timescale framing + 30 ablation runs; thesis isn't "new block X" but "unified neuromodulation" |

---

## 8. Repo structure (planned)

```
frankenstein-slm/
├── model/
│   ├── backbone.py          # hybrid stack
│   ├── mamba2_block.py
│   ├── diff_attn_block.py
│   ├── titans_mag.py        # Tier 2
│   ├── hipporag_adapter.py  # Tier 1 wiring into attn layers
│   ├── lora_runtime.py      # Tier 3
│   └── hormones.py          # Tier 0a + 0b
├── data/
│   ├── pretrain_mix.yaml
│   ├── tool_format.py       # Granite-3 templating
│   └── rag_sft.py
├── train/
│   ├── pretrain.py          # torchtitan-based
│   ├── sft.py
│   ├── dpo.py
│   └── sleep_train.py       # nightly LoRA pass
├── runtime/
│   ├── serve.py             # online inference + memory updates
│   ├── snapshot.py          # ring buffer
│   ├── canary_eval.py
│   └── ingest.py            # HippoRAG doc ingestion
├── evals/
│   ├── frankenbench.py      # unified runner
│   ├── probes/              # the 4 custom probes
│   └── ablation_matrix.yaml
└── docs/
    ├── design.md            # this file
    ├── paper-outline.md
    └── ablation-results.md
```

---

## 9. Build order

1. **Week 1:** Set up cloud, fork torchtitan, implement hybrid backbone with stubbed Titans + HippoRAG (gates at 0). Smoke test on 100M tokens.
2. **Week 2:** Pretrain run begins. While it runs, build HippoRAG ingest + retrieval.
3. **Week 3–4:** Pretrain finishes. SFT stages. Build Titans MAG + hormone scalars.
4. **Week 5:** Tool-calling + RAG SFT. Build Hebbian H + runtime LoRA + snapshot/canary.
5. **Week 6:** FrankenBench harness. Begin ablation runs.
6. **Week 7–8:** Ablations finish. Custom probes. Paper draft.
7. **Week 9+:** Iterate on weak spots; submit.

---

## 10. Open questions deferred to implementation

- Exact RoPE base for the diff-attn layers (32k extension calibration).
- Whether to share one Titans MLP across all layers or one per attention layer (current spec: shared with layer-id bias; revisit if memory recall is weak).
- Sleep-training schedule: nightly fixed vs fatigue-triggered.
- Whether the Hebbian H should also influence retrieval (bias HippoRAG PPR seeds) — likely yes in v2.

---

## Appendix A — Why each component, in one sentence

- **Mamba2 backbone (75%):** subquadratic long context, the only way to cheaply handle 32k for documents.
- **Differential attention (25%):** denoised attention is empirically better for precise recall, tool formatting, structured output, retrieval attendance — exactly the precision-critical jobs.
- **HippoRAG (Tier 1):** symbolic, citable, slow-changing facts. The "library."
- **Titans MAG (Tier 2):** compressed neural memory of recent patterns. The "intuition."
- **Hebbian H (Tier 0b):** fast, interpretable association learning for tool/task pairings. The "habit."
- **Hormones (Tier 0a):** the unifying neuromodulator. Decides when each memory should update, how aggressively, and when to roll back. The "mood / regulator."
- **Runtime LoRA (Tier 3):** slow behavioral drift, snapshot-protected. The "personality drift."
- **Tool calling (Granite-style):** the agent's hands.

This is the Frankenstein. Each piece has a job, each piece is ablatable, the unifying thesis is paper-worthy.
