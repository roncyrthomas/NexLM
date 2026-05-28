# Frankenstein SLM — Master Construction Blueprint

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute each milestone. Steps use checkbox (`- [ ]`) syntax for tracking. Each milestone is self-contained: its **Context Brief** section gives a fresh agent everything needed to begin without reading prior milestones.

**Project goal:** Train a 700M-parameter hybrid Mamba2 + differential-attention SLM with three-tier neuromodulated memory (HippoRAG + Titans MAG + Hebbian/LoRA), evaluate it, and write a COLM-2026-shaped paper. Total wall-clock: ~9 weeks. Total cloud cost: $1170–1470.

**Companion docs:**
- Spec: `docs/superpowers/specs/2026-05-28-frankenstein-slm-design.md`
- Literature paper draft: `docs/frankenstein-slm-paper.html`

**Mode:** Direct (no git repo). Each milestone produces concrete files + a verification run. Snapshots tagged with a date string in `_archive/` directories.

---

## Adversarial-review findings applied (2026-05-28)

This blueprint was reviewed by an Opus-tier adversarial pass before lock-in. The following changes were applied to address CRITICAL and HIGH findings; all change locations are tagged in-place with `⟂ REVIEW`.

- **M3 time/cost re-grounded.** Back-of-envelope: 700M params × 60B tokens × 6 FLOPs = 2.5e20 FLOPs. 8×H100 bf16 peak ~8 PFLOP/s. At 25% MFU (conservative for a two-kernel hybrid) = 2 PFLOP/s sustained → ~35 hours wall-clock. The original "10–14 days" assumed sub-10% MFU. Real plan: 2–4 days wall-clock at 25–35% MFU, $200–500 spot. If MFU < 15% at 24h, **stop and profile** before continuing. Budget contingency 25% added.
- **M6 Titans update fixed.** Replaced `.backward()` with `torch.autograd.grad()`; added explicit WK / WV projections (the surprise objective is now associative `‖MLP(WK·x) − WV·x‖²`, not identity); `TitansMAG` kept unsharded (DDP) so FSDP does not corrupt the inner update; fp32 master copies during the inner step.
- **M8 ablation matrix expanded** from "13+ runs" to the honest 5×2³ + 3 = 43-run factorial *with 3 seeds* needed to defend the unifying neuromodulation thesis at COLM. Budget revised to $500–700.
- **M1a→M1b discontinuity** added to anti-pattern catalog: M1a weights are throwaway; the vocab/RoPE base/seq-len change between M1a and M1b makes the checkpoint unloadable.
- **M1a DiffAttn RoPE order** fixed: RoPE now applied to q1/q2/k1/k2 individually *before* GQA expansion; without this, k2 receives no RoPE and the second attention map loses position equivariance.
- **Titans gate added to always-trainable set** in M7 — it is a backbone parameter, not LoRA, but must remain mutable during online updates and sleep training.
- **Hebbian storage** switched from sparse COO (unsupported in-place indexing in PyTorch) to dense float32 850×850 = 2.7MB.
- **Canary curation moved to M4 Step 0** from held-out splits, with seed documented — prevents accidental training-set contamination of the rollback signal.
- **Stop-loss gates** added at the bottom of this file: hard budget/time gates that force scope cuts rather than silent overruns.
- **File-split pre-plan**: `model/rope.py` and `model/attention_ops.py` carved out of `diff_attn_block.py` in M1a Step 4 to preempt the 600-line invariant violation.

Findings deferred (MEDIUM/LOW) are tracked in the milestone-level notes.

---

## Dependency Graph

```
                     ┌────────────────┐
                     │ M1a smoke      │ (TinyStories validation)
                     └────────┬───────┘
                              ▼
                     ┌────────────────┐
                     │ M1b backbone   │ (700M-shape code)
                     └────────┬───────┘
                              ▼
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│ M2 pretrain  │     │ M5 HippoRAG    │     │ M6 Titans MAG│
│  infra       │     │  pipeline      │     │  module      │
└──────┬───────┘     └────────┬───────┘     └──────┬───────┘
       ▼                      │                    │
┌──────────────┐               │                    │
│ M3 pretrain  │               │                    │
│  60B tokens  │               │                    │
└──────┬───────┘               │                    │
       ▼                       │                    │
┌──────────────┐               │                    │
│ M4 SFT       │               │                    │
│  4 stages    │               │                    │
└──────┬───────┘               │                    │
       ▼                       │                    │
┌──────────────────────────────┼────────────────────┘
▼                              ▼
┌──────────────┐     ┌────────────────┐
│ M7 hormones  │ ◄── │  integrate M5  │
│  Hebbian/LoRA│     │  and M6 into   │
└──────┬───────┘     │  the trained   │
       ▼             │  model         │
┌──────────────┐     └────────────────┘
│ M8 evals     │
│  ablations   │
└──────┬───────┘
       ▼
┌──────────────┐     ┌──────────────┐
│ M9 serving   │     │ M10 paper    │
│  stack       │     │  write-up    │
└──────────────┘     └──────────────┘
```

**Parallelism opportunities (P):**
- M5 (HippoRAG) and M6 (Titans) can be built in parallel with M2/M3 — they touch zero shared files with training infra.
- M8 (FrankenBench eval harness) authoring can begin during M4 — needs only the model checkpoint format, not specific weights.
- M10 (paper) introduction + related-work sections can be drafted any time after M1b.

**Critical path:** M1a → M1b → M2 → M3 → M4 → M7 → M8 → M10. ~7 weeks wall-clock.

---

## Milestone tier and model assignment

| Milestone | Difficulty | Recommended executor model | Why |
|---|---|---|---|
| M1a | Low | sonnet | Mechanical PyTorch, well-precedented |
| M1b | Medium | sonnet | Larger but same patterns as M1a |
| M2 | Medium | sonnet | Infrastructure boilerplate |
| M3 | Low (mostly waiting) | sonnet | Monitor + intervene |
| M4 | Medium | sonnet | Standard SFT/DPO recipes |
| M5 | Medium | sonnet | Graph code, no model novelty |
| M6 | **High** | **opus** | Test-time gradient updates are tricky |
| M7 | **High** | **opus** | Three coupled systems; correctness-critical |
| M8 | Medium | sonnet | Wrapping existing harnesses |
| M9 | Low | sonnet/haiku | FastAPI + standard tooling |
| M10 | Medium | sonnet, with opus review | Writing |

---

## Cross-cutting invariants

These must hold after every milestone. Each milestone's verification block re-checks them.

1. `python -m pytest tests/` exits 0.
2. `python scripts/smoke_forward.py --ckpt <latest>` exits 0 with a one-line tensor-shape report.
3. Total parameter count of the model is within ±10% of the target for the current milestone size.
4. No file in `model/` exceeds 600 lines (refactor before commit if so).
5. `pip-compile requirements.in` produces no new pinned-version drift unless intentional.

---

## File layout (locked at M1a)

```
frankenstein-slm/
├── model/
│   ├── __init__.py
│   ├── config.py             ← all hyperparams in one dataclass
│   ├── mamba2_block.py       ← SSM block
│   ├── diff_attn_block.py    ← differential attention block
│   ├── backbone.py           ← 24-layer hybrid stack assembly
│   ├── titans_mag.py         ← Tier 2 (added in M6)
│   ├── lora_runtime.py       ← Tier 3 (added in M7)
│   ├── hormones.py           ← Tier 0a (added in M7)
│   ├── hebbian.py            ← Tier 0b (added in M7)
│   └── tokenizer_util.py
├── data/
│   ├── pretrain_mix.yaml
│   ├── tool_format.py
│   ├── rag_format.py
│   └── streaming.py
├── train/
│   ├── pretrain.py
│   ├── sft.py
│   ├── dpo.py
│   ├── sleep_train.py
│   └── canary_eval.py
├── runtime/
│   ├── serve.py
│   ├── snapshot.py
│   ├── ingest.py             ← HippoRAG ingest
│   ├── retrieve.py           ← HippoRAG PPR retrieval
│   └── hormone_state.py      ← deployment-time hormone tracking
├── evals/
│   ├── frankenbench.py
│   ├── probes/
│   │   ├── mag_memory_needle.py
│   │   ├── hormone_gating.py
│   │   ├── joint_retrieval_writeback.py
│   │   └── lora_tool_retention.py
│   ├── ablation_matrix.yaml
│   └── results/
├── scripts/
│   ├── smoke_forward.py
│   ├── count_params.py
│   ├── download_data.py
│   └── plot_loss.py
├── tests/
│   ├── test_mamba2_block.py
│   ├── test_diff_attn_block.py
│   ├── test_backbone.py
│   ├── test_titans_mag.py
│   ├── test_hormones.py
│   ├── test_hebbian.py
│   ├── test_lora_runtime.py
│   ├── test_hipporag_ingest.py
│   ├── test_retrieve.py
│   └── test_canary_rollback.py
├── configs/
│   ├── smoke_30m.yaml
│   ├── pretrain_700m.yaml
│   ├── sft_general.yaml
│   ├── sft_tool.yaml
│   ├── sft_rag.yaml
│   └── dpo.yaml
├── requirements.in
├── requirements.txt
└── README.md
```

---

# M1a — Tiny Smoke Model

**Wall-clock:** 2–3 days
**Cost:** ~$10 cloud OR free if you have a 3090/4090/5090 locally
**Depends on:** none
**Parallelizable with:** —

## Context Brief

You are starting the Frankenstein SLM project. Before spending money on a real pretrain run, build a **30M-parameter** smoke version of the hybrid backbone and train it on TinyStories. The goals are: (a) prove the hybrid Mamba2 + differential-attention pattern trains stably, (b) lock in the file layout and hyperparameter dataclass shape, (c) validate the stubbed Titans MAG gate (initialized to zero) does not break gradient flow. This is a learning vehicle as much as an engineering checkpoint — every line of code here will be reused in M1b.

**Architecture for M1a (smaller than the production design):**
- 6 layers total: 4 Mamba2 + 2 DiffAttn (at positions 3 and 6)
- `d_model` = 256
- 4 attention heads, head_dim 64, GQA with 2 kv-heads
- SwiGLU FFN, expansion 2.67×
- Vocab: GPT-2 BPE (50257) — defer Phi-3 tokenizer to M1b
- Sequence length: 512
- Total params: ~30M

**Reference papers to skim before starting:**
1. Mamba2 (arXiv 2405.21060) — Section 3 on the SSD scan
2. Differential Transformer (arXiv 2410.05258) — Section 2, the two-attention subtraction
3. Samba (arXiv 2406.07522) — Section 3, the layer interleaving pattern
4. Karpathy's nanoGPT — fork as the trainer skeleton

**Dataset:** roneneldan/TinyStories (HuggingFace) — ~2GB, 470M tokens.

**Success criterion:** Validation loss < 2.0 after 30k steps (~3 hours on a 4090, ~$8 on a Lambda A10).

## Files

- Create: `model/config.py`, `model/rope.py`, `model/attention_ops.py`, `model/mamba2_block.py`, `model/diff_attn_block.py`, `model/backbone.py`, `model/titans_mag.py` (stub), `model/tokenizer_util.py` ⟂ REVIEW M1: pre-split RoPE and attention ops into their own files now, before `diff_attn_block.py` outgrows 600 lines later.
- Create: `train/pretrain.py`, `scripts/smoke_forward.py`, `scripts/count_params.py`
- Create: `tests/test_mamba2_block.py`, `tests/test_diff_attn_block.py`, `tests/test_backbone.py`, `tests/test_titans_mag.py`
- Create: `configs/smoke_30m.yaml`
- Create: `requirements.in`, `requirements.txt`, `README.md`

## Tasks

- [ ] **Step 1: Set up project skeleton + virtualenv**

```bash
mkdir frankenstein-slm && cd frankenstein-slm
python -m venv .venv
.venv\Scripts\activate              # PowerShell on Windows
pip install --upgrade pip pip-tools
```

Create `requirements.in`:
```
torch>=2.4
mamba-ssm>=2.2.2
causal-conv1d>=1.4
transformers>=4.45
datasets>=2.20
wandb
pyyaml
einops
pytest
pytest-xdist
```

Then:
```bash
pip-compile requirements.in
pip install -r requirements.txt
```

- [ ] **Step 2: Write the config dataclass with tests**

Test (`tests/test_backbone.py` — first failing test):
```python
from model.config import ModelConfig

def test_default_30m_config():
    cfg = ModelConfig.smoke_30m()
    assert cfg.d_model == 256
    assert cfg.n_layers == 6
    assert cfg.attn_layer_positions == [3, 6]
    assert cfg.n_heads == 4
    assert cfg.head_dim == 64
    assert cfg.n_kv_heads == 2
```

Run: `pytest tests/test_backbone.py::test_default_30m_config -v`
Expected: FAIL with `ModuleNotFoundError`.

Then implement `model/config.py`:
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    d_model: int = 1536
    n_layers: int = 24
    attn_layer_positions: List[int] = field(
        default_factory=lambda: [4, 8, 12, 16, 20, 24])
    n_heads: int = 12
    head_dim: int = 128
    n_kv_heads: int = 4
    mamba_state_dim: int = 128
    mamba_expansion: int = 2
    ffn_expansion: float = 2.67
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rope_base: float = 10000.0
    titans_enabled: bool = False
    titans_hidden: int = 2048

    @classmethod
    def smoke_30m(cls):
        return cls(
            d_model=256, n_layers=6,
            attn_layer_positions=[3, 6],
            n_heads=4, head_dim=64, n_kv_heads=2,
            mamba_state_dim=64, vocab_size=50257,
            max_seq_len=512, titans_enabled=False,
        )
```

Run: `pytest tests/test_backbone.py::test_default_30m_config -v` → PASS.

- [ ] **Step 3: Mamba2 block (use mamba-ssm package directly, do not reimplement)**

Test:
```python
import torch
from model.config import ModelConfig
from model.mamba2_block import Mamba2Block

def test_mamba2_forward_shape():
    cfg = ModelConfig.smoke_30m()
    block = Mamba2Block(cfg)
    x = torch.randn(2, 64, cfg.d_model)
    out = block(x)
    assert out.shape == x.shape

def test_mamba2_grad_flows():
    cfg = ModelConfig.smoke_30m()
    block = Mamba2Block(cfg)
    x = torch.randn(2, 64, cfg.d_model, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

Implementation `model/mamba2_block.py`:
```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from model.config import ModelConfig

class Mamba2Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model)
        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.mamba_state_dim,
            expand=cfg.mamba_expansion,
            headdim=64,
        )
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        d_ff = int(cfg.d_model * cfg.ffn_expansion)
        self.ffn_w1 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.ffn_w2 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.ffn_w3 = nn.Linear(d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        x = x + self.mamba(self.norm(x))
        n = self.ffn_norm(x)
        x = x + self.ffn_w3(torch.nn.functional.silu(self.ffn_w1(n)) * self.ffn_w2(n))
        return x
```

Verify: `pytest tests/test_mamba2_block.py -v` → 2 passed.

- [ ] **Step 4: Differential attention block**

Test:
```python
import torch
from model.config import ModelConfig
from model.diff_attn_block import DiffAttnBlock

def test_diff_attn_forward_shape():
    cfg = ModelConfig.smoke_30m()
    block = DiffAttnBlock(cfg, layer_idx=2)
    x = torch.randn(2, 64, cfg.d_model)
    out = block(x)
    assert out.shape == x.shape

def test_diff_attn_lambda_init():
    cfg = ModelConfig.smoke_30m()
    b1 = DiffAttnBlock(cfg, layer_idx=0)
    b2 = DiffAttnBlock(cfg, layer_idx=5)
    # lambda init depends on depth
    assert b1.lambda_init.item() != b2.lambda_init.item()
```

Implementation (`model/diff_attn_block.py` — abbreviated, full code in spec section A):
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class DiffAttnBlock(nn.Module):
    """Differential attention from arXiv:2410.05258, with GQA."""
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.norm = nn.RMSNorm(cfg.d_model)
        # Q and K projections doubled to produce two attention maps
        self.q_proj = nn.Linear(cfg.d_model, 2 * cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, 2 * cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        # lambda parameters per the paper
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_init = nn.Parameter(torch.tensor(lambda_init))
        self.lambda_q1 = nn.Parameter(torch.zeros(cfg.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(cfg.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(cfg.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(cfg.head_dim).normal_(0, 0.1))
        # FFN
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        d_ff = int(cfg.d_model * cfg.ffn_expansion)
        self.ffn_w1 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.ffn_w2 = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.ffn_w3 = nn.Linear(d_ff, cfg.d_model, bias=False)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, _ = x.shape
        h = self.norm(x)
        # split BEFORE rope+GQA so both q1/q2/k1/k2 get position info ⟂ REVIEW H1
        q = self.q_proj(h).view(B, T, 2 * self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 2 * self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # split for differential attention FIRST
        q1, q2 = q[:, :self.n_heads], q[:, self.n_heads:]
        k1_kv, k2_kv = k[:, :self.n_kv_heads], k[:, self.n_kv_heads:]
        # apply RoPE to all four BEFORE GQA expansion
        if rope_cos is not None:
            q1 = apply_rope(q1, rope_cos, rope_sin)
            q2 = apply_rope(q2, rope_cos, rope_sin)
            k1_kv = apply_rope(k1_kv, rope_cos, rope_sin)
            k2_kv = apply_rope(k2_kv, rope_cos, rope_sin)
        # NOW expand K1, K2, V to n_heads via GQA repeat
        rep = self.n_heads // self.n_kv_heads
        k1 = k1_kv.repeat_interleave(rep, dim=1)
        k2 = k2_kv.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scale = 1.0 / math.sqrt(self.head_dim)
        # two attention maps with causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), 1)
        a1 = F.softmax((q1 @ k1.transpose(-1, -2)) * scale + mask, dim=-1)
        a2 = F.softmax((q2 @ k2.transpose(-1, -2)) * scale + mask, dim=-1)
        lam = (torch.exp((self.lambda_q1 * self.lambda_k1).sum())
             - torch.exp((self.lambda_q2 * self.lambda_k2).sum())
             + self.lambda_init)
        attn = (a1 - lam * a2) @ v
        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        x = x + self.o_proj(attn)
        n = self.ffn_norm(x)
        x = x + self.ffn_w3(F.silu(self.ffn_w1(n)) * self.ffn_w2(n))
        return x
```

Verify: `pytest tests/test_diff_attn_block.py -v` → 2 passed.

- [ ] **Step 5: Stubbed Titans MAG (gate at zero, but plumbing in place)**

Test:
```python
import torch
from model.config import ModelConfig
from model.titans_mag import TitansMAG

def test_titans_stub_passthrough():
    cfg = ModelConfig.smoke_30m()
    cfg.titans_enabled = True  # plumbing on
    mem = TitansMAG(cfg)
    x = torch.randn(2, 64, cfg.d_model)
    out = mem(x, layer_idx=0)
    # gate initialized to zero -> output is zero -> backbone unaffected after add
    assert out.abs().max() < 1e-6
```

Implementation (`model/titans_mag.py`):
```python
import torch
import torch.nn as nn
from model.config import ModelConfig

class TitansMAG(nn.Module):
    """Tier 2 memory; stub in M1a — full surprise-gated updates added in M6."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model + 16, cfg.titans_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.titans_hidden, cfg.d_model, bias=False),
        )
        self.layer_embed = nn.Embedding(cfg.n_layers, 16)
        # gate initialized to zero — memory contributes nothing until M6 ramps it up
        self.gate = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, x, layer_idx: int):
        if not self.cfg.titans_enabled:
            return torch.zeros_like(x)
        B, T, D = x.shape
        le = self.layer_embed(torch.tensor(layer_idx, device=x.device))
        le = le.expand(B, T, -1)
        out = self.mlp(torch.cat([x, le], dim=-1))
        return torch.sigmoid(self.gate) * out
```

Verify: `pytest tests/test_titans_mag.py -v` → 1 passed.

- [ ] **Step 6: Backbone assembly**

Test:
```python
import torch
from model.config import ModelConfig
from model.backbone import Frankenstein

def test_backbone_30m_param_count():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert 25_000_000 < n < 35_000_000, f"params={n:,}"

def test_backbone_forward():
    cfg = ModelConfig.smoke_30m()
    model = Frankenstein(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    logits = model(ids)
    assert logits.shape == (2, 64, cfg.vocab_size)
```

Implementation (`model/backbone.py`):
```python
import torch
import torch.nn as nn
from model.config import ModelConfig
from model.mamba2_block import Mamba2Block
from model.diff_attn_block import DiffAttnBlock
from model.titans_mag import TitansMAG

class Frankenstein(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        attn_set = set(cfg.attn_layer_positions)  # 1-indexed
        self.layers = nn.ModuleList([
            DiffAttnBlock(cfg, layer_idx=i) if (i + 1) in attn_set
            else Mamba2Block(cfg)
            for i in range(cfg.n_layers)
        ])
        self.titans = TitansMAG(cfg)
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # tie embedding and lm_head
        self.lm_head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for i, layer in enumerate(self.layers):
            x = layer(x) + self.titans(x, layer_idx=i)
        return self.lm_head(self.final_norm(x))
```

Verify: `pytest tests/test_backbone.py -v` → 3 passed (incl. earlier config test).
Also: `python scripts/count_params.py --config configs/smoke_30m.yaml` should print ~30M.

- [ ] **Step 7: Trainer (nanoGPT-style)**

Write `train/pretrain.py` based on Karpathy's nanoGPT but loading our `Frankenstein` model. Key bits:
- AdamW, lr 3e-4, cosine schedule with 1000-step warmup
- Batch size 32, gradient accumulation if needed
- bf16 autocast on GPU
- Save checkpoint every 5000 steps
- Eval on TinyStories val every 1000 steps
- Log to wandb under project `frankenstein-smoke`

Concrete training command:
```bash
python train/pretrain.py \
    --config configs/smoke_30m.yaml \
    --data tinystories \
    --batch_size 32 \
    --lr 3e-4 \
    --max_steps 30000 \
    --val_every 1000 \
    --save_every 5000 \
    --wandb_project frankenstein-smoke
```

- [ ] **Step 8: Run the smoke train**

Run for 30k steps. ~3 hours on a 4090 / 6 hours on an A10.

- [ ] **Step 9: Verify success criterion**

Check W&B run: validation loss < 2.0 by step 30k. If not, investigate (likely LR or grad clip).

Also run `scripts/smoke_forward.py --ckpt out/smoke_30m_final.pt --prompt "Once upon a time"` and confirm a coherent TinyStories continuation.

## Verification

```bash
pytest tests/ -v                                    # all green
python scripts/count_params.py --config configs/smoke_30m.yaml  # ~30M
python scripts/smoke_forward.py --ckpt out/smoke_30m_final.pt --prompt "Once upon"
```

## Exit criteria

- TinyStories val loss < 2.0 at 30k steps.
- All tests passing.
- Generated continuation reads as a TinyStories sentence (not gibberish).
- M1a deliverable committed to local `_archive/m1a-YYYY-MM-DD/`.

## Rollback strategy

If val loss plateaus above 2.5 or NaNs: bisect by disabling diff-attn (replace with vanilla MHA) and re-running 5k steps to isolate whether the issue is in the attention block, the Mamba2 hookup, or the data pipeline.

---

# M1b — Full 700M Backbone Code

**Wall-clock:** 3–5 days
**Cost:** ~$10 for verification runs
**Depends on:** M1a complete
**Parallelizable with:** —

## Context Brief

M1a proved the hybrid pattern trains. Now scale the code to production hyperparameters (`d_model`=1536, 24 layers, 700M params) and integrate Phi-3 tokenizer. No real training yet — this milestone produces a model object that loads correctly, forward-passes the right tensor shapes, and shows a sensible per-token loss on a 1B-token sanity stream. Everything is the same code path as M1a; you are just swapping the config.

**Critical additions:**
- 16 special tokens added to Phi-3 tokenizer: `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<retrieve>`, `</retrieve>`, `<mem_read>`, `</mem_read>`, `<mem_write>`, `</mem_write>`, `<doc>`, `</doc>`, `<chunk>`, `</chunk>`, `<sys>`, `</sys>`.
- Activation checkpointing for memory.
- FSDP wrapping in trainer (M2 will use it; M1b just needs the model FSDP-friendly).

## Files

- Modify: `model/config.py` (add `production_700m` classmethod)
- Modify: `model/tokenizer_util.py` (Phi-3 + special tokens)
- Modify: `model/diff_attn_block.py` (add RoPE base 1e6 for 32k extension later)
- Create: `configs/pretrain_700m.yaml`
- Modify: existing tests for the larger shape

## Tasks

- [ ] **Step 1: Extend ModelConfig with production_700m classmethod**

```python
@classmethod
def production_700m(cls):
    return cls(
        d_model=1536, n_layers=24,
        attn_layer_positions=[4, 8, 12, 16, 20, 24],
        n_heads=12, head_dim=128, n_kv_heads=4,
        mamba_state_dim=128, vocab_size=32016,  # phi-3 + 16 specials
        max_seq_len=4096, titans_enabled=False,
        rope_base=1_000_000.0,
    )
```

- [ ] **Step 2: Implement tokenizer_util.py**

```python
from transformers import AutoTokenizer

SPECIAL_TOKENS = [
    "<tool_call>","</tool_call>","<tool_response>","</tool_response>",
    "<retrieve>","</retrieve>","<mem_read>","</mem_read>",
    "<mem_write>","</mem_write>","<doc>","</doc>",
    "<chunk>","</chunk>","<sys>","</sys>",
]

def load_tokenizer():
    tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok
```

Test: `len(tok) == 32016`.

- [ ] **Step 3: Update backbone to handle larger vocab and tied embedding**

(No code change needed if M1a was written correctly — the tie already exists. Just re-verify param count.)

- [ ] **Step 4: Param count check**

```bash
python scripts/count_params.py --config configs/pretrain_700m.yaml
```

Expected: within 650M–750M.

- [ ] **Step 5: Activation checkpointing wrapper**

Add `--gradient_checkpointing` flag to `train/pretrain.py`. Wrap each layer with `torch.utils.checkpoint.checkpoint`.

- [ ] **Step 6: 1B-token sanity stream**

Stream FineWeb-Edu for 5000 steps on a rented A100. Verify loss decreases and no NaNs. ~$8 on Lambda.

```bash
python train/pretrain.py \
    --config configs/pretrain_700m.yaml \
    --data fineweb-edu --batch_size 8 --grad_accum 8 \
    --max_steps 5000 --val_every 500 --save_every 2500 \
    --wandb_project frankenstein-700m-sanity
```

## Verification

- All tests pass with 700m config substituted where applicable.
- 5k-step sanity run shows loss curve descending below initial entropy (~10.4 at vocab 32016).
- Memory fits on A100 80GB with grad accum 8 and bf16.

## Exit criteria

- 5k-step sanity loss < 5.0 (well below initial 10.4).
- Snapshot saved.

---

# M2 — Pretrain Infrastructure

**Wall-clock:** 5–7 days
**Cost:** ~$50 (data download, infra testing)
**Depends on:** M1b
**Parallelizable with:** M5, M6 (no shared files)

## Context Brief

Build the production pretrain pipeline: streaming data loader for the 7-corpus mix, FSDP wrapping, checkpoint format compatible with HuggingFace, W&B integration, and resume-from-checkpoint logic. No actual long training run yet — that is M3. This milestone ends when a 24-hour test run on 5B tokens demonstrates stable loss curves and clean checkpoint resume.

## Files

- Modify: `train/pretrain.py` (FSDP + streaming)
- Create: `data/streaming.py` (multi-corpus interleaving)
- Create: `data/pretrain_mix.yaml`
- Create: `scripts/download_data.py`

## Tasks

- [ ] **Step 1: Pretrain mix YAML**

`data/pretrain_mix.yaml`:
```yaml
corpora:
  - name: fineweb-edu
    path: HuggingFaceFW/fineweb-edu
    ratio: 0.40
    split: train
  - name: dclm
    path: mlfoundations/dclm-baseline-1.0
    ratio: 0.20
  - name: cosmopedia
    path: HuggingFaceTB/cosmopedia
    ratio: 0.15
  - name: openwebmath
    path: open-web-math/open-web-math
    ratio: 0.10
  - name: stack-v2
    path: bigcode/the-stack-v2-dedup
    ratio: 0.08
  - name: opencoder
    path: OpenCoder-LLM/opc-fineweb-code-corpus
    ratio: 0.05
  - name: wikibooks
    path: wikimedia/wikipedia
    ratio: 0.02
total_tokens: 60_000_000_000
```

- [ ] **Step 2: Implement streaming multi-corpus loader**

Use HuggingFace `datasets.interleave_datasets` with `stopping_strategy="all_exhausted"` and the ratios from the YAML. Tokenize on the fly using `tokenizer_util.load_tokenizer()`. Buffer with prefetch=8.

- [ ] **Step 3: FSDP wrapping**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Mamba2Block, DiffAttnBlock},
    ),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

- [ ] **Step 4: Checkpoint save/load with HF-format export**

Save both: (a) FSDP shard checkpoints for resume, (b) HF-format single-file `model.safetensors` for downstream use.

- [ ] **Step 5: 5B-token test run**

```bash
torchrun --nproc_per_node=8 train/pretrain.py \
    --config configs/pretrain_700m.yaml \
    --data data/pretrain_mix.yaml \
    --max_tokens 5_000_000_000 \
    --batch_size 4 --grad_accum 16 \
    --val_every 5000 --save_every 10000
```

~24 hours on 8×H100 spot. ~$80.

- [ ] **Step 6: Verify checkpoint resume**

Kill the run at 2B tokens, resume, confirm loss curve continues smoothly.

## Verification

- 5B-token loss curve monotone-decreasing, no spikes >2σ above moving average.
- Resume produces identical losses ±0.01 after 100 steps.
- HF-format export loads with `AutoModel.from_pretrained` (custom modeling class registered).

## Exit criteria

- Pipeline can run 60B tokens unattended for 10 days.
- Snapshot saved.

---

# M3 — Pretrain Run (60B tokens)

**Wall-clock:** 2–4 days at 25–35% MFU; up to 7 days at 15% MFU ⟂ REVIEW C1
**Cost:** $200–500 (with 25% contingency: $250–625)
**Depends on:** M2
**Parallelizable with:** M5, M6, M8 authoring

## Context Brief

Execute the full 60B-token pretrain on 8×H100 spot. Your job: babysit. Watch loss curves, intervene on instabilities (spike → reduce LR, restart from last good checkpoint), and produce the final base checkpoint that all downstream milestones depend on.

**MFU sanity math (do this before launch):** 700M params × 60B tokens × 6 FLOPs/param/token = 2.5e20 FLOPs. 8×H100 bf16 peak ≈ 8 PFLOP/s. Target sustained MFU ≥ 25% → ~35 hours wall-clock at ideal; budget 2–4 days realistic accounting for restarts and preemptions. **If after 24 hours of clean running MFU is < 15%, STOP and profile** — the two-kernel hybrid (Mamba2 SSD + diff-attn) commonly underperforms vs. expected; root-cause before continuing. Common culprits: Mamba2 CUDA kernels not finding their fast path, FSDP communication bottleneck, dataloader starvation.

**Spot-pricing realism:** May 2026 spot 8×H100 nodes are frequently unavailable during business hours. Maintain a fallback to 4×H100 on-demand (will roughly double wall-clock, stay within cost envelope at $30–40/hr).

## Tasks

- [ ] **Step 1: Launch the run**

Same command as M2 step 5 but `--max_tokens 60_000_000_000`.

- [ ] **Step 2: Daily monitoring**

Check W&B dashboard each morning. Watch for:
- Loss spikes (>2σ above EMA → likely bad batch; reduce LR or restart)
- NaN gradients (immediate stop, investigate)
- Spot instance preemption (auto-resume from checkpoint)

- [ ] **Step 3: Intermediate evals at 10B, 20B, 40B, 60B**

Run `lm-eval-harness` on a small common-sense suite (HellaSwag, PIQA, ARC-e) to track downstream-task acquisition curve. Expect HellaSwag acc to cross 50% around 20B tokens for a healthy 700M.

- [ ] **Step 4: Long-context extension**

After 60B pretrain done, extend to 32k context with ProLong-64K + LongAlign for 5B tokens. RoPE base already set to 1e6 in M1b.

```bash
torchrun --nproc_per_node=8 train/pretrain.py \
    --resume out/700m_60b_final.pt \
    --config configs/pretrain_700m.yaml \
    --max_seq_len 32768 \
    --data prolong-64k,longalign \
    --max_tokens 5_000_000_000 \
    --lr 3e-5
```

## Verification

- Final val PPL on C4 < 12.0 (rough target for healthy 700M).
- HellaSwag > 45%, PIQA > 65% (Phi-1.5 territory at 700M).
- 32k NIAH retrieval > 90% accuracy at depth 50%.

## Exit criteria

- Base + long-context checkpoint archived.
- Snapshot saved.

---

# M4 — SFT Stages

**Wall-clock:** 3–5 days
**Cost:** ~$200
**Depends on:** M3
**Parallelizable with:** M5 finalization, M8 authoring

## Context Brief

Four SFT stages run sequentially on the long-context base checkpoint: general, tool-calling, RAG, then DPO. Each stage is 4–8 hours on 4×H100.

## Tasks

- [ ] **Step 0: Curate the canary eval set BEFORE any SFT touches the model** ⟂ REVIEW M3

Authoring the canary in week 5+ risks contamination — examples leaked into M4 training data would invalidate the rollback signal. Do this NOW, deterministically:

```bash
python scripts/build_canary.py \
    --tool_src Salesforce/xlam-function-calling-60k \
    --rag_src nvidia/ChatRAG-Bench \
    --reason_src HuggingFaceH4/MATH-500 \
    --n_per_source 33 \
    --seed 20260528 \
    --output evals/canary/canary_v1.jsonl
```

The 100 prompts (33 tool + 33 RAG + 34 reasoning) are extracted from held-out splits, with the SHA-256 of every prompt recorded in `evals/canary/canary_v1.manifest.json`. Any SFT data loader in later steps MUST filter rows whose SHA matches the manifest.

- [ ] **Step 1: General SFT — SmolTalk + Tulu-3 selective (1B tokens)**

```bash
torchrun --nproc_per_node=4 train/sft.py \
    --resume out/700m_60b_longctx.pt \
    --data smoltalk,tulu3-selective \
    --max_tokens 1_000_000_000 \
    --lr 2e-5 --batch_size 8 \
    --output out/sft-general.pt
```

- [ ] **Step 2: Tool-calling SFT — Glaive-FC-v2 + Hermes-FC (200M tokens)**

Reformat both datasets to Granite-3 `<tool_call>{json}</tool_call>` schema using `data/tool_format.py`. Eval on BFCL v3 every 500 steps.

- [ ] **Step 3: RAG SFT — RAG-Instruct + ChatQA + Self-RAG + RAGTruth (200M tokens)**

Reformat to use `<retrieve>...</retrieve>` and `<chunk>...</chunk>` tokens.

- [ ] **Step 4: DPO — UltraFeedback + Nectar subset (50M tokens)**

Standard DPO with `beta=0.1`.

## Verification

- BFCL v3 score > 60 after Step 2 (Phi-3-mini equivalent).
- MuSiQue EM > 25 after Step 3.
- DPO does not regress BFCL by more than 3 points.

## Exit criteria

- Final SFT+DPO checkpoint: `out/sft-final.pt`.

---

# M5 — HippoRAG Ingest and Retrieval

**Wall-clock:** 3–5 days
**Cost:** ~$30 (Phi-3-mini API calls for IE if not local)
**Depends on:** none (parallelizable with M2/M3/M4)
**Parallelizable with:** all training stages

## Context Brief

Build the HippoRAG knowledge-graph pipeline as a standalone module that can ingest a directory of documents and respond to PPR retrieval queries. This ships separately from the model and is integrated in M7. Reference paper: arXiv:2405.14831.

## Files

- Create: `runtime/ingest.py`, `runtime/retrieve.py`
- Create: `tests/test_hipporag_ingest.py`, `tests/test_retrieve.py`

## Tasks

- [ ] **Step 1: OpenIE pipeline using Phi-3-mini**

Prompt Phi-3-mini to extract triples from chunked documents. Output schema: `[{"subject": ..., "predicate": ..., "object": ..., "chunk_id": ...}]`. Confidence filter at 0.7.

- [ ] **Step 2: NetworkX graph construction**

Nodes are entities (canonicalized via embedding similarity > 0.9). Edges carry predicate strings and chunk pointers. Store graph as pickle; chunks in SQLite.

- [ ] **Step 3: Personalized PageRank retrieval**

```python
import networkx as nx
def ppr_retrieve(query, graph, k=8):
    seeds = match_entities(query, graph)
    personalization = {n: 1.0 if n in seeds else 0.0 for n in graph}
    scores = nx.pagerank(graph, personalization=personalization, alpha=0.5)
    top_nodes = sorted(scores, key=scores.get, reverse=True)[:k]
    return chunks_for(top_nodes)
```

- [ ] **Step 4: Eval on MuSiQue dev**

Use HippoRAG paper's eval protocol. Target recall@2 > 0.55 (their reported number for similar-scale IE backends).

## Verification

```bash
python -m runtime.ingest --docs ./test_docs/
python -m runtime.retrieve --query "what is churn rate"
pytest tests/test_hipporag_ingest.py tests/test_retrieve.py -v
```

## Exit criteria

- Standalone retrieval works end-to-end on a 100-doc test corpus.

---

# M6 — Titans MAG Memory Module

**Wall-clock:** 4–6 days
**Cost:** ~$100 (test-time gradient is tricky, expect debugging)
**Depends on:** M1b
**Parallelizable with:** M2/M3/M4
**Recommended executor model:** opus

## Context Brief

Replace the M1a stub with the real Titans MAG implementation: surprise-gated test-time gradient updates of the memory MLP weights, layer-shared with positional bias. The tricky parts: (a) autograd on memory weights while the backbone is in inference mode, (b) keeping the memory's optimizer state per-session, (c) the gate-ramp schedule that brings the gate from 0 → effective over the first 5B fine-tune tokens.

## Inputs from spec (cold-start required reading) ⟂ REVIEW C5

If executing this milestone fresh, you need these definitions before writing any code:

- **`tau_surprise`**: scalar threshold for triggering an inner update. Initialize 0.5; tune later. Modulated by `boredom` hormone at runtime (M7).
- **`eta_titans`**: inner-loop learning rate for the memory MLP. Initialize 1e-3.
- **Surprise metric**: per-token squared associative reconstruction error `‖MLP(WK·x) − WV·x‖²`, mean-reduced over the sequence. The keys and values are *learned projections* of the layer input, NOT raw `x` (Titans paper §3.1). Avoid the identity-objective trap.
- **Inner-update isolation**: TitansMAG MUST be excluded from FSDP sharding. Use FSDP `ignored_modules=[model.titans]`. Inner updates use `torch.autograd.grad()` against a *separate* small graph, never `.backward()`, so the outer training step's gradient accumulation is untouched.
- **fp32 master copies**: maintain fp32 copies of `titans.mlp.parameters()` for the inner Adam step; cast back to bf16 for forward.
- **Gate-ramp source**: the gate is a backbone parameter (not LoRA); it remains trainable in *every* stage from M6 onward, including the M7 LoRA-only fine-tunes. Add `model.titans.gate` to the always-trainable parameter group. ⟂ REVIEW H2
- **Warmup data**: the 5B-token gate-ramp must happen on POST-SFT data (e.g., a held-out slice of SmolTalk), not FineWeb-Edu — the memory needs to learn associations relevant to the agent task, not generic web text.

## Files

- Modify: `model/titans_mag.py` (full implementation)
- Create: `train/titans_warmup.py`
- Modify: `tests/test_titans_mag.py`

## Tasks

- [ ] **Step 1: Surprise-gated test-time update — corrected per review** ⟂ REVIEW C2

```python
# model/titans_mag.py — production version
import torch
import torch.nn as nn
from model.config import ModelConfig

class TitansMAG(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # learned key/value projections (associative memory objective)
        self.WK = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.WV = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model + 16, cfg.titans_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.titans_hidden, cfg.d_model, bias=False),
        )
        self.layer_embed = nn.Embedding(cfg.n_layers, 16)
        self.gate = nn.Parameter(torch.zeros(cfg.d_model))
        # inner-loop hyperparams (modulated at runtime by hormones)
        self.tau_surprise = 0.5
        self.eta_titans = 1e-3
        # fp32 master copies of MLP params for the inner Adam step
        self.register_buffer("_initialized", torch.tensor(False))

    def forward(self, x, layer_idx: int, allow_inner_update: bool = False):
        if not self.cfg.titans_enabled:
            return torch.zeros_like(x)
        B, T, D = x.shape
        le = self.layer_embed(torch.tensor(layer_idx, device=x.device))
        le = le.expand(B, T, -1)
        k = self.WK(x)
        v = self.WV(x)
        # forward through MLP
        pred = self.mlp(torch.cat([k, le], dim=-1))
        # inner-loop surprise gate: NEVER call .backward() here
        if allow_inner_update:
            with torch.enable_grad():
                # build a small detached graph; outer accumulators are untouched
                k_d = k.detach().requires_grad_(False)
                v_d = v.detach().requires_grad_(False)
                pred_d = self.mlp(torch.cat([k_d, le.detach()], dim=-1))
                surprise = ((pred_d - v_d) ** 2).mean()
                if surprise.item() > self.tau_surprise:
                    grads = torch.autograd.grad(
                        surprise, list(self.mlp.parameters()),
                        retain_graph=False, create_graph=False,
                    )
                    with torch.no_grad():
                        for p, g in zip(self.mlp.parameters(), grads):
                            # fp32 master-copy update; cast back to bf16
                            p_fp32 = p.float() - self.eta_titans * g.float()
                            p.copy_(p_fp32.to(p.dtype))
        return torch.sigmoid(self.gate) * pred
```

Key correctness invariants:
- The objective is `‖MLP(WK·x) − WV·x‖²` (associative), not `‖MLP(x) − x‖²` (identity).
- `torch.autograd.grad()` keeps the inner step's graph isolated from any outer optimizer state.
- `TitansMAG` is excluded from FSDP wrapping via `ignored_modules=[model.titans]` in M2's FSDP setup.

- [ ] **Step 2: Gate-ramp schedule**

Linear from 0 → 1 over the first 5B tokens of any fine-tune that wants memory enabled.

- [ ] **Step 3: Tests**

- Cold-start: memory does not affect output (gate at 0).
- Warm gate: memory output is non-zero and gradient flows through gate.
- Test-time update: feeding the same input twice produces lower surprise the second time (memory learned it).

- [ ] **Step 4: Warmup train**

Run a short fine-tune (~1B tokens of FineWeb-Edu) with memory enabled and gate ramping. Verify the BABILong score improves vs the same model with memory disabled.

## Verification

- All Titans tests pass.
- BABILong score with memory > BABILong score without (by 3+ points).

## Exit criteria

- Production Titans module integrated, tested, warmed up.

---

# M7 — Hormones + Hebbian + LoRA

**Wall-clock:** 5–7 days
**Cost:** ~$80
**Depends on:** M4, M6
**Recommended executor model:** opus

## Context Brief

Build Tier 0a (hormone scalars), Tier 0b (Hebbian co-firing matrix), and Tier 3 (runtime LoRA with snapshot ring). These three systems are tightly coupled: hormones modulate LoRA LR and Hebbian update rate; Hebbian biases tool-selection at inference; LoRA snapshot rollback is triggered by hormones reaching the panic threshold.

## Inputs from spec (cold-start required reading) ⟂ REVIEW C5

Symbols referenced in the tasks below, defined here so a fresh agent need not re-read the spec:

- **`last_known_good`**: highest canary-eval score observed in the current snapshot ring; used as the rollback threshold (rollback if new score drops below `last_known_good − 0.05`).
- **`active_features`**: list of integer feature ids fired in the current turn. Feature space (|F| = 850): task_intent (512 k-means cluster ids), tools_used (~64 tool types), retrieval_sources (256 HippoRAG community ids), response_shape (16 categories). Each turn produces ~5–15 active features.
- **`task_intent_id`**: single integer id from the 512 task_intent clusters; computed by k-means over the query's sentence embedding.
- **`tool_embeddings`**: a learned `(n_tools, d_emb)` embedding matrix; initialized randomly and trained jointly with LoRA during M4 tool-calling SFT.
- **`tau_panic`**: hormone threshold above which a rollback is forced. Initialize `frustration > 0.7 AND fatigue > 0.5`.
- **Trainable parameter groups in M7:**
  - LoRA adapters (rank-16 on Q/K/V/O of diff-attn + in/out projections of Mamba2)
  - `model.titans.gate` (backbone parameter; remains trainable across M6/M7/sleep training) ⟂ REVIEW H2
  - `tool_embeddings`
  - Base backbone is otherwise FROZEN.
- **Canary set**: the 100-prompt held-out set curated in **M4 Step 0** (see that milestone); paths in `evals/canary/`.

## Files

- Create: `model/hormones.py`, `model/hebbian.py`, `model/lora_runtime.py`
- Create: `runtime/snapshot.py`, `runtime/hormone_state.py`
- Create: `train/canary_eval.py`, `train/sleep_train.py`
- Create: `tests/test_hormones.py`, `tests/test_hebbian.py`, `tests/test_lora_runtime.py`, `tests/test_canary_rollback.py`

## Tasks

- [ ] **Step 1: HormoneState class with EMA scalars**

```python
@dataclass
class HormoneState:
    joy: float = 0.0
    frustration: float = 0.0
    confidence: float = 0.0
    fatigue: float = 0.0
    boredom: float = 0.0
    alpha: float = 0.1  # EMA rate

    def update(self, reward, retry, entropy, surprise_var, updates_since_snapshot):
        self.joy = (1-self.alpha)*self.joy + self.alpha*max(0, reward)
        self.frustration = (1-self.alpha)*self.frustration + self.alpha*max(0, -reward) + self.alpha*retry
        self.confidence = (1-self.alpha)*self.confidence + self.alpha*max(0, (1-entropy)*reward)
        self.fatigue = updates_since_snapshot / 1000.0
        self.boredom = (1-self.alpha)*self.boredom + self.alpha*(1.0 / (surprise_var + 1e-3))
```

- [ ] **Step 2: Hebbian H matrix — dense storage** ⟂ REVIEW H3

PyTorch's sparse COO tensors do not support in-place indexed updates, so the dense form is both correct and simpler. 850 × 850 × 4 bytes = 2.7 MB; effectively free.

```python
import torch
import itertools

class HebbianH:
    def __init__(self, n_features: int = 850, eta: float = 0.01, cap: float = 5.0):
        # dense float32 - 2.7MB total, trivial cost
        self.H = torch.zeros(n_features, n_features, dtype=torch.float32)
        self.eta = eta
        self.cap = cap  # absolute cap on entries to prevent runaway

    def update(self, active_features: list[int], joy: float, frustration: float):
        if joy > 0.3 and len(active_features) >= 2:
            for i, j in itertools.combinations(active_features, 2):
                delta = self.eta * joy
                self.H[i, j] = torch.clamp(self.H[i, j] + delta, max=self.cap)
                self.H[j, i] = self.H[i, j]  # keep symmetric
        if frustration > 0.3:
            self.H *= (1.0 - 0.005 * frustration)

    def bias_logits(self, task_intent_id: int, tool_embeddings: torch.Tensor, lam: float = 0.1):
        row = self.H[task_intent_id]
        return lam * (row[:tool_embeddings.shape[0]] @ tool_embeddings)

    def save(self, path: str): torch.save(self.H, path)
    def load(self, path: str): self.H = torch.load(path)
```

- [ ] **Step 3: LoRA runtime**

Use `peft` library's LoRA implementation. Rank 16 on Q/K/V/O of attn layers + Mamba2 in/out projections. Frozen base.

- [ ] **Step 4: Snapshot ring buffer**

```python
class SnapshotRing:
    def __init__(self, size=8, path="out/snapshots/"):
        self.size = size; self.path = path
        self.ring = []  # list of (timestamp, lora_state_dict, canary_score)

    def snapshot(self, lora_module, canary_score):
        self.ring.append((time.time(), copy.deepcopy(lora_module.state_dict()), canary_score))
        if len(self.ring) > self.size: self.ring.pop(0)

    def rollback(self, lora_module):
        # roll back to most recent snapshot with canary_score >= last_canary_score - 0.05
        for ts, sd, score in reversed(self.ring):
            if score >= self.last_known_good:
                lora_module.load_state_dict(sd)
                return ts
```

- [ ] **Step 5: Canary eval (100 frozen prompts)**

Curate 100 prompts covering tool format adherence, RAG faithfulness, and basic reasoning. Run after every 100 online updates. Trigger rollback on >5% degradation.

- [ ] **Step 6: Sleep training**

Nightly batch fine-tune of LoRA on the day's buffered interactions. Standard cosine LR. Triggered by `fatigue > 0.5` or scheduled.

- [ ] **Step 7: Integration test**

End-to-end: simulate 200 interactions with bursts of positive and negative reward. Verify hormones track correctly, Hebbian H builds the right associations, LoRA updates without exploding canary scores.

## Verification

- All Tier 0 + Tier 3 tests pass.
- Integration test shows Hebbian H[churn_intent, chart_tool] rises after 5 simulated churn→chart→joy interactions.
- Canary-triggered rollback restores prior LoRA state cleanly.

## Exit criteria

- Full Frankenstein assembled: base + Titans + HippoRAG + hormones + Hebbian + LoRA.

---

# M8 — FrankenBench Evaluation Suite

**Wall-clock:** 10–14 days ⟂ REVIEW C3 (expanded from 5–7d to support the real factorial)
**Cost:** $500–700 (was $200; expanded for the full factorial + 3 seeds)
**Depends on:** M4 (can author earlier)

## Context Brief

Build a unified eval harness wrapping lm-eval-harness, RULER, BFCL v3, TRACE, and the four paper-original custom probes. Run the **full factorial ablation matrix**: 5 memory-tier configurations × 2³ modulator configurations = 40 runs, plus 3 external compute-matched baselines (Samba-700M, Llama-3.2-1B+RAG, Titans-MAG-700M on pure-attention) = **43 runs total, replicated across 3 seeds = ~120 runs** to support the unifying neuromodulation thesis at COLM. ⟂ REVIEW C3

If budget cannot bear the full factorial, the documented fallback is to drop to 1 seed (43 runs, ~$300) and accept reduced statistical claims — this is the workshop-paper target. **Do not** silently shrink the matrix without updating the paper's claims.

## Files

- Create: `evals/frankenbench.py`, `evals/probes/*.py`, `evals/ablation_matrix.yaml`

## Tasks

- [ ] **Step 1: Wire up lm-eval-harness common-sense suite**

`pip install lm-eval`. Wrapper that loads our HF-format checkpoint and runs the 7-task suite.

- [ ] **Step 2: RULER and BABILong**

Use NVIDIA RULER's repo as-is. BABILong from HF dataset `RMT-team/babilong`.

- [ ] **Step 3: BFCL v3**

Use Gorilla LLM's evaluation harness. Format adapter to our `<tool_call>` schema.

- [ ] **Step 4: TRACE**

Sequential task suite from `BeyonderXX/TRACE-Benchmark`. Wrap to use our checkpoint.

- [ ] **Step 5: Custom probes**

1. MAG memory needle: plant fact into Titans state, query k steps later.
2. Hormone gating eval: alternate stable + novel tasks, measure forgetting with hormones on/off.
3. Joint retrieval-writeback: stream docs, query implied facts.
4. LoRA tool retention: BFCL × TRACE composition.

- [ ] **Step 6: Ablation runner**

```yaml
# evals/ablation_matrix.yaml
configurations:
  - name: backbone_only
    memory: { hipporag: false, titans: false, hebbian: false, lora_runtime: false }
  - name: plus_hipporag
    memory: { hipporag: true, titans: false, hebbian: false, lora_runtime: false }
  # ... etc, 13 total
```

- [ ] **Step 7: Run the matrix**

Each ablation is a short fine-tune from the same base checkpoint, ~3 hours on 4×H100. ~$15 each × 13 = ~$200.

## Verification

- All benchmarks produce numbers in expected ranges for a 700M model.
- Custom probes produce ablation-sensitive differences.

## Exit criteria

- `evals/results/ablation_table.csv` complete.

---

# M9 — Serving Stack

**Wall-clock:** 3–5 days
**Cost:** ~$0 (local)
**Depends on:** M7

## Context Brief

Build a FastAPI-based inference server that ties together the trained model, HippoRAG retrieval, Titans memory state, hormone tracker, Hebbian H, and LoRA. Single-user local deployment; no multi-tenant concerns.

## Tasks

- [ ] **Step 1: FastAPI server with `/chat` endpoint**
- [ ] **Step 2: `/ingest` endpoint for adding documents to HippoRAG**
- [ ] **Step 3: `/feedback` endpoint that takes `accept|reject|correction` and updates hormones**
- [ ] **Step 4: Background nightly sleep-training job**
- [ ] **Step 5: Minimal CLI client for end-to-end smoke test**

## Verification

- 100-turn conversation produces measurable Hebbian H growth and at least one LoRA snapshot.

## Exit criteria

- Personal-agent deployment working on local hardware.

---

# M10 — Paper Write-Up

**Wall-clock:** 1–2 weeks
**Cost:** $0
**Depends on:** M8 results

## Tasks

- [ ] **Step 1: Convert design spec into paper introduction + related work**
- [ ] **Step 2: Methods sections directly from milestone documentation**
- [ ] **Step 3: Results section with ablation table from M8**
- [ ] **Step 4: Discussion section addressing the three reviewer attacks (Section 9.3 of spec)**
- [ ] **Step 5: Strongest-model adversarial review of paper draft**
- [ ] **Step 6: Final formatting for COLM 2026 (use COLM LaTeX template)**

## Exit criteria

- Submission-ready PDF in `paper/`.

---

# Anti-pattern catalog

Watch for these during execution. They signal a step is going wrong.

| Anti-pattern | Why it's bad | Correct response |
|---|---|---|
| Trying to resume M1b from M1a weights | Vocab (50257→32016), seq_len (512→4096), and RoPE base (1e4→1e6) all change; checkpoint is unloadable. Will burn a day debugging. ⟂ REVIEW C4 | M1a is *code* validation only. M1b initializes from scratch. Mark M1a checkpoints as `_archive/m1a-*/` and never `--resume` from them. |
| Reimplementing Mamba2 by hand | Two months of debugging instead of two days | Use `mamba-ssm` package |
| Training the canary eval set | Reward hacking the rollback trigger | Canary prompts must never appear in any training stream |
| Letting Hebbian H grow unboundedly | "Rut" behaviors | Hard cap on H entries; frustration-driven decay |
| Updating LoRA on self-generated outputs without filter | Model collapse | Only update on user-confirmed positive outcomes |
| Skipping M1a smoke | Find bugs at $50/hour instead of free | Always run M1a even if it feels redundant |
| Coupling Titans and Hebbian state | Untestable | Keep state objects independent; modulation through hormone scalars only |
| Hand-tuning ablation-matrix runs | Cherry-picking accusation | All runs use identical scripts; differences from YAML config only |

---

# Stop-loss gates ⟂ REVIEW M5

A solo project drifts. These are hard gates that force scope decisions rather than silent overruns. At each gate, if the condition fires, **stop and choose explicitly** between (a) cut scope per the listed fallback or (b) deliberately accept the overrun in writing.

| Gate | Condition | Fallback |
|---|---|---|
| **G1: post-M2 budget** | Remaining < $1100 after M2 | Drop long-context extension (M3 stays at 60B, no 32k stretch); model becomes 4k-context-only |
| **G2: post-M3 budget** | Remaining < $600 after M3 finishes | Drop 1 of the 4 SFT stages; recommend dropping DPO (least load-bearing) |
| **G3: post-M3 quality** | C4 val PPL > 14.0 (vs target < 12.0) | Do NOT scale to long-context yet; investigate base-model issue first, save remaining budget for one re-run |
| **G4: post-M4 timeline** | More than 5 weeks elapsed | Skip M9 (serving stack); deliver evaluation + paper only |
| **G5: post-M7 capability** | Online LoRA destabilizes after fewer than 50 turns despite snapshot rollback | Disable online LoRA for v1; sleep-training-only mode. Document as future work. |
| **G6: post-M8 ablation** | Full 43-run × 3-seed factorial requires > $700 | Drop to 1 seed; reframe paper for workshop venue, not COLM main track |
| **G7: total budget** | Cumulative spend exceeds $1800 | Hard stop; no more cloud runs. Write paper with what exists. |

---

# Plan mutation protocol

If during execution you need to split a step, insert a new step, reorder, or abandon a step:

1. Open this file.
2. Add an entry under the milestone's "Mutation log" section (create the section if missing) noting: timestamp, the mutation, and the reason.
3. Apply the mutation to the task list.
4. Re-run the milestone's verification block from scratch after mutation.

Never silently change the plan. The audit trail matters when you come back to this in week 7.

---

# Cross-milestone progress tracker

Update this table at the end of each milestone.

| Milestone | Status | Started | Completed | Notes |
|---|---|---|---|---|
| M1a | pending | | | |
| M1b | pending | | | |
| M2 | pending | | | |
| M3 | pending | | | |
| M4 | pending | | | |
| M5 | pending | | | |
| M6 | pending | | | |
| M7 | pending | | | |
| M8 | pending | | | |
| M9 | pending | | | |
| M10 | pending | | | |

---

*End of blueprint. Companion spec: `docs/superpowers/specs/2026-05-28-frankenstein-slm-design.md`. Companion paper draft: `docs/frankenstein-slm-paper.html`.*
