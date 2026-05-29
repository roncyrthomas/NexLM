"""AgentConfig — controls base model choice, LoRA, and (later) tier toggles.

Defaults are tuned for local development on a single 8GB consumer GPU:
- Base: SmolLM2-1.7B-Instruct (3.4GB bf16, fits comfortably with LoRA training)
- LoRA: rank 16 on Q/K/V/O of all attention layers

For cloud experiments (P5+), swap base to Phi-3-mini and crank LoRA rank up.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    # ─── Base model ────────────────────────────────────────────────────────
    base_model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    base_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    quantization: str | None = None  # None | "8bit" | "4bit" (cloud only)
    trust_remote_code: bool = False

    # ─── LoRA (Tier 3) ─────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Target module names — these are SmolLM2/Llama-style. Phi-3 uses different
    # names ("qkv_proj" instead of separate q/k/v); set per-model.
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ─── Agent layer tier toggles (off until each tier is implemented) ────
    enable_hormones: bool = False     # Tier 0a (P2)
    enable_hebbian: bool = False      # Tier 0b (P2)
    enable_hipporag: bool = False     # Tier 1 (P3)
    enable_titans: bool = False       # Tier 2 (P4)
    train_lora_online: bool = False   # Tier 3 runtime updates (P2 paired with hormones)

    # ─── Tier 2 (Titans MAG) — per-layer placement ────────────────────────
    # Which transformer block(s) the Titans hook attaches to. Negative indices
    # count from the back (-1 = last). A list allows multi-layer ablations.
    titans_layer_indices: list[int] = field(default_factory=lambda: [-1])
    titans_d_hidden: int = 2048
    titans_eta: float = 1e-3
    titans_tau_surprise: float = 0.5

    # ─── Inference defaults ────────────────────────────────────────────────
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    # ─── Convenience presets ───────────────────────────────────────────────
    @classmethod
    def smollm2_local(cls) -> "AgentConfig":
        """Default local-dev preset. Runs on an 8GB GPU."""
        return cls()

    @classmethod
    def phi3_cloud(cls) -> "AgentConfig":
        """Cloud preset for the paper's Phi-3-mini experiments."""
        return cls(
            base_model_name="microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
            # Phi-3 packs Q/K/V into one fused projection
            lora_target_modules=["qkv_proj", "o_proj"],
        )

    @classmethod
    def tinyllama_smoke(cls) -> "AgentConfig":
        """Smallest preset for fast CI / unit tests."""
        return cls(
            base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        )
