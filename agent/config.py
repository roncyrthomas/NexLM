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

    # ─── Frank v2 — self-derived learning tier toggles ────────────────────
    enable_predictive: bool = False    # Tier P (V2.1)
    enable_episodic: bool = False      # Tier E (V2.2)
    enable_habits: bool = False        # Tier H (V2.3)
    enable_dreamer: bool = False       # Tier D (V2.4)
    enable_metaplastic: bool = False   # Tier M (V2.5)

    # Frank v2 hyperparams
    episodic_buffer_size: int = 10_000
    episodic_similarity_threshold: float = 0.85
    habits_compile_threshold: int = 10
    habits_reward_threshold: float = 0.7
    dream_n_samples: int = 64
    metaplastic_alpha_up: float = 0.05
    metaplastic_alpha_down: float = 0.03

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

    # ─── Three-way comparison presets (the paper's actual A/B/C) ─────────
    @classmethod
    def vanilla(cls, base: str | None = None) -> "AgentConfig":
        """Baseline — LoRA only, no agent layer tiers."""
        cfg = cls() if base is None else cls(base_model_name=base)
        return cfg

    @classmethod
    def frank_v1(cls, base: str | None = None) -> "AgentConfig":
        """Frank v1 — external-reward-driven agent layer."""
        cfg = cls() if base is None else cls(base_model_name=base)
        cfg.enable_hormones = True
        cfg.enable_hebbian = True
        cfg.enable_hipporag = True
        cfg.enable_titans = True
        cfg.train_lora_online = True
        return cfg

    @classmethod
    def frank_v2(cls, base: str | None = None) -> "AgentConfig":
        """Frank v2 — adds self-derived learning signals."""
        cfg = cls.frank_v1(base=base)
        cfg.enable_predictive = True
        cfg.enable_episodic = True
        cfg.enable_habits = True
        cfg.enable_dreamer = True
        cfg.enable_metaplastic = True
        return cfg
