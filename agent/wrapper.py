"""NexAgent — frozen base model + trainable agent layer.

Scope after P2–P4:
  - Load base model + tokenizer
  - Apply LoRA adapters via PEFT (Tier 3)
  - Optionally instantiate Tier 0a (hormones), 0b (Hebbian),
    Tier 1 (HippoRAG), Tier 2 (Titans MAG over base hidden states)
  - turn() — single-turn pipeline that applies the agent layer:
      retrieve from HippoRAG → render context → bias logits with Hebbian →
      modulate sampling temperature with hormones → generate → update tiers
  - Tools — Granite-style tool_call format via ToolRegistry
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.config import AgentConfig
from agent.hebbian import HebbianH
from agent.hormones import HormoneState
from agent.tools import ToolRegistry, parse_tool_calls

try:
    from agent.hipporag import HippoRAG, regex_stub_extractor

    HAVE_HIPPORAG = True
except ImportError:
    HAVE_HIPPORAG = False

from agent.titans import TitansMAG

# Frank v2 tiers (each optional; instantiated only when its flag is on)
from agent.predictive import PredictiveCoder, predict_next_user_input
from agent.episodic import EpisodicMemory
from agent.habits import HabitsCache
from agent.dreamer import Dreamer
from agent.metaplastic import MetaPlastic


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class NexAgent(nn.Module):
    """Frozen base + trainable agent layer."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = self._load_tokenizer()
        self.base = self._load_base()
        self.base = self._freeze_base(self.base)
        self.base = self._attach_lora(self.base)

        # ─── Tier instantiation (each opt-in via cfg flag) ─────────────
        self.hormones: HormoneState | None = HormoneState() if cfg.enable_hormones else None
        self.hebbian: HebbianH | None = HebbianH() if cfg.enable_hebbian else None
        self.hipporag = None
        if cfg.enable_hipporag and HAVE_HIPPORAG:
            self.hipporag = HippoRAG()
        self.titans: TitansMAG | None = None
        self._titans_hooks: list = []  # one hook per attached layer
        if cfg.enable_titans:
            d_model = self._infer_d_model()
            self.titans = TitansMAG(
                d_model=d_model,
                d_hidden=cfg.titans_d_hidden,
                eta=cfg.titans_eta,
                tau_surprise=cfg.titans_tau_surprise,
            )
            # Attach to every configured layer index. Ablation studies vary this:
            # [-1]      = top of stack only
            # [-1, -8]  = top + mid-depth
            # [0, 8, 16, 24] = early/mid/late/top
            for li in cfg.titans_layer_indices:
                self._titans_hooks.append(self.titans.attach_to_base(self.base, layer_index=li))

        # ─── Frank v2 tiers ─────────────────────────────────────────────
        self.predictive: PredictiveCoder | None = (
            PredictiveCoder() if cfg.enable_predictive else None
        )
        self.episodic: EpisodicMemory | None = (
            EpisodicMemory(
                max_size=cfg.episodic_buffer_size,
                similarity_threshold=cfg.episodic_similarity_threshold,
            )
            if cfg.enable_episodic
            else None
        )
        self.habits: HabitsCache | None = (
            HabitsCache(
                compile_threshold=cfg.habits_compile_threshold,
                reward_threshold=cfg.habits_reward_threshold,
            )
            if cfg.enable_habits
            else None
        )
        self.dreamer: Dreamer | None = (
            Dreamer(n_samples_per_dream=cfg.dream_n_samples)
            if cfg.enable_dreamer
            else None
        )
        self.metaplastic: MetaPlastic | None = (
            MetaPlastic(
                alpha_up=cfg.metaplastic_alpha_up,
                alpha_down=cfg.metaplastic_alpha_down,
            )
            if cfg.enable_metaplastic
            else None
        )

        # ─── Tools registry — always present (cheap; may be unused) ────
        self.tools = ToolRegistry()

    # ────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ────────────────────────────────────────────────────────────────────
    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.cfg.base_model_name,
            trust_remote_code=self.cfg.trust_remote_code,
        )

    def _load_base(self):
        dtype = _DTYPE_MAP[self.cfg.base_dtype]
        kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.cfg.trust_remote_code,
        }
        if self.cfg.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif self.cfg.quantization == "8bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(self.cfg.base_model_name, **kwargs)

    @staticmethod
    def _freeze_base(model: nn.Module) -> nn.Module:
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _attach_lora(self, model: nn.Module) -> nn.Module:
        lora_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_cfg)

    def _infer_d_model(self) -> int:
        cfg = self.base.config if hasattr(self.base, "config") else None
        if cfg is None:
            return 2048  # safe default
        for key in ("hidden_size", "d_model", "n_embd"):
            if hasattr(cfg, key):
                return getattr(cfg, key)
        return 2048

    # ────────────────────────────────────────────────────────────────────
    # Public surface
    # ────────────────────────────────────────────────────────────────────
    def count_params(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(self, input_ids: torch.LongTensor, **kwargs):
        return self.base(input_ids=input_ids, **kwargs)

    # ────────────────────────────────────────────────────────────────────
    # Single-turn agent pipeline
    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, prompt: str, **gen_kwargs) -> str:
        """Plain inference — bypasses the agent layer. Used by tests + tools."""
        device = next(self.parameters()).device
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        defaults = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        defaults.update(gen_kwargs)
        out = self.base.generate(ids, **defaults)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def turn(self, user_query: str, retrieve: bool = True, intent_id: int | None = None, **gen_kwargs) -> dict:
        """One full agent turn — pipeline order matches the v2 spec.

        Order:
          1. Habit check    (v2): if compiled triple matches, BYPASS generation.
          2. Episodic recall (v2): KNN past similar situations as few-shot hint.
          3. HippoRAG retrieve (v1): KG retrieval.
          4. Tool descriptions (v1): catalogue rendering.
          5. Hormone-modulated temperature (v1).
          6. Predictive forecast (v2): cache next-user prediction for surprise scoring.
          7. Generate.
          8. Parse + return.
        """
        bypassed = False

        # 1. Habit bypass (v2)
        if self.habits is not None and intent_id is not None:
            frust = self.hormones.frustration if self.hormones is not None else 0.0
            habit = self.habits.maybe_bypass(intent_id, frustration=frust)
            if habit is not None and habit.cached_response:
                # Habit fires — return cached response, skip generation entirely
                return {
                    "prompt": user_query,
                    "response": habit.cached_response,
                    "tool_calls": parse_tool_calls(habit.cached_response),
                    "retrieved_chunks": [],
                    "episodic_hits": [],
                    "habit_fired": True,
                    "hormone_state": self.hormones.to_dict() if self.hormones else None,
                }

        # 2. Episodic recall (v2)
        episodic_hits = []
        episodic_block = ""
        if self.episodic is not None and self.episodic.is_familiar(user_query):
            episodic_hits = self.episodic.recall(user_query, k=3)
            positive_hits = [e for e in episodic_hits if e.reward > 0]
            if positive_hits:
                episodic_block = (
                    "Similar past examples that worked:\n"
                    + "\n".join(
                        f"  Q: {e.query}\n  A: {e.response[:200]}"
                        for e in positive_hits[:2]
                    )
                    + "\n\n"
                )

        # 3. HippoRAG retrieve (v1)
        retrieved = []
        ctx_block = ""
        if retrieve and self.hipporag is not None:
            try:
                chunks = self.hipporag.retrieve(user_query, k=3)
                retrieved = chunks
                if chunks:
                    ctx_block = (
                        "<retrieve>\n"
                        + "\n".join(f"<chunk>{c.text}</chunk>" for c in chunks)
                        + "\n</retrieve>\n"
                    )
            except Exception:
                pass

        # 4. Tool descriptions
        tools_block = self.tools.render_descriptions() + "\n" if self.tools.tools else ""

        # 5. Hormone-modulated temperature
        gen_kwargs = dict(gen_kwargs)
        if self.hormones is not None:
            gen_kwargs.setdefault(
                "temperature", self.hormones.sampling_temperature(self.cfg.temperature)
            )

        # 6. Predictive forecast — cache for surprise scoring at next turn
        if self.predictive is not None:
            try:
                conversation = [{"role": "user", "content": user_query}]
                logits = predict_next_user_input(self.base, self.tokenizer, conversation)
                self.predictive.predict_from_logits(logits)
            except Exception:
                pass

        # 7. Generate
        prompt = tools_block + ctx_block + episodic_block + f"User: {user_query}\nAssistant: "
        response = self.generate(prompt, **gen_kwargs)

        # 8. Parse + return
        tool_calls = parse_tool_calls(response)
        return {
            "prompt": prompt,
            "response": response,
            "tool_calls": tool_calls,
            "retrieved_chunks": retrieved,
            "episodic_hits": episodic_hits,
            "habit_fired": False,
            "hormone_state": self.hormones.to_dict() if self.hormones else None,
        }

    def observe_feedback(
        self,
        reward: float = 0.0,
        retry_signal: float = 0.0,
        active_features: list[int] | None = None,
        last_user_query: str | None = None,
        last_response: str | None = None,
        next_user_message: str | None = None,
        intent_id: int | None = None,
        tool_id: int | None = None,
        shape_id: int | None = None,
    ) -> dict:
        """Feed an outcome back into the agent layer. Returns diagnostic dict.

        v1 effects:
          - Hormones update from (reward, retry_signal).
          - Hebbian H strengthens active feature pairs on joy > τ.

        v2 effects:
          - Predictive surprise (from next_user_message) feeds hormones.
          - Episodic stores (last_user_query, last_response, reward).
          - Habits observe (intent_id, tool_id, shape_id, reward).
          - Metaplastic credits eta[i,j] for recently-updated pairs.
          - Dreamer may fire if fatigue threshold reached.
        """
        diag = {}

        # v2: predictive surprise from next user message
        predictive_surprise = 0.5
        if self.predictive is not None and next_user_message:
            predictive_surprise = self.predictive.observe_text(
                next_user_message, self.tokenizer
            )
            diag["predictive_surprise"] = predictive_surprise

        # v1: hormone update (now also takes predictive surprise as variance proxy)
        if self.hormones is not None:
            self.hormones.update(
                reward=reward,
                retry_signal=retry_signal,
                surprise_variance=1.0 / (predictive_surprise + 1e-3),
            )

        # v1: Hebbian
        if self.hebbian is not None and active_features:
            joy = self.hormones.joy if self.hormones else max(0.0, reward)
            frust = self.hormones.frustration if self.hormones else max(0.0, -reward)
            n_changed = self.hebbian.update(active_features, joy=joy, frustration=frust)
            diag["hebbian_pairs_changed"] = n_changed
            # v2: metaplastic credit on the same active features
            if self.metaplastic is not None:
                self.metaplastic.record_update(active_features)
                self.metaplastic.credit_outcome(reward)

        # v2: episodic store
        if self.episodic is not None and last_user_query and last_response:
            self.episodic.remember(last_user_query, last_response, reward=reward)
            diag["episodic_size"] = len(self.episodic.episodes)

        # v2: habits observe
        if (
            self.habits is not None
            and intent_id is not None
            and tool_id is not None
            and shape_id is not None
        ):
            self.habits.observe(
                intent_id, tool_id, shape_id, reward=reward, cached_response=last_response
            )
            self.habits.tick()
            diag["habits"] = self.habits.stats()

        # v2: dream consolidation
        if (
            self.dreamer is not None
            and self.hormones is not None
            and self.dreamer.should_dream(self.hormones)
        ):
            stats = self.dreamer.dream(self)
            diag["dream"] = stats

        return diag

    # ────────────────────────────────────────────────────────────────────
    # Persistence (LoRA + Hebbian + hormones)
    # ────────────────────────────────────────────────────────────────────
    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # LoRA via PEFT
        self.base.save_pretrained(str(path))
        # Hebbian H
        if self.hebbian is not None:
            self.hebbian.save(path / "hebbian.pt")
        # Hormone scalars
        if self.hormones is not None:
            import json
            (path / "hormones.json").write_text(json.dumps(self.hormones.to_dict()))
        # Titans MLP weights
        if self.titans is not None:
            torch.save(self.titans.state_dict(), path / "titans.pt")
        # HippoRAG graph
        if self.hipporag is not None and len(self.hipporag.graph) > 0:
            self.hipporag.save(path / "hipporag.pkl")

    def load_state(self, path: str | Path) -> None:
        path = Path(path)
        # Reload LoRA adapter on the un-LoRA'd base
        self.base = PeftModel.from_pretrained(self.base.get_base_model(), str(path))
        if self.hebbian is not None and (path / "hebbian.pt").exists():
            self.hebbian = HebbianH.load(path / "hebbian.pt")
        if self.hormones is not None and (path / "hormones.json").exists():
            import json
            self.hormones = HormoneState.from_dict(json.loads((path / "hormones.json").read_text()))
        if self.titans is not None and (path / "titans.pt").exists():
            self.titans.load_state_dict(torch.load(path / "titans.pt", map_location="cpu", weights_only=True))
        if self.hipporag is not None and (path / "hipporag.pkl").exists():
            self.hipporag = HippoRAG.load(path / "hipporag.pkl")

    # Back-compat shim used by older test code
    save_lora = save_state

    @classmethod
    def from_pretrained_lora(cls, cfg: AgentConfig, lora_path: str | Path) -> "NexAgent":
        agent = cls(cfg)
        agent.load_state(lora_path)
        return agent
