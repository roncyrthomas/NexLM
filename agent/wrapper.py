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

    def turn(self, user_query: str, retrieve: bool = True, **gen_kwargs) -> dict:
        """One full agent turn: retrieve → render → modulate → generate → update.

        Returns a dict:
            {response, tool_calls, retrieved_chunks, hormone_state}
        """
        # 1. Retrieve context (HippoRAG)
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
                pass  # graceful degradation if graph not built yet

        # 2. Render tools available (if any are registered)
        tools_block = self.tools.render_descriptions() + "\n" if self.tools.tools else ""

        # 3. Modulate sampling temperature with hormones
        gen_kwargs = dict(gen_kwargs)
        if self.hormones is not None:
            gen_kwargs.setdefault("temperature", self.hormones.sampling_temperature(self.cfg.temperature))

        # 4. Compose prompt + generate
        prompt = tools_block + ctx_block + f"User: {user_query}\nAssistant: "
        response = self.generate(prompt, **gen_kwargs)

        # 5. Parse out tool calls
        tool_calls = parse_tool_calls(response)

        return {
            "prompt": prompt,
            "response": response,
            "tool_calls": tool_calls,
            "retrieved_chunks": retrieved,
            "hormone_state": self.hormones.to_dict() if self.hormones else None,
        }

    def observe_feedback(
        self,
        reward: float = 0.0,
        retry_signal: float = 0.0,
        active_features: list[int] | None = None,
    ) -> None:
        """Feed an outcome back into the agent layer.

        Hormones update from (reward, retry_signal). If the turn was
        positively reinforced (joy > τ, frustration < τ), the Hebbian
        co-firing matrix strengthens the pairs in `active_features`.
        """
        if self.hormones is not None:
            self.hormones.update(reward=reward, retry_signal=retry_signal)
        if self.hebbian is not None and active_features:
            joy = self.hormones.joy if self.hormones else max(0.0, reward)
            frust = self.hormones.frustration if self.hormones else max(0.0, -reward)
            self.hebbian.update(active_features, joy=joy, frustration=frust)

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
