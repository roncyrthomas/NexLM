"""NexAgent — wraps a frozen HuggingFace base model with our trainable agent layer.

Current scope (P1):
  - Load base model + tokenizer
  - Apply LoRA adapters via PEFT
  - generate() for sanity testing
  - Save / load LoRA-only checkpoints

Hooks for future tiers (P2–P4) are stubbed; the wrapper grows incrementally
without breaking the load/generate interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.config import AgentConfig


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class NexAgent(nn.Module):
    """The Frankenstein agent — frozen base + trainable agent layer."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = self._load_tokenizer()
        self.base = self._load_base()
        self.base = self._freeze_base(self.base)
        self.base = self._attach_lora(self.base)

        # Tier 0a (hormones), 0b (Hebbian), 1 (HippoRAG), 2 (Titans) — added in P2+
        self.hormones = None
        self.hebbian = None
        self.hipporag = None
        self.titans = None

    # ────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ────────────────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────────────
    # Public surface
    # ────────────────────────────────────────────────────────────────────────
    def count_params(self) -> tuple[int, int]:
        """Return (total_params, trainable_params). Trainable should be LoRA-only."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(self, input_ids: torch.LongTensor, **kwargs):
        return self.base(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def generate(self, prompt: str, **gen_kwargs) -> str:
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

    # ────────────────────────────────────────────────────────────────────────
    # Persistence (LoRA only — base weights stay on HF)
    # ────────────────────────────────────────────────────────────────────────
    def save_lora(self, path: str | Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(str(path))

    @classmethod
    def from_pretrained_lora(cls, cfg: AgentConfig, lora_path: str | Path) -> "NexAgent":
        """Rebuild agent and load a saved LoRA adapter."""
        agent = cls(cfg)
        # peft loads the adapter weights in-place on the base
        agent.base = PeftModel.from_pretrained(agent.base.get_base_model(), str(lora_path))
        return agent
