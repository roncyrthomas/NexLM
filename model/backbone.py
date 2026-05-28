"""Frankenstein backbone: hybrid Mamba2 + Differential-Attention stack.

Layer interleave is driven by `cfg.attn_layer_positions` (1-indexed):
positions in that list become DiffAttn blocks; all other positions are Mamba2.

For smoke_30m (6 layers, attn at [3, 6]): M M D M M D
For production_700m (24 layers, attn at [4,8,12,16,20,24]): Samba 3:1.

Memory wiring: the Titans MAG module runs in PARALLEL with each backbone
block (Memory-as-Gating). For M1a it returns zeros so this is a no-op.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.diff_attn_block import DiffAttnBlock
from model.mamba2_block import Mamba2Block
from model.titans_mag import TitansMAG


class Frankenstein(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        attn_positions = set(cfg.attn_layer_positions)  # 1-indexed
        self.layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            pos = i + 1  # 1-indexed
            if pos in attn_positions:
                self.layers.append(DiffAttnBlock(cfg, layer_idx=i))
            else:
                self.layers.append(Mamba2Block(cfg, layer_idx=i))

        self.titans = TitansMAG(cfg)
        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # tie embed and lm_head weights — standard for small LMs, saves vocab*d_model params
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        # standard nanoGPT-style init for new linear layers; HF Mamba2Mixer self-inits
        for name, p in self.named_parameters():
            if "lm_head" in name or "embed" in name:
                if p.dim() == 2:
                    nn.init.normal_(p, mean=0.0, std=0.02)
            elif "lambda_" in name or "gate" in name:
                # leave at module-defined init
                pass
            elif p.dim() == 2 and "ffn" in name and "w3" in name:
                # scaled init for residual output projections
                nn.init.normal_(
                    p, mean=0.0, std=0.02 / (2 * self.cfg.n_layers) ** 0.5
                )

    def forward(
        self,
        ids: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
    ):
        x = self.embed(ids)
        for i, layer in enumerate(self.layers):
            # backbone block
            block_out = layer(x)
            # parallel Titans memory branch (zero in M1a)
            mem_out = self.titans(x, layer_idx=i)
            x = block_out + mem_out
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.LongTensor:
        self.eval()
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.cfg.max_seq_len :]
            logits = self.forward(ids_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
