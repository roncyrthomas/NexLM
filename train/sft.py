"""LoRA SFT trainer for NexAgent.

Train the agent's LoRA adapters on instruction-response pairs.
Standard cross-entropy on the response tokens; instruction tokens are masked.
Optionally hormone-modulates the learning rate when AgentConfig.enable_hormones
and AgentConfig.train_lora_online are both true.

Datasets supported (via load_dataset):
  - Glaive-FC-v2 (tool calling) — column `chat`
  - HuggingFaceH4/ultrachat_200k (general SFT) — `messages`
  - any custom JSONL with {"messages": [...]} OpenAI-format
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset

from agent.config import AgentConfig
from agent.wrapper import NexAgent


@dataclass
class SFTConfig:
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    dataset_split: str = "train_sft"
    text_field: str = "messages"  # or "chat" for Glaive

    max_seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_steps: int = 50
    max_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    log_every: int = 25
    save_every: int = 500
    output_dir: str = "out/sft"
    wandb_project: str = "nexlm-sft"


def format_messages_for_sft(messages: list[dict]) -> tuple[str, str]:
    """Render OpenAI-style messages into (prompt, response) for next-token training."""
    parts = []
    response = ""
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = m.get("content", "")
        if i == len(messages) - 1 and role == "assistant":
            response = content
        else:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, role.capitalize())
            parts.append(f"{tag}: {content}")
    prompt = "\n".join(parts) + "\nAssistant: "
    return prompt, response


class SFTDataset(torch.utils.data.IterableDataset):
    def __init__(self, sft_cfg: SFTConfig, tokenizer, device: str = "cuda"):
        self.cfg = sft_cfg
        self.tokenizer = tokenizer
        self.device = device

    def __iter__(self):
        ds = load_dataset(self.cfg.dataset_name, split=self.cfg.dataset_split, streaming=True)
        ds = ds.shuffle(buffer_size=1000, seed=0)
        for row in ds:
            messages = row.get(self.cfg.text_field) or row.get("messages") or row.get("chat")
            if not messages:
                continue
            prompt, response = format_messages_for_sft(messages)
            if not response:
                continue

            full = prompt + response
            ids = self.tokenizer(full, return_tensors="pt", truncation=True, max_length=self.cfg.max_seq_len).input_ids[0]
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_seq_len).input_ids[0]

            # Build labels: -100 on the prompt part, real ids on the response part
            labels = ids.clone()
            labels[: len(prompt_ids)] = -100
            yield {"input_ids": ids, "labels": labels}


def run_sft(agent: NexAgent, sft_cfg: SFTConfig, use_wandb: bool = False) -> None:
    """Run a LoRA SFT training loop on the given agent."""
    device = next(agent.parameters()).device
    agent.train()

    # Only LoRA params get gradients (base is already frozen)
    trainable = [p for p in agent.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=sft_cfg.lr, weight_decay=sft_cfg.weight_decay)

    if use_wandb:
        import wandb
        wandb.init(project=sft_cfg.wandb_project, config={
            "agent_base": agent.cfg.base_model_name,
            "lora_r": agent.cfg.lora_r,
            **sft_cfg.__dict__,
        })

    ds = SFTDataset(sft_cfg, agent.tokenizer, device=device)
    loader = iter(ds)

    out_dir = Path(sft_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    running_loss = 0.0
    while step < sft_cfg.max_steps:
        # warmup + cosine LR
        if step < sft_cfg.warmup_steps:
            lr = sft_cfg.lr * (step + 1) / sft_cfg.warmup_steps
        else:
            import math
            prog = (step - sft_cfg.warmup_steps) / max(1, sft_cfg.max_steps - sft_cfg.warmup_steps)
            lr = 0.5 * sft_cfg.lr * (1 + math.cos(math.pi * min(1.0, prog)))
        # Hormone modulation if enabled
        if agent.hormones is not None and agent.cfg.train_lora_online:
            lr = agent.hormones.lora_lr_multiplier(lr)
        for g in opt.param_groups:
            g["lr"] = lr

        loss_accum = 0.0
        for _ in range(sft_cfg.grad_accum):
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(ds)
                batch = next(loader)
            ids = batch["input_ids"].unsqueeze(0).to(device)
            labels = batch["labels"].unsqueeze(0).to(device)
            out = agent.base(input_ids=ids, labels=labels)
            (out.loss / sft_cfg.grad_accum).backward()
            loss_accum += out.loss.item()
        loss_accum /= sft_cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable, sft_cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        running_loss += loss_accum
        step += 1

        if step % sft_cfg.log_every == 0:
            avg = running_loss / sft_cfg.log_every
            running_loss = 0.0
            print(f"step {step}/{sft_cfg.max_steps} | loss {avg:.4f} | lr {lr:.2e}")
            if use_wandb:
                wandb.log({"train/loss": avg, "train/lr": lr}, step=step)

        if step % sft_cfg.save_every == 0:
            ckpt = out_dir / f"step_{step}"
            agent.save_state(ckpt)
            print(f"[save] {ckpt}")

    agent.save_state(out_dir / "final")
    print(f"[done] saved final to {out_dir / 'final'}")


if __name__ == "__main__":
    cfg = AgentConfig.smollm2_local()
    sft_cfg = SFTConfig()
    agent = NexAgent(cfg).cuda()
    run_sft(agent, sft_cfg, use_wandb=False)
