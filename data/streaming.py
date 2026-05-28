"""Streaming dataloader for TinyStories (M1a) and pretrain mixes (M1b+).

TinyStories is small (~470M tokens) so we just tokenize once at startup and
serve random windows. For M1b's 60B-token mix we'll switch to HuggingFace's
streaming + interleave_datasets API; same generator interface.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast


def build_tinystories_loader(
    split: str = "train",
    seq_len: int = 512,
    batch_size: int = 32,
    seed: int = 0,
    device: str = "cuda",
):
    """Yield (input_ids, target_ids) tuples forever.

    Concatenates all stories with an EOS-like separator and serves random windows.
    """
    ds = load_dataset("roneneldan/TinyStories", split=split)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_id = tok.eos_token_id or tok.convert_tokens_to_ids("<|endoftext|>")

    # tokenize everything once; for TinyStories this fits in RAM (~470M tokens at int32 = ~1.9GB)
    print(f"[streaming] tokenizing {split} split ({len(ds)} stories)...")
    all_ids: list[int] = []
    for story in ds:
        all_ids.extend(tok.encode(story["text"]))
        all_ids.append(eos_id)
    tokens = torch.tensor(all_ids, dtype=torch.long)
    print(f"[streaming] {split}: {len(tokens):,} tokens ready")

    g = torch.Generator().manual_seed(seed)
    n = len(tokens) - seq_len - 1
    while True:
        starts = torch.randint(0, n, (batch_size,), generator=g)
        x = torch.stack([tokens[s : s + seq_len] for s in starts])
        y = torch.stack([tokens[s + 1 : s + 1 + seq_len] for s in starts])
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)
