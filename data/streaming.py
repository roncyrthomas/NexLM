"""Streaming dataloaders.

Two backends:

- `build_tinystories_loader` — small (~470M tokens), fits in RAM, tokenize once.
  Used by M1a smoke runs.

- `build_fineweb_loader` — large (~1.3T tokens), streamed on the fly with the
  Phi-3 tokenizer. Used by M1b sanity and M3 pretrain.

Both yield (input_ids, target_ids) tuples forever, suitable for the trainer's
infinite-step loop. Same interface, different internals.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast


# --------------------------------------------------------------------------- #
# M1a: TinyStories — in-memory tokenize-once
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# M1b / M3: FineWeb-Edu — streaming, on-the-fly Phi-3 tokenize
# --------------------------------------------------------------------------- #
def _doc_token_stream(
    hf_split,
    tokenizer,
    eos_id: int,
) -> Iterator[int]:
    """Yield one int token id at a time from streamed documents."""
    for example in hf_split:
        text = example.get("text") or example.get("content") or ""
        if not text:
            continue
        for tid in tokenizer.encode(text, add_special_tokens=False):
            yield tid
        yield eos_id


def build_fineweb_loader(
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 4,
    seed: int = 0,
    device: str = "cuda",
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_subset: str | None = "sample-10BT",
):
    """Stream FineWeb-Edu and pack tokens into (seq_len+1)-windows for batching.

    Yields (input_ids[B, seq_len], target_ids[B, seq_len]) tuples.

    Why pack: HF streaming gives one doc at a time, with variable lengths.
    We greedily concatenate token streams across docs (separated by EOS),
    chunk into fixed (seq_len+1) windows, batch them, and yield.
    """
    from model.tokenizer_util import load_tokenizer

    tokenizer = load_tokenizer()
    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("</sys>")

    kwargs = {"streaming": True, "split": split}
    if dataset_subset:
        kwargs["name"] = dataset_subset
    ds = load_dataset(dataset_name, **kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    token_iter = _doc_token_stream(ds, tokenizer, eos_id)

    win = seq_len + 1
    need = batch_size * win
    buf: list[int] = []

    while True:
        while len(buf) < need:
            try:
                buf.append(next(token_iter))
            except StopIteration:
                ds = ds.shuffle(seed=seed + 1, buffer_size=10_000)
                token_iter = _doc_token_stream(ds, tokenizer, eos_id)

        flat = torch.tensor(buf[:need], dtype=torch.long)
        buf = buf[need:]
        flat = flat.view(batch_size, win)
        x = flat[:, :-1].contiguous()
        y = flat[:, 1:].contiguous()
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# --------------------------------------------------------------------------- #
# Dispatch helper: pick loader by name (called by trainer)
# --------------------------------------------------------------------------- #
def build_loader(data_name: str, **kwargs):
    if data_name in ("tinystories", "tiny_stories"):
        return build_tinystories_loader(**kwargs)
    if data_name in ("fineweb-edu", "fineweb_edu", "fineweb"):
        return build_fineweb_loader(**kwargs)
    raise ValueError(f"unknown data backend: {data_name}")
