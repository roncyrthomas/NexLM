"""Tokenizer for the production 700M model: Phi-3 BPE + 16 added special tokens.

The 16 specials reserve token IDs for the structured operations the Frankenstein
performs at inference time: tool calling, retrieval (HippoRAG), memory ops,
document/chunk delimiters, system framing.

After adding the specials, vocab grows from 32,000 → 32,016. The backbone is
parameterized on `cfg.vocab_size`, so it automatically resizes the embed +
lm_head when production_700m is loaded.
"""

from __future__ import annotations

from transformers import AutoTokenizer

# 16 specials, ordered. Their IDs are deterministic — the order MUST stay
# stable across the whole project lifecycle, otherwise checkpoints break.
SPECIAL_TOKENS = [
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    "<retrieve>", "</retrieve>",
    "<mem_read>", "</mem_read>",
    "<mem_write>", "</mem_write>",
    "<doc>", "</doc>",
    "<chunk>", "</chunk>",
    "<sys>", "</sys>",
]


def load_tokenizer(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Load Phi-3 tokenizer and add the 16 specials in canonical order.

    Returns the tokenizer with `len(tok) == 32016` (32k Phi-3 + 16 specials).
    """
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    n_added = tok.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    # Validation: every special must be in the vocab now
    for s in SPECIAL_TOKENS:
        assert tok.convert_tokens_to_ids(s) is not None, f"missing special: {s}"
    return tok


def special_token_ids(tok) -> dict[str, int]:
    """Map each special token string to its integer id (for code that needs them)."""
    return {s: tok.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS}
