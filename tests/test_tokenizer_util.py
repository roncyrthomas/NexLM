"""Tests for the production tokenizer."""

import pytest

from model.tokenizer_util import SPECIAL_TOKENS, load_tokenizer, special_token_ids


def test_special_tokens_count():
    assert len(SPECIAL_TOKENS) == 16


def test_specials_have_pairs():
    """Every <tag> has a matching </tag>."""
    pairs = {s.replace("<", "").replace(">", "").replace("/", "") for s in SPECIAL_TOKENS}
    assert len(pairs) == 8  # 16 tokens / 2 per pair


@pytest.mark.slow
def test_load_tokenizer_phi3():
    """Loads the actual Phi-3 tokenizer and confirms vocab=32016.

    Marked slow because it network-downloads ~10 MB on first run.
    """
    tok = load_tokenizer()
    assert len(tok) == 32016, f"expected vocab 32016, got {len(tok)}"
    ids = special_token_ids(tok)
    # All 16 specials should map to distinct ids in the upper vocab range
    assert len(set(ids.values())) == 16
    for tok_str, tok_id in ids.items():
        assert tok_id >= 32000, f"{tok_str} id {tok_id} should be >= 32000"


@pytest.mark.slow
def test_specials_roundtrip():
    """Encoding then decoding a special token returns it intact."""
    tok = load_tokenizer()
    for s in SPECIAL_TOKENS:
        ids = tok.encode(s, add_special_tokens=False)
        decoded = tok.decode(ids)
        assert s in decoded, f"{s} did not round-trip: got {decoded!r}"
