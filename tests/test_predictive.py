"""Tests for Tier P — Predictive coding."""

import torch

from agent.predictive import PredictiveCoder


def test_no_prior_returns_neutral():
    pc = PredictiveCoder()
    assert pc.observe_first_token(42) == 0.5


def test_high_prediction_low_surprise():
    pc = PredictiveCoder()
    vocab = 100
    logits = torch.full((vocab,), -10.0)
    logits[7] = 10.0  # very confident about token 7
    pc.predict_from_logits(logits)
    # First observation must initialize EMAs, so call twice for steady state
    pc.observe_first_token(7)
    s = pc.observe_first_token(7)
    assert s < 0.6  # low surprise when prediction matches


def test_unexpected_token_high_surprise():
    pc = PredictiveCoder()
    vocab = 100
    logits = torch.full((vocab,), -10.0)
    logits[7] = 10.0
    pc.predict_from_logits(logits)
    # Warm EMAs with expected tokens
    for _ in range(3):
        pc.observe_first_token(7)
    # Now observe an unexpected token
    pc.predict_from_logits(logits)
    s = pc.observe_first_token(42)
    assert s > 0.5


def test_reset_clears_prediction():
    pc = PredictiveCoder()
    logits = torch.zeros(10)
    pc.predict_from_logits(logits)
    pc.reset()
    assert pc.observe_first_token(0) == 0.5
