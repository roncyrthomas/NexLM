"""Tests for Tier 0a hormone scalars."""

from agent.hormones import HormoneState


def test_default_state():
    h = HormoneState()
    assert h.joy == 0.0
    assert h.frustration == 0.0
    assert h.confidence == 0.0
    assert h.fatigue == 0.0


def test_joy_rises_on_positive_reward():
    h = HormoneState(alpha=0.5)
    h.update(reward=1.0)
    assert h.joy > 0.0
    assert h.frustration == 0.0


def test_frustration_rises_on_negative_reward():
    h = HormoneState(alpha=0.5)
    h.update(reward=-1.0)
    assert h.frustration > 0.0
    assert h.joy == 0.0


def test_clamped_to_unit_interval():
    h = HormoneState(alpha=1.0)
    for _ in range(10):
        h.update(reward=10.0, retry_signal=10.0)
    for k in ("joy", "frustration", "confidence", "fatigue", "boredom"):
        assert 0.0 <= getattr(h, k) <= 1.0


def test_lora_lr_multiplier_joy_increases():
    h = HormoneState()
    h.joy = 0.5
    assert h.lora_lr_multiplier(1e-4) > 1e-4


def test_lora_lr_multiplier_frustration_decreases():
    h = HormoneState()
    h.frustration = 0.5
    assert h.lora_lr_multiplier(1e-4) < 1e-4


def test_panic_rollback_threshold():
    h = HormoneState()
    h.frustration = 0.8
    h.fatigue = 0.6
    assert h.should_panic_rollback() is True
    h.frustration = 0.5
    assert h.should_panic_rollback() is False


def test_snapshot_resets_fatigue():
    h = HormoneState()
    h.fatigue = 0.9
    h.updates_since_snapshot = 1000
    h.snapshot_taken()
    assert h.fatigue == 0.0
    assert h.updates_since_snapshot == 0


def test_roundtrip_dict():
    h = HormoneState()
    h.joy = 0.42
    h.frustration = 0.13
    d = h.to_dict()
    h2 = HormoneState.from_dict(d)
    assert abs(h2.joy - 0.42) < 1e-6
    assert abs(h2.frustration - 0.13) < 1e-6


def test_sampling_temperature_clamped():
    h = HormoneState()
    h.frustration = 0.9
    T_high = h.sampling_temperature(base_T=0.7)
    h.frustration = 0.0
    h.confidence = 0.9
    T_low = h.sampling_temperature(base_T=0.7)
    assert T_high > T_low
    assert T_high <= 2.0
    assert T_low >= 0.1
