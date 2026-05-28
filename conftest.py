"""Pytest config: register custom markers."""

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: tests that download from HF Hub or take >10s; opt-in only"
    )
