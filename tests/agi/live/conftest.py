import pytest
from pathlib import Path
from tests.agi.live.live_harness import LiveAuraHarness, PROJECT_ROOT

@pytest.fixture
def live_harness():
    harness = LiveAuraHarness(PROJECT_ROOT)
    yield harness
    harness.cleanup()
