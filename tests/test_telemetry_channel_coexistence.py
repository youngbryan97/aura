"""Live appraisal and identity telemetry must share one dictionary."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("interiority_first", [False, True])
def test_interiority_and_phenomena_can_initialize_in_either_order(interiority_first):
    interior = "import core.interiority.telemetry"
    phenomena = "from core.fsw import phenomena_channels; phenomena_channels.declare()"
    steps = [interior, phenomena] if interiority_first else [phenomena, interior]
    script = "\n".join(steps) + """
from core.fsw.telemetry_dictionary import write, channel_value
write('identity.coherence', 0.75)
write('interiority.faculties_fired', 3)
assert channel_value('identity.coherence').value == 0.75
assert channel_value('interiority.faculties_fired').value == 3
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
