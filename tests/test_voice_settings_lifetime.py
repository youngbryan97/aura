"""A settings poll cannot close a conversation the person opened."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_focused_voice_lifetime_and_microphone_disable():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable; voice settings lifetime was not measured")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / "tests/js/voice_settings_lifetime.mjs"), str(root)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
