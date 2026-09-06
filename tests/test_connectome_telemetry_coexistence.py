"""Connectome observations must coexist with the felt-state channels."""

import subprocess
import sys

from core.connectome import integration
from core.fsw import telemetry_dictionary


def test_connectome_and_interiority_register_in_one_process():
    result = subprocess.run(
        [sys.executable, "-c", """
import core.interiority.telemetry
from core.connectome.integration import declare_telemetry
assert len(declare_telemetry()) == 6
assert declare_telemetry() == []
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_registration_remains_retryable(monkeypatch):
    monkeypatch.setattr(integration, "_DECLARED", False)
    attempts = []

    def declare(**spec):
        attempts.append(spec["name"])
        if len(attempts) == 1:
            raise ValueError("registration unavailable")

    monkeypatch.setattr(telemetry_dictionary, "channel", declare)
    assert len(integration.declare_telemetry()) == 5
    assert integration._DECLARED is False
    assert len(integration.declare_telemetry()) == 6
    assert integration._DECLARED is True
