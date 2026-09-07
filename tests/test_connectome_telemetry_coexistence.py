"""Connectome observations must coexist with the felt-state channels.

The counts come from ``WRITTEN_CHANNELS`` rather than from a literal, so adding
a channel does not fail a test about coexistence — the invariant that every
written channel is declared is the one that should catch a missing declaration,
and it does.
"""

import subprocess
import sys

from core.connectome import integration
from core.connectome.invariants import WRITTEN_CHANNELS
from core.fsw import telemetry_dictionary


def test_connectome_and_interiority_register_in_one_process():
    result = subprocess.run(
        [sys.executable, "-c", """
import core.interiority.telemetry
from core.connectome.integration import declare_telemetry
from core.connectome.invariants import WRITTEN_CHANNELS
declared = declare_telemetry()
assert set(declared) == set(WRITTEN_CHANNELS), sorted(set(WRITTEN_CHANNELS) - set(declared))
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
    total = len(WRITTEN_CHANNELS)
    assert len(integration.declare_telemetry()) == total - 1
    assert integration._DECLARED is False
    assert len(integration.declare_telemetry()) == total
    assert integration._DECLARED is True
