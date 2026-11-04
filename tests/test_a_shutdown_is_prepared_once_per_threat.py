"""Her emergency protocol prepares a graceful shutdown once for a threat that stands, not on every evaluation.

On the full disk of 1 November it logged "CRITICAL THREAT ... preparing graceful shutdown" at error 61 times, flushing
her LoRA buffer each time, and shut nothing down.
"""
from __future__ import annotations

import pytest

from core.security import emergency_protocol as module

pytestmark = pytest.mark.unit


def test_prepared_once_and_again_after_the_threat_clears(monkeypatch):
    protocol = module.EmergencyProtocol.__new__(module.EmergencyProtocol)
    protocol._snapshot_taken = True
    protocol._last_snapshot_score = 1.0
    protocol._minimal_mode = True
    protocol._shutdown_prepared = False
    prepared: list[float] = []
    monkeypatch.setattr(protocol, "_graceful_shutdown", lambda: prepared.append(protocol._threat_score))
    for score in (0.95, 0.97, 0.99):
        protocol._threat_score = score
        protocol._respond_to_threat()
    assert prepared == [0.95]
    protocol._threat_score = 0.5
    protocol._respond_to_threat()
    protocol._threat_score = 0.95
    protocol._respond_to_threat()
    assert prepared == [0.95, 0.95]
