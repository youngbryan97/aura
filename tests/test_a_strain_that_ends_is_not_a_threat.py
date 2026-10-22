"""A resource strain that has ended stops counting as a threat.

The resource monitor flagged a strain while it lasted and said nothing when it
ended, and the emergency protocol recomputed its score only on a new flag. In
the validation run of 21 October a declared strain came and went each round and
the score climbed over four of them into minimal mode, which it never left.
"""
from __future__ import annotations

import pytest

from core.security import emergency_protocol as ep
from core.security.enforcement import ResourceMonitor


@pytest.fixture()
def protocol(monkeypatch):
    found = ep.EmergencyProtocol()
    monkeypatch.setattr(found, "_log_threat", lambda _signal: None)
    monkeypatch.setattr(found, "_take_encrypted_snapshot", lambda: None)
    monkeypatch.setattr(found, "_rotate_vault_snapshots", lambda: None)
    monkeypatch.setattr(found, "_graceful_shutdown", lambda: None)
    return found


def test_a_cleared_strain_takes_its_score_and_minimal_mode_with_it(protocol):
    for _ in range(6):
        protocol.flag_threat("immune:resource_monitor", "resource strain: mem at 93%", 0.9)
    assert protocol.is_minimal_mode
    protocol.clear_threat("immune:resource_monitor")
    assert protocol.threat_score == 0.0
    assert not protocol.is_minimal_mode


def test_clearing_one_source_leaves_another_standing(protocol):
    protocol.flag_threat("hostile_user_interaction", "hostile signals", 0.5)
    for _ in range(4):
        protocol.flag_threat("immune:resource_monitor", "resource strain: cpu at 95%", 0.9)
    before = protocol.threat_score
    protocol.clear_threat("immune:resource_monitor")
    assert 0.0 < protocol.threat_score <= before
    assert [signal.source for signal in protocol._signals] == ["hostile_user_interaction"]


def test_the_monitor_says_when_its_strain_ends(monkeypatch, protocol):
    monkeypatch.setattr(ep, "get_emergency_protocol", lambda: protocol)
    cleared: list[str] = []
    monkeypatch.setattr(protocol, "clear_threat", cleared.append)
    monitor = ResourceMonitor()
    readings = iter([{"cpu": 10.0, "mem": 95.0, "disk": 10.0}, {"cpu": 10.0, "mem": 40.0, "disk": 10.0}])
    monkeypatch.setattr(monitor, "sample", lambda: next(readings))
    assert monitor.check_and_report() is not None
    assert monitor.check_and_report() is None
    assert cleared == ["immune:resource_monitor"]
