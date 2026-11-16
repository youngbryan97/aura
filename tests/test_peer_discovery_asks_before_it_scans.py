"""Her peer discovery asks whether a standing grant would let it scan before it scans.

LIVE 14 to 16 November 2026: her belief-sync discovery loop asked for a LAN scan (sovereign_network, mode discovery)
under no origin, no standing grant names one, and each attempt was a constitution refusal in her log, 28 of them.
"""
from __future__ import annotations

import logging

import pytest

from core.collective import belief_sync
from tests.test_standing_authority import _manager

pytestmark = pytest.mark.unit


def test_no_standing_grant_lets_her_scan_for_peers():
    assert _manager().grant_that_would_lease("sovereign_network", dict(belief_sync._DISCOVERY),
                                             origin=belief_sync._DISCOVERY_ORIGIN) == ""


def test_without_a_grant_she_does_not_scan_and_says_so_once(monkeypatch, caplog):
    asked: list[str] = []

    class Authority:
        def grant_that_would_lease(self, tool, arguments, *, origin):
            asked.append(origin)
            return ""

    monkeypatch.setattr(belief_sync.ServiceContainer, "get",
                        lambda name, default=None: Authority() if name == "standing_authority" else default)
    sync = belief_sync.BeliefSync.__new__(belief_sync.BeliefSync)

    with caplog.at_level(logging.INFO):
        assert sync._may_look_for_peers() is False
        assert sync._may_look_for_peers() is False

    assert asked == [belief_sync._DISCOVERY_ORIGIN] * 2
    assert sum("discovery waits for one" in record.getMessage() for record in caplog.records) == 1


def test_with_a_grant_she_may(monkeypatch):
    class Authority:
        def grant_that_would_lease(self, tool, arguments, *, origin):
            return "aura.peer-discovery"

    monkeypatch.setattr(belief_sync.ServiceContainer, "get",
                        lambda name, default=None: Authority() if name == "standing_authority" else default)
    sync = belief_sync.BeliefSync.__new__(belief_sync.BeliefSync)

    assert sync._may_look_for_peers() is True
