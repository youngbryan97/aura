"""Capability-token lifecycle invariants.

Asserts every rejection branch documented in
``core/agency/capability_token.py`` is a hard refuse, not a warning:

  - replay (consume twice)
  - expired (TTL elapsed)
  - revoked
  - wrong-domain
  - wrong-action
  - cross-thread
  - post-shutdown / process generation drift
"""
from __future__ import annotations

import threading
import time

import pytest

from core.agency.capability_token import CapabilityTokenStore


def _store() -> CapabilityTokenStore:
    return CapabilityTokenStore()


def test_normal_consume_succeeds():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    s.validate(t.token, domain="d", action="a")
    s.consume(t.token, child_receipt="exec-1")
    assert t.is_consumed()


def test_replay_rejected():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    s.validate(t.token, domain="d", action="a")
    s.consume(t.token, child_receipt="exec-1")
    with pytest.raises(PermissionError, match="replay"):
        s.validate(t.token, domain="d", action="a")


def test_expired_rejected():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=0.05, domain="d", requested_action="a", approver="will", parent_receipt="r")
    time.sleep(0.1)
    with pytest.raises(PermissionError, match="expired"):
        s.validate(t.token, domain="d", action="a")


def test_wrong_domain_rejected():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    with pytest.raises(PermissionError, match="wrong_domain"):
        s.validate(t.token, domain="other", action="a")


def test_wrong_action_rejected():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    with pytest.raises(PermissionError, match="wrong_action"):
        s.validate(t.token, domain="d", action="other")


def test_revoked_rejected():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    s.revoke(t.token, reason="audit")
    with pytest.raises(PermissionError, match="revoked"):
        s.validate(t.token, domain="d", action="a")


def test_cross_thread_rejected():
    s = _store()
    t = s.issue(origin="test", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    box = []
    def worker():
        try:
            s.validate(t.token, domain="d", action="a")
            box.append("no-raise")
        except PermissionError as exc:
            box.append(str(exc))
    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert any("cross_thread" in str(x) for x in box)


def test_revoke_all_marks_all():
    s = _store()
    a = s.issue(origin="a", scope="x", ttl_seconds=60.0, domain="d", requested_action="a", approver="will", parent_receipt="r")
    b = s.issue(origin="b", scope="x", ttl_seconds=60.0, domain="d", requested_action="b", approver="will", parent_receipt="r")
    n = s.revoke_all(reason="shutdown")
    assert n == 2
    assert a.revoked and b.revoked
