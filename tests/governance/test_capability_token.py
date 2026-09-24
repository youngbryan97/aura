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


def _issued(store):
    return store.issue(
        origin="chat", scope="tool_execution:desktop_task:user", ttl_seconds=60.0,
        domain="tool_execution", requested_action="foreground_desktop_action",
        approver="gateway", parent_receipt="r",
    )


def test_a_task_the_request_started_may_use_its_token():
    """The work a request starts runs in tasks of its own.

    LIVE 2026-09-23: the desktop task a chat turn started asked for
    computer_use, and the turn's own token was refused as cross-task on every
    launch.
    """
    import asyncio

    async def request():
        store = CapabilityTokenStore()
        tok = _issued(store)

        async def its_own_step():
            return store.validate(
                tok.token, domain="tool_execution", action="foreground_desktop_action"
            )

        return await asyncio.create_task(its_own_step())

    assert asyncio.run(request()).token.startswith("CT-")


def test_an_unrelated_task_still_may_not():
    import asyncio

    import pytest

    async def two_requests():
        store = CapabilityTokenStore()
        issued: dict = {}
        ready = asyncio.Event()

        async def first():
            issued["token"] = _issued(store).token
            ready.set()

        async def second():
            await ready.wait()
            store.validate(
                issued["token"], domain="tool_execution", action="foreground_desktop_action"
            )

        # Both started before the token existed, so neither inherits it.
        one, other = asyncio.create_task(first()), asyncio.create_task(second())
        await one
        await other

    with pytest.raises(PermissionError, match="capability_token_cross_task"):
        asyncio.run(two_requests())
