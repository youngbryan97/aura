"""The primary lane's warmup may run while a foreground turn waits for it.

LIVE, 2026-09-21. The boot warmup asked the cortex for a one-token
precompile and then for the visible readiness probe. Both were refused by
``_generate_inner``'s foreground-ownership guard, which returns ``None``
without saying why, so the prover reported ``no_text`` — a claim about the
worker, which had never been asked anything. The lane went to ``recovering``
with ``warmup_readiness_no_text`` and the chat request that was waiting for
readiness got nothing back.

The exemption for the primary lane already existed, one level up, at the
warmup gate in ``mlx_warmup_and_adapters``, with the 2026-07-10 deadlock it
was written for in its comment. It was written there and not at the line that
does the refusing.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from core.brain.llm import mlx_client as mlx
from core.brain.llm.mlx_client import was_declined_before_the_worker


class _Lane:
    """Only the parts of the client the guard reads."""

    model_path = "/models/Aura-Qwen3.8-27B-persona-crsm"

    def __init__(self, *, primary: bool) -> None:
        self._primary = primary
        self._deliberate_no_text_reason: str | None = None
        self.reached_the_worker = False

    def _is_primary_lane(self) -> bool:
        return self._primary

    def _coerce_generation_kwargs(self, *a, **k):
        raise _PastTheGuard

    def __getattr__(self, name: str):
        # Anything the method reaches for after the guard means the guard let
        # it through. Reading a private attribute is not the same as calling
        # the worker, so only a call counts.
        if name.startswith("__"):
            raise AttributeError(name)

        def _reached(*_a, **_k):
            raise _PastTheGuard

        return _reached

    consume_deliberate_no_text_reason = (
        mlx.MLXLocalClient.consume_deliberate_no_text_reason
    )


class _PastTheGuard(Exception):
    """Raised by the first thing ``_generate_inner`` does after the guard."""


async def _past_the_guard(lane: _Lane, **kwargs) -> str | None:
    """Drive the REAL ``_generate_inner`` as far as the guard.

    A double that re-implements the guard would keep passing after somebody
    deleted the exemption, so this calls the shipped method and lets the next
    thing it touches say that control got through.
    """
    try:
        return await mlx.MLXLocalClient._generate_inner(lane, "hi", **kwargs)
    except _PastTheGuard:
        lane.reached_the_worker = True
        return "ready"


@pytest.fixture
def owned_foreground(monkeypatch):
    monkeypatch.setattr(mlx, "_foreground_owner_active", lambda: True)


def test_the_primary_lanes_readiness_probe_is_not_refused(owned_foreground):
    lane = _Lane(primary=True)
    said = asyncio.run(
        _past_the_guard(lane, request_is_background=True, health_probe=True)
    )
    assert said == "ready"
    assert lane.reached_the_worker


def test_the_primary_lanes_precompile_is_not_refused(owned_foreground):
    lane = _Lane(primary=True)
    said = asyncio.run(
        _past_the_guard(lane, request_is_background=True, warmup_precompile=True)
    )
    assert said == "ready"


def test_ordinary_background_work_still_yields_to_the_turn(owned_foreground):
    lane = _Lane(primary=True)
    said = asyncio.run(_past_the_guard(lane, request_is_background=True))
    assert said is None
    assert not lane.reached_the_worker


def test_a_background_lanes_warmup_still_yields_to_the_turn(owned_foreground):
    lane = _Lane(primary=False)
    said = asyncio.run(
        _past_the_guard(lane, request_is_background=True, health_probe=True)
    )
    assert said is None


def test_the_refusal_says_it_was_a_refusal(owned_foreground):
    lane = _Lane(primary=True)
    asyncio.run(_past_the_guard(lane, request_is_background=True))
    assert lane._deliberate_no_text_reason == "skipped_during_foreground_ownership"
    assert was_declined_before_the_worker(lane._deliberate_no_text_reason)


def test_a_worker_that_answered_nothing_is_not_called_a_refusal():
    assert not was_declined_before_the_worker("")
    assert not was_declined_before_the_worker("generation_deadline_worker_healthy")
    assert was_declined_before_the_worker("stopped_before_worker_spawn:cortex_startup_quiet")


def test_the_prover_reports_the_refusal_rather_than_no_text():
    """``prove_visible_readiness`` must not report a refusal as empty output."""
    client = types.SimpleNamespace(
        _deliberate_no_text_reason="skipped_during_foreground_ownership",
        _last_visible_readiness_at=0.0,
    )
    client.is_alive = lambda: True
    client.consume_deliberate_no_text_reason = types.MethodType(
        mlx.MLXLocalClient.consume_deliberate_no_text_reason, client
    )

    async def _declined(*_a, **_k):
        return None

    client._generate_inner = _declined
    client._set_lane_state = lambda *a, **k: None

    proved = asyncio.run(
        mlx.MLXLocalClient.prove_visible_readiness(client, budget_s=1.0)
    )
    assert proved == "declined:skipped_during_foreground_ownership"
    assert client._last_visible_readiness_at == 0.0


def test_a_declined_probe_does_not_mark_the_lane_recovering():
    """The warmup stands down; it does not record a failure against her."""
    import inspect

    from core.brain.llm import mlx_warmup_and_adapters as warmup

    source = inspect.getsource(warmup._WarmsUpAndSwapsAdapters._run_warmup_precompile)
    declined = source.index('proved.startswith("declined:")')
    recovering = source.index('self._set_lane_state("recovering", f"warmup_readiness_')
    assert declined < recovering, (
        "the declined branch must be taken before the lane is marked recovering"
    )
