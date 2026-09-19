"""Cancelling a healthy generation is a deferral, not a broken endpoint.

Observed live on the desktop surface 2026-07-26:

    [MLX] Cortex ran past this turn's deadline (56.6s elapsed, budget 55.9s)
          but is healthy (heartbeat 1.7s ago, livelock ceiling 120.0s).
          Cancelling the request and KEEPING the warm lane.
    Circuit OPEN for Cortex on transient runtime failure.
          Reason: client_returned_no_text

The client deliberately ended a generation whose worker it had just described
as healthy, and kept the model warm on purpose. The router then read the empty
result as endpoint damage and opened the Cortex circuit — so the turn after it
could not use the real mind either, and the user got bounded filler instead of
an answer.

That is the recurring category error in this runtime: a deferral we chose,
counted as damage we suffered. These contracts pin the seam.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import MLXLocalClient
from core.brain.llm_health_router import (
    CircuitState,
    EndpointHealth,
    HealthAwareLLMRouter,
    _consume_deliberate_no_text_reason,
)


class _ClientWithDeliberateCancel:
    """Stands in for MLXLocalClient after a healthy-worker budget cancel."""

    def __init__(self, reason: str = "first_token_deadline_exceeded_worker_healthy"):
        self._deliberate_no_text_reason = reason

    def consume_deliberate_no_text_reason(self) -> str:
        reason = self._deliberate_no_text_reason or ""
        self._deliberate_no_text_reason = None
        return reason


def test_deliberate_cancel_reason_is_reported_once() -> None:
    """The reason is available exactly once, then gone."""
    client = _ClientWithDeliberateCancel()
    assert (
        _consume_deliberate_no_text_reason(client)
        == "first_token_deadline_exceeded_worker_healthy"
    )
    # A later empty result that failed for its own reasons must not inherit it.
    assert _consume_deliberate_no_text_reason(client) == ""


def test_reason_is_found_through_the_inference_gate_wrapper() -> None:
    """The foreground path hands the router a gate, not the MLX client."""
    gate = SimpleNamespace(_mlx_client=_ClientWithDeliberateCancel())
    assert (
        _consume_deliberate_no_text_reason(gate)
        == "first_token_deadline_exceeded_worker_healthy"
    )


def test_a_client_that_did_not_cancel_reports_nothing() -> None:
    """Silence here means 'this empty result was not our choice'."""
    assert _consume_deliberate_no_text_reason(_ClientWithDeliberateCancel("")) == ""
    assert _consume_deliberate_no_text_reason(SimpleNamespace()) == ""
    assert _consume_deliberate_no_text_reason(None) == ""


def test_a_broken_accessor_never_excuses_the_failure() -> None:
    """If we cannot prove we chose it, it counts as damage — fail closed."""

    class _Exploding:
        def consume_deliberate_no_text_reason(self) -> str:
            raise RuntimeError("client is wedged")

    assert _consume_deliberate_no_text_reason(_Exploding()) == ""


def test_mlx_client_publishes_the_reason_where_the_router_reads_it() -> None:
    """The producing side and the reading side agree on the seam."""
    from pathlib import Path

    source = Path("core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert "def consume_deliberate_no_text_reason" in source
    # Set at the healthy-worker cancellation, which is the only place we choose
    # to end a generation that the worker could still have completed.
    #
    # Counted as a STATEMENT, not as a run of characters. This matched a
    # multi-line literal carrying twenty-eight spaces of indentation, so the
    # method-size sweep re-indenting the block dropped the count to zero
    # while the assignment stayed exactly where it was.
    published = re.findall(
        r"self\._deliberate_no_text_reason\s*=\s*\(?\s*"
        r'"first_token_deadline_exceeded_worker_healthy"',
        source,
    )
    assert len(published) == 1, (
        "the reason must be published exactly at the healthy-worker cancel"
    )


def test_router_does_not_trip_the_circuit_on_a_deliberate_cancel() -> None:
    """The no-text branch consults the deferral before scoring damage."""
    from pathlib import Path

    source = Path("core/brain/llm_health_router.py").read_text(encoding="utf-8")
    tail = source[source.index("deliberate = _consume_deliberate_no_text_reason(client)") :]
    # The deferral is consulted BEFORE the damage path.
    assert tail.index("if deliberate:") < tail.index("ep.trip_temporarily")
    # …and the block it guards returns without ever reaching the trip.
    guarded = tail[tail.index("if deliberate:") : tail.index("if ep.is_local:")]
    assert 'f"deliberate_no_text:{deliberate}"' in guarded, (
        "a deliberate cancel must return its own non-damaging error"
    )
    assert "trip_temporarily" not in guarded, (
        "the deliberate-cancel path must not open the circuit"
    )
    # The ordinary broken-client path must still be able to open it.
    assert 'ep.trip_temporarily("client_returned_no_text")' in tail


def test_general_healthy_deadline_publishes_reason_and_preserves_worker() -> None:
    client = MLXLocalClient.__new__(MLXLocalClient)
    client._deliberate_no_text_reason = None
    client._deferred_reboot_reason = None
    cancellations = []
    client.soft_cancel_active_generation = cancellations.append

    client._mark_healthy_generation_deadline(foreground_request=True)

    assert cancellations == ["abandoned_generation_deadline"]
    assert client._deferred_reboot_reason is None
    assert (
        client.consume_deliberate_no_text_reason()
        == "generation_deadline_worker_healthy"
    )


@pytest.mark.asyncio
async def test_router_keeps_circuit_closed_for_general_healthy_deadline() -> None:
    class _HealthyDeadlineClient:
        def __init__(self):
            self._reason = "generation_deadline_worker_healthy"

        def is_available(self):
            return True

        async def think(self, *_args, **_kwargs):
            return None

        def consume_deliberate_no_text_reason(self):
            reason = self._reason
            self._reason = ""
            return reason

    endpoint = EndpointHealth(
        name="Cortex",
        url="local://cortex",
        model="Aura-32B",
        is_local=True,
        client=_HealthyDeadlineClient(),
    )
    router = HealthAwareLLMRouter()

    result = await router._call_endpoint(
        endpoint,
        "reason about this",
        None,
        5.0,
        origin="user",
    )

    assert result == {
        "ok": False,
        "error": "deliberate_no_text:generation_deadline_worker_healthy",
    }
    assert endpoint.state is CircuitState.CLOSED
    assert endpoint.failure_count == 0
    assert endpoint.lifetime_failure_count == 0
    assert endpoint.transient_trip_count == 0
