"""One unvalidated depth setting took the conversation surface down.

Live 2026-07-28. Asked to reverse-engineer 2048 onto the Desktop, nothing
happened — no reply, no build, and every message sent afterwards came back
with "I still have the previous turn open." The log::

    Cortex returned no text on user-facing request. Retrying once after 2s...
    Cortex-RETRY-1 produced an unsafe user-facing draft (too_short_for_user_turn, len=5)
    Cortex bounded retry failed.
    Cortex failed validation: user_facing_assessment_rejected
    ...
    recurrent loops=2 (was None, depth_present=True)

The chain: a request carrying several action verbs and a desktop surface is
classed "extended" by ``_foreground_compute_profile``, extended asks for
recurrent depth 2, the cortex returned five characters, the turn died, and the
foreground lane latched busy — so the failure was not one lost turn but every
turn after it.

Depth 1 is the identity depth: CP226 measured a T=1 gap of 0.0 against the live
weights, so a single pass is the model answering normally. Depth 2 is a
different claim and the evidence for it does not exist — the CP227 accuracy
gate meant to establish it ran with the adapter dark outside
``recurrence_adapter_scope``, was voided, and has never been re-run.

So the live ceiling is 1. Raising it is an experiment, and an experiment has to
be asked for rather than arrived at by prompt shape.
"""
from __future__ import annotations

import pytest

from core.brain.llm import mlx_worker


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("AURA_USER_SURFACE_RECURRENT_MAX_LOOPS", raising=False)
    monkeypatch.delenv("AURA_USER_SURFACE_RECURRENT_LOOPS", raising=False)


def test_the_live_ceiling_is_the_identity_depth() -> None:
    assert mlx_worker._live_recurrent_ceiling() == 1


def test_the_request_that_broke_it_now_runs_at_depth_one() -> None:
    """The extended profile asks for 2; the surface refuses to go there."""
    assert mlx_worker._surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": 2}
    ) == 1


@pytest.mark.parametrize("asked", [2, 3, 5, 9, 100])
def test_no_prompt_shape_can_buy_extra_depth(asked: int) -> None:
    assert mlx_worker._surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": asked}
    ) == 1


@pytest.mark.parametrize("asked", [0, -1, None, "", "nonsense"])
def test_a_missing_or_bad_value_still_answers_normally(asked) -> None:
    """Depth must never fall below one — that would be no forward pass."""
    assert mlx_worker._surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": asked}
    ) == 1


def test_a_turn_that_asks_for_nothing_gets_the_identity_depth() -> None:
    assert mlx_worker._surface_control_recurrent_loops({}) == 1


def test_depth_two_is_available_when_explicitly_asked_for(monkeypatch) -> None:
    """The lane is not deleted, it is gated. Supervised runs can still open it."""
    monkeypatch.setenv("AURA_USER_SURFACE_RECURRENT_MAX_LOOPS", "2")
    assert mlx_worker._live_recurrent_ceiling() == 2
    assert mlx_worker._surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": 2}
    ) == 2


def test_an_explicit_ceiling_is_still_a_ceiling(monkeypatch) -> None:
    monkeypatch.setenv("AURA_USER_SURFACE_RECURRENT_MAX_LOOPS", "2")
    assert mlx_worker._surface_control_recurrent_loops(
        {"clean_user_surface_recurrent_loops": 9}
    ) == 2


def test_surface_receipt_does_not_claim_a_clamped_depth_was_applied() -> None:
    receipt = mlx_worker._surface_generation_control_receipt(
        {
            "clean_user_surface_recurrent_loops": 2,
            "max_tokens": 256,
        },
        {
            "enabled": True,
            "recurrent_inner": object(),
            "recurrent_runtime_loops_applied": 1,
        },
    )

    assert receipt["recurrent_runtime_loops_requested"] == 2
    assert receipt["recurrent_runtime_loops_applied"] == 1
    assert receipt["recurrent_runtime_loops_applied_ok"] is False
    assert receipt["applied"] is False


def test_surface_receipt_proves_an_exact_depth_application() -> None:
    receipt = mlx_worker._surface_generation_control_receipt(
        {
            "clean_user_surface_recurrent_loops": 1,
            "max_tokens": 256,
        },
        {
            "enabled": True,
            "recurrent_inner": object(),
            "recurrent_runtime_loops_applied": 1,
        },
    )

    assert receipt["recurrent_runtime_loops_applied_ok"] is True
    assert receipt["applied"] is True
