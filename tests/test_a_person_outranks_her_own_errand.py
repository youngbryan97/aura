"""A run that plays a game for forty minutes should not make her unable to talk.

The lane that serves conversation is held by one owner at a time, and a person
outranks background work immediately — unless the holder is marked user-facing,
in which case it cannot be preempted at all.

Every move of a long desktop task is a foreground generation, so every move
marked the lane user-facing. The person who started the errand then could not
interrupt it: asking her anything while she played got "no endpoints matched
routing plan for tier 'primary'" and, at the far end, an apology.

LIVE 2026-08-29, mid-2048: "You've been playing for a while. What do you
actually know about how that board behaves?" — refused, while she was playing.

Being foreground and having somebody waiting are different questions. An answer
that is read by code has nobody waiting on it, however foreground the errand it
belongs to, and `internal_inference` already says exactly that.
"""

from __future__ import annotations

import inspect

import pytest

from core.brain.llm import mlx_client


def owner_context_source():
    return inspect.getsource(mlx_client._foreground_owner_context)


# ── the distinction exists ───────────────────────────────────────────────

def test_being_foreground_and_having_somebody_waiting_are_different_questions():
    assert "a_person_is_waiting" in inspect.signature(
        mlx_client._foreground_owner_context
    ).parameters


def test_what_protects_a_holder_is_whether_a_person_is_waiting():
    body = owner_context_source()
    lines = body.splitlines()
    setting = next(
        n for n, line in enumerate(lines)
        if "_FOREGROUND_OWNER_IS_USER_FACING = bool(" in line
    )
    # The value it is set FROM, not the global declaration above it.
    assert any("a_person_is_waiting" in line for line in lines[setting : setting + 5])


def test_a_caller_that_says_nothing_keeps_the_old_meaning():
    """Nothing that has not been taught the difference loses its protection."""
    body = owner_context_source()
    assert "if a_person_is_waiting is None" in body


# ── and the generation sites use it ──────────────────────────────────────

def test_a_generation_read_by_code_does_not_claim_somebody_is_waiting():
    source = inspect.getsource(mlx_client)
    claims = [
        line
        for line in source.splitlines()
        if "a_person_is_waiting=" in line and not line.strip().startswith("#")
    ]
    assert len(claims) >= 2, "a generation site still claims a person is waiting on it"
    assert all("internal_inference" in line or "not bool(" in line for line in claims)


def test_her_own_reasoning_still_says_it_is_read_by_code():
    """The flag this rests on has to keep being set where decisions are made."""
    from core.agency import her_reasoning

    assert "internal_inference=True" in inspect.getsource(her_reasoning)


def test_and_a_person_still_outranks_background_work():
    body = owner_context_source()
    assert "A person outranks background work, immediately." in body


# ── and a warmup does not tear down the answer it exists to enable ───────

def test_a_warmup_retry_stands_down_while_somebody_is_being_answered():
    """LIVE 2026-08-29: a person interrupted a long errand, was correctly given
    the lane, and had her answer cancelled underneath her by a warmup retry —
    "generation cancelled during expected reboot (warmup_precompile_retry)" —
    receiving a stub about being cut short.

    A warmup exists to make the lane ready. Tearing the worker down mid-reply
    to do it defeats the thing it is for.
    """
    body = inspect.getsource(mlx_client.MLXLocalClient._recover_worker_for_warmup_retry)
    assert "_FOREGROUND_OWNER_IS_USER_FACING" in body
    assert "stood down" in body


def test_and_it_gives_up_rather_than_waiting_for_ever():
    body = inspect.getsource(mlx_client.MLXLocalClient._recover_worker_for_warmup_retry)
    assert "_WAIT_OUT_A_REPLY_S" in body
    assert mlx_client._WAIT_OUT_A_REPLY_S <= 60.0


def test_the_wait_is_longer_than_a_reply_takes():
    assert mlx_client._WAIT_OUT_A_REPLY_S >= 10.0
