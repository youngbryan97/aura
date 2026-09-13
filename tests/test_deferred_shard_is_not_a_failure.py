"""A queued swarm shard has produced nothing because it never ran.

The 2026-07-25 capability run raised twenty of these:

    NEW INCIDENT [degraded] degradation:delegator:
    RuntimeError: Swarm cognitive engine returned empty output

with this immediately above each one:

    Router: Queueing background inference until admission clears for
    origin=response_generation_background_reflection
    reason=foreground_headroom_reserved

Admission was doing exactly its job — holding a background shard so a real
person on the foreground lane gets the machine — and the delegator recorded
that as a broken cognitive engine, twenty times, with an incident each.

A shard that was deferred is deferred. One that actually ran and produced
nothing is still a failure and still raises.
"""
from __future__ import annotations

import pytest

from core.collective.delegator import _deferred_generation_reason

pytestmark = pytest.mark.unit


class Result:
    def __init__(self, **fields):
        for k, v in fields.items():
            setattr(self, k, v)


class TestDeferralIsRecognised:
    @pytest.mark.parametrize(
        "field",
        ["status", "error", "reason", "deferral_reason"],
    )
    def test_any_status_field_can_carry_it(self, field):
        assert _deferred_generation_reason(
            Result(**{field: "queued until admission clears"})
        )

    @pytest.mark.parametrize(
        "reason",
        [
            "foreground_headroom_reserved",
            "queued until admission clears",
            "model_load_admission_denied",
            "warmup_deferred",
            "resource_busy",
            "backpressure",
        ],
    )
    def test_the_live_deferral_shapes(self, reason):
        assert _deferred_generation_reason(Result(status=reason)) == reason

    def test_a_dict_result_works_too(self):
        assert _deferred_generation_reason(
            {"status": "queued until admission clears"}
        )


class TestARealFailureStillFails:
    @pytest.mark.parametrize(
        "reason",
        [
            "model_crashed",
            "tokenizer_error",
            "ValueError: bad prompt",
            "",
        ],
    )
    def test_a_genuine_error_is_not_a_deferral(self, reason):
        assert _deferred_generation_reason(Result(status=reason)) == ""

    def test_a_bare_result_is_not_a_deferral(self):
        assert _deferred_generation_reason(object()) == ""

    def test_none_is_not_a_deferral(self):
        assert _deferred_generation_reason(None) == ""


class TestTheShardPath:
    def test_the_deferral_branch_returns_rather_than_raising(self):
        import inspect

        from core.collective import delegator

        src = inspect.getsource(delegator)
        block = src[src.index("if not agent.result:") :][:2200]
        assert "_deferred_generation_reason" in block
        assert 'agent.status = "DEFERRED"' in block, (
            "a queued shard must be recorded as deferred, not failed"
        )
        # The raise must still be reachable for a shard that genuinely ran.
        assert "Swarm cognitive engine returned empty output" in block


class TestAStatusCodeIsAnIdentifier:
    """Every marker was missed, because `\\b` cannot see inside an identifier.

    `names_any("foreground_headroom_reserved", ("headroom",))` is False: an
    underscore is a word character, so there is no boundary beside it. The
    recogniser therefore returned "" for every deferral it lists — including
    `background_deferred:memory_pressure`, which the markers' own comment says
    "deferred" already catches — and a shard that never ran raised "Swarm
    cognitive engine returned empty output".
    """

    def test_the_live_identifier_shapes_are_recognised(self):
        from core.collective.delegator import _marks_a_deferral

        for status in (
            "foreground_headroom_reserved",
            "model_load_admission_denied",
            "background_deferred:memory_pressure",
            "candidate_worker_not_ready",
            "generation_queued",
            "warmup_in_progress",
        ):
            assert _marks_a_deferral(status), status

    def test_a_word_that_merely_contains_a_marker_is_not_one(self):
        """The word-awareness the boundary matching was for, kept."""
        from core.collective.delegator import _marks_a_deferral

        for status in ("headroomless", "readmission", "queuedish"):
            assert not _marks_a_deferral(status), status

    def test_a_worker_that_died_still_raises(self):
        """Deliberately absent from the markers, and it has to stay absent."""
        from core.collective.delegator import _marks_a_deferral

        assert not _marks_a_deferral("worker_died_during_generation")
        assert not _marks_a_deferral("ok")
