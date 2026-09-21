"""A turn running out of time is not a wedged worker.

LIVE DEFECT, 2026-07-25. Bryan asked a follow-up question and got nothing
back at all. The trace:

    First-token HARD CEILING exceeded (livelocked: heartbeats but zero
      tokens) for Aura-32B-... (82.5s elapsed, sla=240.0s, hard=82.0s).
    Cortex still sending heartbeats (1.8s ago). Recycling after this
      abandoned foreground request so late text cannot bleed into the
      next turn.
    Cortex generation exceeded inference-gate timeout 86.0s
    [MLX] Abort ... arrived after the generation finished ...; nothing to
      abort, leaving the worker up.
    Endpoint Cortex failed validation: client_returned_no_text
    Circuit OPEN for Cortex

Three things in that trace are worth separating.

The 82.0s "hard ceiling" was NOT the livelock formula — that computes
around 450s for a foreground request. It was the caller's remaining
wall-clock minus a reserve, derived from an 86s inference-gate budget. The
label said livelock; the number came from a deadline.

The generation FINISHED, a few seconds after we stopped waiting. The abort
arrived to find nothing to abort. The worker was never wedged — the turn
was slower than its budget under 80% RAM.

And then a healthy, heartbeating, 20GB warm model was marked for recycle,
which made the NEXT turn slower, which made the next deadline likelier to
expire. The recycle was the part of that cascade we chose.

Orphaned output is already fenced three ways: the pending generation is
dropped, the request id no longer matches, and the worker is soft-cancelled
between tokens. Destroying the warm lane was never what kept late text out
of the next turn.
"""
from __future__ import annotations

import re

from core.brain.llm import mlx_client
from tests.source_contract import family_text


def _abandonment_source() -> str:
    """The waiting method and the helpers lifted out of it.

    This used to be the slice between `livelock_ceiling = ...` and the soft
    cancel. The method-size sweep split that branch across a helper and its
    caller, so the two markers stopped bracketing one run of code and the
    slice no longer held `request_hard_ceiling` or the request-id fence —
    both still there, both still on the path, neither inside the cut.
    """
    from tests.source_contract import function_with_its_helpers

    return function_with_its_helpers(
        mlx_client, "MLXLocalClient._wait_for_generation_result"
    )


class TestTheTwoCeilingsAreDistinguished:
    def test_the_livelock_ceiling_is_computed_separately(self):
        """Previously the formula ceiling was immediately overwritten by
        min(formula, deadline), so by the time the branch ran there was no
        way to ask which one had fired."""
        assert "livelock_ceiling = self._first_token_hard_ceiling" in _abandonment_source()

    def test_the_livelock_verdict_tests_the_formula_ceiling(self):
        assert re.search(
            r"livelocked\s*=\s*elapsed_without_token\s*>\s*livelock_ceiling",
            _abandonment_source(),
        )

    def test_the_effective_ceiling_is_still_the_stricter_of_the_two(self):
        """Bounding by the caller's deadline must not regress — waiting past
        the caller is its own defect."""
        source = _abandonment_source()
        assert "min(" in source
        assert "request_hard_ceiling," in source


def _the_block_after(source: str, marker: str) -> str:
    """The indented body that follows `marker`, and nothing after it.

    A negative assertion needs a boundary. `_abandonment_source()` is the
    waiting method plus its helpers, so slicing "from here to the end"
    reaches into functions the branch has nothing to do with, and
    "_deferred_reboot_reason is absent" stops meaning anything.
    """
    lines = source.splitlines()
    at = next(i for i, line in enumerate(lines) if marker in line)
    opener = len(lines[at]) - len(lines[at].lstrip())
    body = [lines[at]]
    for line in lines[at + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= opener:
            break
        body.append(line)
    return "\n".join(body)


class TestRecycleIsReservedForRealWedges:
    def test_a_healthy_worker_past_its_deadline_is_not_recycled(self):
        """The branch Bryan's turn took. It must not set a reboot reason."""
        source = _abandonment_source()
        healthy_branch = _the_block_after(
            source[source.index("elif livelocked") :], "else:"
        )
        assert "_deferred_reboot_reason" not in healthy_branch
        assert "KEEPING the warm lane" in healthy_branch

    def test_a_genuine_livelock_still_recycles(self):
        """The protection this ceiling exists for is intact."""
        source = _abandonment_source()
        livelock_branch = source[
            source.index("elif livelocked"): source.index("else:", source.index("elif livelocked"))
        ]
        assert "_deferred_reboot_reason" in livelock_branch

    def test_a_silent_worker_still_recycles(self):
        """No heartbeat for 30s is a wedge regardless of which ceiling
        fired, and stays the hardest verdict."""
        source = _abandonment_source()
        assert "heartbeat_age > 30.0" in source
        silent_branch = source[
            source.index("if heartbeat_age > 30.0"): source.index("elif livelocked")
        ]
        assert 'self._deferred_reboot_reason = "first_token_sla_exceeded"' in silent_branch

    def test_the_missed_deadline_is_still_recorded(self):
        """Keeping the worker must not make the miss invisible — a turn that
        ran out of budget is real degradation even with a healthy lane."""
        source = _abandonment_source()
        assert "first_token_deadline_exceeded_worker_healthy" in source


class TestOrphanedOutputIsStillFenced:
    def test_the_pending_generation_is_dropped(self):
        assert "self._pending_generations.pop(req_id, None)" in _abandonment_source()

    def test_the_worker_is_soft_cancelled_between_tokens(self):
        source = family_text(mlx_client)
        assert 'soft_cancel_active_generation("abandoned_first_token_sla")' in source

    def test_the_request_id_guards_the_branch(self):
        """Fencing by identity is what actually stops bleed into the next
        turn; the recycle was only ever belt-and-braces on top of it."""
        assert "req_id == self._current_request_id" in _abandonment_source()


class TestTheWarmLaneSurvives:
    """The property, run rather than grepped.

    A healthy, heartbeating worker that overran the caller's deadline must be
    SOFT-CANCELLED and left loaded. Killing it costs a full model reload and
    makes the next turn slower, which makes the next deadline likelier to
    expire — the cascade the module docstring describes.
    """

    @staticmethod
    def _client_with_an_active_generation():
        from types import SimpleNamespace

        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client.model_path = "/models/Aura-32B-test"
        client._cancel_seq = SimpleNamespace(value=0)
        client._current_request_seq = 7
        client._current_request_started_at = 1.0
        return client

    def test_a_soft_cancel_marks_the_active_job_and_nothing_else(self):
        """The cancel channel is a single shared word; targeting is the whole
        contract. Cancelling job 7 must not stop job 8."""
        client = self._client_with_an_active_generation()

        receipt = client.soft_cancel_active_generation("abandoned_first_token_sla")

        assert receipt["requested"] is True
        assert receipt["reason"] == "abandoned_first_token_sla"
        assert client._cancel_seq.value == 7
        assert mlx_client.MLXLocalClient  # module intact

    def test_the_worker_stops_only_the_job_that_was_cancelled(self):
        """Worker side of the same contract, against the real predicate."""
        from types import SimpleNamespace

        from core.brain.llm.mlx_worker import soft_cancel_requested

        channel = SimpleNamespace(value=7)
        assert soft_cancel_requested(channel, 7) is True
        assert soft_cancel_requested(channel, 8) is False
        # Nothing requested at all.
        assert soft_cancel_requested(SimpleNamespace(value=0), 7) is False
        # No channel is not a licence to keep going forever, but it is also
        # not a cancel: the caller escalates instead of the worker guessing.
        assert soft_cancel_requested(None, 7) is False

    def test_a_cancel_with_no_active_generation_is_reported_not_faked(self):
        """`requested: False` is the honest answer, and callers escalate on it.

        Reporting a cancel that was never delivered would make the preemption
        ladder skip its next rung believing the cheap one worked.
        """
        client = self._client_with_an_active_generation()
        client._current_request_started_at = 0.0

        receipt = client.soft_cancel_active_generation("abandoned_first_token_sla")

        assert receipt["requested"] is False
        assert receipt["detail"] == "no_active_generation"

    def test_a_lost_cancel_write_is_reported_as_not_requested(self):
        """A concurrent job start can clear a superseded sequence.

        The parent used to report the cancel as requested regardless; it now
        writes, reads back, and says what actually happened — because a
        preemption ladder that believes a lost write succeeded never escalates.
        """

        class _Overwritten:
            """A channel that silently discards writes, as a racing start does."""

            value = 0

            def __setattr__(self, name, _value):
                object.__setattr__(self, name, 0)

        client = self._client_with_an_active_generation()
        client._cancel_seq = _Overwritten()

        receipt = client.soft_cancel_active_generation("abandoned_first_token_sla")

        assert receipt["requested"] is False
