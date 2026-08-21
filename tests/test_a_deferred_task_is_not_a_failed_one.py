"""Waiting is not failing, and a failure has to say why.

LIVE, 2026-08-20. "build me a small web app" was accepted into governed
background execution and recorded as FAILED 58 milliseconds later, with
error=None and nothing in the log. The engine had returned

    deferred_reason = "background inference admission"
    summary = "Planning is queued until background inference admission clears."

Planning was queued, not broken. Nothing read `deferred_reason` — a writer
with no reader — so it was filed as a failure, and the person had been told
work was underway while the task was already dead.
"""

from __future__ import annotations

from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "core" / "agency" / "task_commitment_verifier.py"


class _Deferred:
    succeeded = False
    summary = "Planning is queued until background inference admission clears."
    deferred_reason = "background inference admission"


class _Failed:
    succeeded = False
    summary = "The plan could not be grounded."
    deferred_reason = ""


def _verifier_class():
    from core.agency.task_commitment_verifier import TaskCommitmentVerifier

    return TaskCommitmentVerifier


def test_a_deferral_is_recognised() -> None:
    assert _verifier_class()._deferral_reason(_Deferred()) == "background inference admission"


def test_an_ordinary_failure_is_not_a_deferral() -> None:
    assert _verifier_class()._deferral_reason(_Failed()) == ""
    assert _verifier_class()._deferral_reason(object()) == ""


def test_a_deferred_task_is_recorded_as_deferred() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    assert 'status="deferred"' in source
    assert "deferred_attempts" in source


def test_a_failure_carries_its_reason() -> None:
    """The ledger showed "failed" beside error=None, with the reason living
    only in the summary and nothing logging it."""
    source = VERIFIER.read_text(encoding="utf-8")
    assert source.count('error="" if succeeded else str(summary or "")[:400]') >= 1


def test_the_retry_is_bounded_and_says_when_it_gives_up() -> None:
    """A deferral that is never retried is a failure with better manners; one
    that retries forever is a loop."""
    cls = _verifier_class()
    assert cls.DEFERRED_RETRY_LIMIT >= 1
    assert cls.DEFERRED_RETRY_DELAY_S > 0
    source = VERIFIER.read_text(encoding="utf-8")
    body = source[source.index("async def _retry_deferred_task") :]
    body = body[: body.index("\n    def ", 10)]
    assert "still waiting on" in body
    assert "attempts > self.DEFERRED_RETRY_LIMIT" in body


def test_a_retry_is_never_blind() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    assert "_remember_dispatch_state" in source
    body = source[source.index("def _remember_dispatch_state") :]
    body = body[: body.index("\n    async def ", 10)]
    assert "len(self._deferred_states) > 32" in body


def test_a_deferral_is_its_own_outcome() -> None:
    """Distinct from STARTED, which tells the person to wait, and from FAILED,
    which tells them it broke."""
    from core.agency.task_commitment_verifier import DispatchOutcome

    assert DispatchOutcome.DEFERRED.value == "deferred"
    assert DispatchOutcome.DEFERRED is not DispatchOutcome.STARTED
    assert DispatchOutcome.DEFERRED is not DispatchOutcome.FAILED


def test_nothing_launched_means_nothing_promised() -> None:
    """LIVE: planning was deferred every time and by a different gate each
    restart — foreground_chat_active, then foreground_quiet_window, then
    welfare_memory_integrity_0.18. Background planning never ran, and the
    person was told to wait for something that could not begin. Inline, the
    same request produced the page in 33 seconds."""
    unitary = Path(__file__).resolve().parents[1] / "core" / "phases" / "response_generation_unitary.py"
    source = unitary.read_text(encoding="utf-8")
    assert 'if last_task_outcome == "deferred":' in source
    assert source.index('last_task_outcome == "deferred"') < source.index(
        'last_task_outcome == "started"'
    )


def test_the_inline_dispatch_reports_a_deferral_as_one() -> None:
    source = VERIFIER.read_text(encoding="utf-8")
    assert "outcome = DispatchOutcome.DEFERRED" in source
