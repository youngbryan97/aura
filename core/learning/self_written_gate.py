"""core/learning/self_written_gate.py — a change she wrote, gated by a test she wrote.

Aura can synthesise tests and can modify her own code, and the two are not
connected. A self-modification is admitted on review and on the suite passing,
which means it is admitted on the suite that existed before the change - and a
change whose whole point is new behaviour is exactly the change the old suite
cannot judge.

The gate is one rule: **a change is admissible only if it carries a test that
fails without it**. Not a test that passes with it - that is satisfied by
``assert True``. The test must be run against the unmodified code and observed
to fail, which is the only evidence that it is testing the change rather than
the weather.

Three refusals fall out of that:

* **A test that passes on the unmodified code** is not testing this change.
* **A test that errors on the unmodified code** is ambiguous: an import error
  is not a failing assertion, and accepting it would let a test that cannot run
  at all count as evidence.
* **A change with no test** is refused whatever the change is, including a
  change that removes code. Deleting something is a behavioural claim too.

Coverage is not the metric
--------------------------
This does not ask whether the test covers the changed lines. Coverage is easy
to satisfy and does not establish anything: a test that executes a line and
asserts nothing about it covers it. Failing before and passing after is the
weaker-sounding and stronger property.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["Outcome", "Change", "SelfWrittenGate", "GateVerdict"]


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"


@dataclass(frozen=True, slots=True)
class Change:
    """One self-authored modification and the test she wrote for it."""

    change_id: str
    description: str
    apply: Callable[[], Any]
    revert: Callable[[], Any]
    test: Callable[[], Outcome] | None = None
    test_name: str = ""


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Whether the change may land, and the two observations behind it."""

    change_id: str
    admitted: bool
    before: Outcome | None = None
    after: Outcome | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "admitted": self.admitted,
            "test_before_change": self.before.value if self.before else None,
            "test_after_change": self.after.value if self.after else None,
            "reason": self.reason,
        }


class SelfWrittenGate:
    """Run the test against the unmodified code, then against the change."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._verdicts: list[GateVerdict] = []

    def admit(self, change: Change) -> GateVerdict:
        if change.test is None:
            return self._record(GateVerdict(
                change.change_id, False,
                reason="no test; removing code is a behavioural claim too",
            ))

        before = self._run(change.test)
        if before is Outcome.PASSED:
            return self._record(GateVerdict(
                change.change_id, False, before,
                reason="the test passes without the change, so it is not testing it",
            ))
        if before is Outcome.ERRORED:
            return self._record(GateVerdict(
                change.change_id, False, before,
                reason=(
                    "the test errored on the unmodified code; an import error is not a "
                    "failing assertion and cannot count as evidence"
                ),
            ))

        change.apply()
        after = self._run(change.test)
        if after is not Outcome.PASSED:
            change.revert()
            return self._record(GateVerdict(
                change.change_id, False, before, after,
                reason=f"the test still {after.value} with the change applied",
            ))
        return self._record(GateVerdict(change.change_id, True, before, after))

    @staticmethod
    def _run(test: Callable[[], Outcome]) -> Outcome:
        try:
            return test()
        except AssertionError:
            return Outcome.FAILED
        except Exception:  # noqa: BLE001 - anything else is an error, not a failure
            return Outcome.ERRORED

    def _record(self, verdict: GateVerdict) -> GateVerdict:
        with self._lock:
            self._verdicts.append(verdict)
        return verdict

    def report(self) -> dict[str, Any]:
        with self._lock:
            verdicts = list(self._verdicts)
        return {
            "considered": len(verdicts),
            "admitted": sum(1 for v in verdicts if v.admitted),
            "refused": [v.to_dict() for v in verdicts if not v.admitted],
            "admission_rate": (
                sum(1 for v in verdicts if v.admitted) / len(verdicts) if verdicts else None
            ),
        }
