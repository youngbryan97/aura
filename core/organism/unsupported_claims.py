"""Claims registered against tests, and the ones with nothing behind them.

A claim about Aura has to name the test that validates it. This is what reads
that register back: which claims have no passing test, which are measured only
here rather than in the runtime, and the report that says so plainly.
"""
from __future__ import annotations

from typing import Any


class _SaysWhichClaimsHaveNoTest:
    """Lifted whole from ValidationSuite; see model_validation.py."""

    @staticmethod
    def _unmeasured_only_here(channels: tuple[str, ...]) -> bool:
        try:
            from core.organism.claim_liveness import unmeasured_only_here

            return unmeasured_only_here(channels)
        except (ImportError, AttributeError, TypeError, ValueError):
            return False

    def unsupported_claims(self) -> list[dict[str, Any]]:
        """Claims whose test last failed, could not run, or measured nothing.

        This is the machine-checked version of CLAIMS_NOT_SUPPORTED.md.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .model_validation import (
            Outcome,
        )

        out: list[dict[str, Any]] = []
        with self._lock:
            claims = list(self._claims.values())
            last = dict(self._last)
        for claim in claims:
            # A claim can lose its footing two ways: its test stops passing,
            # or the live measurement behind it stops arriving. The second
            # was invisible until claims could bind to telemetry, and it is
            # the one that produced "a claim that outlived the code".
            resolved, liveness_note = claim.effective_evidence()
            if liveness_note and resolved is not claim.evidence:
                out.append(
                    {
                        **claim.to_dict(),
                        "reason": liveness_note,
                        # A claim whose channels are silent because THIS
                        # process runs no publisher for them has no evidence
                        # here; it has not decayed. The distinction is the
                        # difference between a live organ that stopped
                        # reporting and a test process that never asked.
                        "outcome": (
                            str(Outcome.NOT_MEASURED)
                            if self._unmeasured_only_here(claim.live_channels)
                            else "evidence_decayed"
                        ),
                    }
                )
                continue
            relevant = [r for (test, _model), r in last.items() if test == claim.test]
            if not relevant:
                # A suite that has not run cannot have run THIS test, and every
                # claim in it reads "never run" at once. The desktop defers the
                # empirical run on purpose — several tests monopolize the
                # interpreter for tens of seconds and belong to an explicit
                # validation process — and reporting the consequence of that
                # decision as a hundred structural errors raised an emergency
                # incident on every verifier pass, tainted the runtime, and
                # drove the resilience layer to full depletion. LIVE,
                # 2026-09-10: "100 invariants: 105 error(s)", 35 times.
                #
                # Unrun is unevidenced. A test the suite DID run and that
                # produced nothing for this claim is a different fact and
                # keeps its error.
                out.append(
                    {
                        **claim.to_dict(),
                        "reason": (
                            "the validation suite has not run in this process"
                            if self.runs == 0
                            else "never run"
                        ),
                        **(
                            {"outcome": str(Outcome.NOT_MEASURED)}
                            if self.runs == 0
                            else {}
                        ),
                    }
                )
                continue
            if any(r.score.outcome in self._UNSUPPORTING for r in relevant):
                worst = next(
                    r for r in relevant if r.score.outcome in self._UNSUPPORTING
                )
                out.append(
                    {
                        **claim.to_dict(),
                        "reason": worst.score.interpretation,
                        "outcome": str(worst.score.outcome),
                    }
                )
        return out

    def report(self) -> dict[str, Any]:
        with self._lock:
            tests = [t.to_dict() for t in self._tests.values()]
            claims = [c.to_dict() for c in self._claims.values()]
            last = {f"{t}/{m}": r.to_dict() for (t, m), r in self._last.items()}
        return {
            "tests": tests,
            "claims": claims,
            "models": sorted(self._models),
            "runs": self.runs,
            "last_results": last,
            "unsupported_claims": self.unsupported_claims(),
            "tests_without_claims": sorted(
                {t["name"] for t in tests} - {c["test"] for c in claims}
            ),
        }
