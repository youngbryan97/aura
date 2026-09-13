"""Structural invariants for the ontogenetic organ.

These sit next to the thing they protect, and each one exists because there is
a specific way this organ could quietly become dishonest. A learned controller
that decides on Aura's behalf has to be checkable by something other than the
learned controller.

The four that matter:

1. **A live corpus contains only lived experience.** If test or benchmark rows
   ever reach the live store, every number above them is fiction.
2. **Authority implies observation.** A head holding a decision whose outcomes
   have stopped being observed is not learning; it is accumulating confident
   ignorance while continuing to act. This is the failure that looks healthiest
   from the outside.
3. **Authority implies exploration.** The counterfactual slice is what makes a
   promotion falsifiable. If it stops running, the comparison that justified
   the grant can never be repeated, and the grant becomes permanent by default.
4. **Unobserved outcomes never carry a utility.** The moment an unobserved
   episode acquires a number, it becomes indistinguishable from a measurement,
   and the distinction this whole layer rests on is gone.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.ontogeny.experience import OutcomeKind
from core.runtime.sqlite_support import connecting
from core.verify.invariants import Severity, Violation, invariant

#: Below this observation rate a deciding head is no longer learning from what
#: it does. Chosen low on purpose: the check is for collapse, not for a dip.
_MIN_OBSERVATION_RATE = 0.05

#: Episodes a control point must have swept before its observation rate means
#: anything at all.
_MIN_SWEPT = 50


def _organ() -> object | None:
    try:
        from core.ontogeny.service import get_ontogeny

        return get_ontogeny()
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        return None


@invariant(
    "ontogeny.live_corpus_is_lived_experience",
    scope="ontogeny",
    owner="core/ontogeny/experience.py",
    description="the live experience store contains only Provenance.LIVE episodes",
)
def _corpus_is_lived() -> Iterator[Violation]:
    core = _organ()
    if core is None:
        return
    spine = getattr(core, "_spine", None)
    if spine is None or spine.store_kind != "live":
        return
    try:
        # `connecting`, not the connection's own context manager. A sqlite3
        # connection used as a context manager commits or rolls back and does
        # NOT close, so every invariant sweep left an open handle on the
        # experience database -- which the hermetic guard sees as three leaked
        # files, the db and its write-ahead log and shared-memory index. Every
        # other call site in this package already wraps it.
        with connecting(spine._connect()) as conn:  # noqa: SLF001 — the invariant guards this file
            rows = conn.execute(
                "SELECT provenance, COUNT(*) FROM episodes WHERE provenance != 'live' GROUP BY 1"
            ).fetchall()
    except Exception:  # noqa: BLE001 — a check that raises counts as a violation
        raise
    for provenance, count in rows:
        yield Violation(
            subject=f"experience_spine/{provenance}",
            message=f"{count} non-live episodes in the live corpus",
            remedy="route test and benchmark writes to a sandbox store (AURA_ONTOGENY_DB)",
        )


@invariant(
    "ontogeny.authority_implies_observation",
    scope="ontogeny",
    owner="core/ontogeny/authority.py",
    description="a head that decides must still be seeing the outcomes of its decisions",
)
def _authority_implies_observation() -> Iterator[Violation]:
    core = _organ()
    if core is None:
        return
    report = core.authority_observation_report(min_closed=_MIN_SWEPT)
    for control_point, stats in report.get("control_points", {}).items():
        if not stats.get("eligible"):
            continue
        swept = int(stats.get("closed") or 0)
        rate = float(stats.get("observation_rate") or 0.0)
        if rate <= _MIN_OBSERVATION_RATE:
            yield Violation(
                subject=control_point,
                message=(
                    f"holds authority but only {rate:.1%} of {swept} swept episodes "
                    "were ever observed"
                ),
                remedy="register a resolver that can actually grade this control point, "
                       "or revoke the grant until one exists",
            )


@invariant(
    "ontogeny.authority_implies_exploration",
    scope="ontogeny",
    owner="core/ontogeny/reservation.py",
    description="a promoted head must keep giving the incumbent a reserved slice",
)
def _authority_implies_exploration() -> Iterator[Violation]:
    core = _organ()
    if core is None:
        return
    report = core.reservation_report()
    if float(report.get("counterfactual_rate") or 0.0) <= 0.0:
        for control_point in core.control_points():
            if core.authority.has_authority(control_point):
                yield Violation(
                    subject=control_point,
                    message="holds authority with no counterfactual slice reserved",
                    remedy="the incumbent must keep deciding a fixed share of episodes, "
                           "or the promotion can never be re-tested",
                )


@invariant(
    "ontogeny.unobserved_carries_no_utility",
    scope="ontogeny",
    owner="core/ontogeny/experience.py",
    severity=Severity.ERROR,
    description="an unobserved outcome never carries a magnitude",
)
def _unobserved_has_no_utility() -> Iterator[Violation]:
    core = _organ()
    if core is None:
        return
    spine = getattr(core, "_spine", None)
    if spine is None:
        return
    try:
        with connecting(spine._connect()) as conn:  # noqa: SLF001
            count = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE outcome_kind = ? "
                "AND outcome_utility IS NOT NULL",
                (str(OutcomeKind.UNOBSERVED),),
            ).fetchone()[0]
    except Exception:  # noqa: BLE001
        raise
    if count:
        yield Violation(
            subject="experience_spine",
            message=f"{count} unobserved episodes carry a utility",
            remedy="an unmeasured outcome has no magnitude; drop the value",
        )


@invariant(
    "ontogeny.verbalization_preserves_accepted_claims",
    scope="ontogeny",
    owner="core/ontogeny/conclusion.py",
    description="speaking a conclusion may not assert more than the conclusion accepted",
)
def _verbalization_preserves_claims() -> Iterator[Violation]:
    """The quiet failure: a hedge dropped between deciding and speaking.

    Omissions are reported by the checker but tolerated here — a response that
    leaves something out is incomplete, not dishonest. An *overstatement* is
    different in kind: it says something she never accepted, in her own voice,
    and nothing downstream can tell that it was never a finding.
    """
    try:
        from core.ontogeny.conclusion import get_verbalization_ledger

        overstatements = get_verbalization_ledger().recent_overstatements()
    except (ImportError, RuntimeError, ValueError, TypeError):
        return
    for entry in overstatements:
        offending = [v for v in entry["violations"] if v["kind"] == "overstated"]
        yield Violation(
            subject=f"conclusion/{entry['conclusion_id']}",
            message=(
                f"{len(offending)} claim(s) stated more firmly than accepted: "
                + "; ".join(v["subject"] for v in offending[:2])
            ),
            remedy="keep the hedge on tentative and speculative claims, or raise the "
                   "claim's confidence in the conclusion where the evidence supports it",
        )


def install() -> list[str]:
    """Import-time registration is enough; this names them for the boot report."""
    return [
        "ontogeny.live_corpus_is_lived_experience",
        "ontogeny.authority_implies_observation",
        "ontogeny.authority_implies_exploration",
        "ontogeny.unobserved_carries_no_utility",
        "ontogeny.verbalization_preserves_accepted_claims",
    ]


__all__ = ["install"]
