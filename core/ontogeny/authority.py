"""L7 — earned authority: one control point at a time, and revocable in one call.

Nothing here promotes a head because it is clever. A head earns a decision by
outscoring the rules that currently make it, on outcomes, with the comparison
done the conservative way round: the challenger is judged by the pessimistic
end of its confidence interval and the incumbent by the optimistic end of its
own. Ties go to the incumbent. Sixty years of hand-written thresholds are a
strong prior, and a learner that cannot clear them by a real margin has not
demonstrated anything worth the risk.

The ladder has four rungs, and the middle two matter as much as the last:

  OBSERVE    Recording only. Not enough graded experience to fit anything.
  SHADOW     The head predicts on every episode and is scored, but never acts
             except on the reserved probe slice — which exists so its
             preferred actions get tried and scored at all.
  ADVISORY   The head's prediction becomes a real, bounded input to the
             incumbent's decision. It is causal here — this is not a waiting
             room — but the incumbent still decides and can override it.
  AUTHORITY  The head decides. The incumbent keeps a permanent reserved slice
             so the comparison that justified this can be repeated forever.

Revocation is not a ceremony. Any of: calibration drift past the grant-time
baseline, the counterfactual slice showing the incumbent doing better, an open
invariant violation, or a human saying so. Revocation returns to ADVISORY
rather than to silence — a head that has stopped being trustworthy enough to
decide is usually still worth hearing.

Every transition is receipted with the evidence that justified it, because a
learned controller that cannot show its licence has no business holding one.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.ontogeny.calibration import CalibrationMonitor, wilson
from core.ontogeny.experience import Episode, OutcomeKind
from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Ontogeny.Authority")

AUTHORITY_SCHEMA = "aura.ontogeny.authority.v1"

#: Graded episodes required from *each* side before a comparison is allowed to
#: mean anything.
MIN_TRIALS = 40

#: Calibration ceiling for advisory and above. A head whose stated confidence
#: is off by more than this is not yet honest enough to be listened to.
MAX_ECE_FOR_ADVISORY = 0.15

#: Held-out margin the challenger must clear. Wilson bounds already make this
#: conservative; the margin makes it decisive.
MIN_MARGIN = 0.02


class AuthorityStage(StrEnum):
    OBSERVE = "observe"
    SHADOW = "shadow"
    ADVISORY = "advisory"
    AUTHORITY = "authority"

    @property
    def rank(self) -> int:
        return {"observe": 0, "shadow": 1, "advisory": 2, "authority": 3}[str(self)]

    @property
    def decides(self) -> bool:
        return self is AuthorityStage.AUTHORITY

    @property
    def advises(self) -> bool:
        return self.rank >= AuthorityStage.ADVISORY.rank


@dataclass(frozen=True)
class Comparison:
    """The held-out head-to-head that a promotion or revocation rests on."""

    control_point: str
    challenger_successes: int
    challenger_total: int
    incumbent_successes: int
    incumbent_total: int

    @property
    def challenger_rate(self) -> float:
        return self.challenger_successes / self.challenger_total if self.challenger_total else 0.0

    @property
    def incumbent_rate(self) -> float:
        return self.incumbent_successes / self.incumbent_total if self.incumbent_total else 0.0

    @property
    def challenger_lower(self) -> float:
        return wilson(self.challenger_successes, self.challenger_total, upper=False)

    @property
    def incumbent_upper(self) -> float:
        return wilson(self.incumbent_successes, self.incumbent_total, upper=True)

    @property
    def sufficient(self) -> bool:
        return self.challenger_total >= MIN_TRIALS and self.incumbent_total >= MIN_TRIALS

    @property
    def challenger_wins(self) -> bool:
        return self.sufficient and self.challenger_lower > self.incumbent_upper + MIN_MARGIN

    @property
    def incumbent_wins(self) -> bool:
        """The revocation test — deliberately not the mirror image.

        Promotion needs separation; revocation needs only that the incumbent's
        pessimistic bound has caught the challenger's optimistic one. Trust is
        harder to gain than to lose, which is the correct asymmetry when the
        thing being trusted decides on Aura's behalf.
        """
        if not self.sufficient:
            return False
        incumbent_lower = wilson(self.incumbent_successes, self.incumbent_total, upper=False)
        challenger_upper = wilson(self.challenger_successes, self.challenger_total, upper=True)
        return incumbent_lower > challenger_upper

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "challenger": {
                "successes": self.challenger_successes,
                "total": self.challenger_total,
                "rate": round(self.challenger_rate, 4),
                "lower_bound": round(self.challenger_lower, 4),
            },
            "incumbent": {
                "successes": self.incumbent_successes,
                "total": self.incumbent_total,
                "rate": round(self.incumbent_rate, 4),
                "upper_bound": round(self.incumbent_upper, 4),
            },
            "sufficient": self.sufficient,
            "challenger_wins": self.challenger_wins,
            "incumbent_wins": self.incumbent_wins,
            "min_trials": MIN_TRIALS,
            "min_margin": MIN_MARGIN,
        }


@dataclass
class Grant:
    """A control point's current standing, and how it got there."""

    control_point: str
    stage: AuthorityStage = AuthorityStage.OBSERVE
    since: float = field(default_factory=time.time)
    reason: str = "initial"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: Bumped every time the stage is lowered. A head that has been revoked
    #: twice is held to the same bar, but the history travels with it.
    revocations: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "stage": str(self.stage),
            "since": self.since,
            "age_s": round(time.time() - self.since, 1),
            "reason": self.reason,
            "revocations": self.revocations,
            "evidence": dict(self.evidence),
        }


class AuthorityLedger:
    """Who may decide what, with the receipts."""

    def __init__(self, path: Path | None = None, *, calibration: CalibrationMonitor | None = None) -> None:
        self._path = path or _default_authority_path()
        self._lock = checked_lock("ontogeny.authority", rank=LockRank.LEAF, reentrant=True)
        self._grants: dict[str, Grant] = {}
        self._calibration = calibration or CalibrationMonitor()
        self._frozen = False
        self._load()

    def attach_calibration(self, monitor: CalibrationMonitor) -> None:
        """Share the organ's calibration monitor.

        The ledger judges honesty from the same measurements the trainer
        records. Two monitors would mean the ledger reasoning about a head
        whose scores it has never seen — which is how a promotion gate ends up
        gating on nothing.
        """
        self._calibration = monitor

    # ── queries ──────────────────────────────────────────────────────────

    def stage(self, control_point: str) -> AuthorityStage:
        with self._lock:
            if self._frozen:
                return AuthorityStage.SHADOW
            grant = self._grants.get(control_point)
            return grant.stage if grant else AuthorityStage.OBSERVE

    def has_authority(self, control_point: str) -> bool:
        return self.stage(control_point).decides

    def advises(self, control_point: str) -> bool:
        return self.stage(control_point).advises

    def grant_of(self, control_point: str) -> Grant | None:
        with self._lock:
            return self._grants.get(control_point)

    # ── transitions ──────────────────────────────────────────────────────

    def set_stage(
        self, control_point: str, stage: AuthorityStage, *, reason: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> Grant:
        with self._lock:
            previous = self._grants.get(control_point)
            revocations = previous.revocations if previous else 0
            if previous and stage.rank < previous.stage.rank:
                revocations += 1
            grant = Grant(
                control_point=control_point, stage=stage, reason=reason,
                evidence=dict(evidence or {}), revocations=revocations,
            )
            self._grants[control_point] = grant
            previous_stage = str(previous.stage) if previous else None
            snapshot = self._snapshot_locked()
        # Persistence happens outside the lock on purpose: writing a receipt
        # fsyncs, and a blocking syscall under a held lock is precisely how a
        # runtime freezes. The in-memory transition is already committed.
        self._receipt(
            "stage_change",
            control_point=control_point,
            previous=previous_stage,
            stage=str(stage),
            reason=reason,
            evidence=dict(evidence or {}),
        )
        self._save(snapshot)
        logger.info(
            "ontogeny: %s -> %s (%s)", control_point, stage, reason,
        )
        return grant

    def evaluate(
        self,
        control_point: str,
        episodes: Iterable[Episode],
        *,
        head_ready: bool,
        holdout_accuracy: float | None = None,
    ) -> dict[str, Any]:
        """Consider a promotion or a revocation on current evidence.

        This is the only path that raises a stage. It is called from the
        trainer after a fit, never from the decision path — a promotion should
        be a deliberate act with evidence attached, not something that happens
        mid-conversation because a counter crossed a line.
        """
        comparison = compare(control_point, episodes)
        current = self.stage(control_point)
        calibration = self._calibration.report(control_point)
        ece = calibration.ece if calibration else None
        drifted, drift_reason = self._calibration.drifted(control_point)

        verdict: dict[str, Any] = {
            "control_point": control_point,
            "stage": str(current),
            "comparison": comparison.as_dict(),
            "calibration": calibration.as_dict() if calibration else None,
            "head_ready": head_ready,
            "holdout_accuracy": holdout_accuracy,
            "action": "hold",
        }

        if current is AuthorityStage.AUTHORITY:
            if drifted:
                self.set_stage(
                    control_point, AuthorityStage.ADVISORY,
                    reason=f"calibration_drift: {drift_reason}", evidence=verdict,
                )
                verdict["action"] = "revoked_calibration"
                return verdict
            if comparison.incumbent_wins:
                self.set_stage(
                    control_point, AuthorityStage.ADVISORY,
                    reason="counterfactual slice shows the incumbent ahead", evidence=verdict,
                )
                verdict["action"] = "revoked_outcomes"
                return verdict
            return verdict

        if not head_ready:
            if current is AuthorityStage.OBSERVE:
                return verdict
            self.set_stage(
                control_point, AuthorityStage.OBSERVE,
                reason="head no longer has enough graded evidence to be fitted",
                evidence=verdict,
            )
            verdict["action"] = "demoted_unready"
            return verdict

        if current is AuthorityStage.OBSERVE:
            self.set_stage(
                control_point, AuthorityStage.SHADOW,
                reason="head fitted; predictions now recorded and scored", evidence=verdict,
            )
            verdict["action"] = "promoted_shadow"
            return verdict

        if current is AuthorityStage.SHADOW:
            if ece is not None and ece <= MAX_ECE_FOR_ADVISORY and (calibration and calibration.samples >= MIN_TRIALS):
                self.set_stage(
                    control_point, AuthorityStage.ADVISORY,
                    reason=f"calibrated within {MAX_ECE_FOR_ADVISORY} over {calibration.samples} scored predictions",
                    evidence=verdict,
                )
                verdict["action"] = "promoted_advisory"
            return verdict

        if current is AuthorityStage.ADVISORY and comparison.challenger_wins:
            if ece is not None and ece > MAX_ECE_FOR_ADVISORY:
                verdict["action"] = "held_calibration"
                return verdict
            self._calibration.set_baseline(control_point)
            self.set_stage(
                control_point, AuthorityStage.AUTHORITY,
                reason=(
                    f"held-out: challenger {comparison.challenger_lower:.3f} (lower) > "
                    f"incumbent {comparison.incumbent_upper:.3f} (upper)"
                ),
                evidence=verdict,
            )
            verdict["action"] = "promoted_authority"
        return verdict

    def revoke(self, control_point: str, reason: str) -> Grant:
        """Hand a decision back. Always available, never needs a quorum."""
        return self.set_stage(
            control_point, AuthorityStage.ADVISORY, reason=f"revoked: {reason}"
        )

    def freeze(self, reason: str) -> None:
        """Global kill switch: every head drops to shadow immediately.

        For the case where something is wrong and nobody yet knows which
        control point owns it. Reversible, and does not erase any grant — the
        ladder is remembered and resumes when the freeze lifts.
        """
        with self._lock:
            self._frozen = True
        self._receipt("freeze", reason=reason)
        self._save()
        logger.warning("ontogeny: all learned authority frozen (%s)", reason)

    def unfreeze(self, reason: str) -> None:
        with self._lock:
            self._frozen = False
        self._receipt("unfreeze", reason=reason)
        self._save()
        logger.info("ontogeny: learned authority unfrozen (%s)", reason)

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    # ── persistence ──────────────────────────────────────────────────────

    def _receipt(self, kind: str, **payload: Any) -> None:
        record = {"schema": AUTHORITY_SCHEMA, "kind": kind, "at": time.time(), **payload}
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "ontogeny_authority", domain="state_mutation", receipt_prefix="ontogeny-authority"
            ):
                gateway.ensure_directory(self._path.parent, source="ontogeny_authority")
                gateway.append_text(
                    self._path.parent / "authority_receipts.jsonl",
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n",
                    source="ontogeny_authority",
                )
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation(
                "ontogeny_authority", exc, severity="warning",
                action="authority receipt not written; the transition still applied",
            )

    def _snapshot_locked(self) -> dict[str, Any]:
        """Serialisable state, taken under the lock and written outside it."""
        return {
            "schema": AUTHORITY_SCHEMA,
            "saved_at": time.time(),
            "frozen": self._frozen,
            "grants": {cp: g.as_dict() for cp, g in self._grants.items()},
        }

    def _save(self, payload: dict[str, Any] | None = None) -> None:
        if payload is None:
            with self._lock:
                payload = self._snapshot_locked()
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "ontogeny_authority", domain="state_mutation", receipt_prefix="ontogeny-authority"
            ):
                gateway.ensure_directory(self._path.parent, source="ontogeny_authority")
                gateway.write_text(
                    self._path,
                    json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True),
                    source="ontogeny_authority",
                )
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation(
                "ontogeny_authority", exc, severity="warning",
                action="authority state not persisted; in-memory grants stand",
            )

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._frozen = bool(payload.get("frozen", False))
            for cp, raw in (payload.get("grants") or {}).items():
                self._grants[cp] = Grant(
                    control_point=cp,
                    stage=AuthorityStage(raw.get("stage", "observe")),
                    since=float(raw.get("since", time.time())),
                    reason=str(raw.get("reason", "restored")),
                    evidence=dict(raw.get("evidence") or {}),
                    revocations=int(raw.get("revocations", 0)),
                )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record_degradation(
                "ontogeny_authority", exc, severity="warning",
                action="authority state unreadable; every control point starts at observe",
            )

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": AUTHORITY_SCHEMA,
                "frozen": self._frozen,
                "grants": {cp: g.as_dict() for cp, g in self._grants.items()},
                "thresholds": {
                    "min_trials": MIN_TRIALS,
                    "min_margin": MIN_MARGIN,
                    "max_ece_for_advisory": MAX_ECE_FOR_ADVISORY,
                },
            }


def compare(control_point: str, episodes: Iterable[Episode]) -> Comparison:
    """Score challenger against incumbent on episodes each of them actually decided.

    Only graded outcomes count. Unobserved episodes are excluded on both sides
    — they are not evidence for anyone, and letting them fall to whichever
    decider happened to own them would silently bias the comparison toward the
    one whose outcomes are easier to observe.
    """
    c_success = c_total = i_success = i_total = 0
    for episode in episodes:
        outcome = episode.outcome
        if outcome is None or not outcome.kind.is_evidence:
            continue
        if episode.decider.startswith("explore:"):
            # Randomly-actioned episodes belong to neither policy. They are
            # the corpus's causal backbone and would poison a head-to-head
            # score: whichever side inherited them would be judged partly on
            # decisions it did not make.
            continue
        weight = max(1, int(episode.repeat_count))
        won = outcome.kind is OutcomeKind.SUCCESS
        if episode.decider.startswith("ontogeny"):
            c_total += weight
            c_success += weight if won else 0
        else:
            i_total += weight
            i_success += weight if won else 0
    return Comparison(
        control_point=control_point,
        challenger_successes=c_success,
        challenger_total=c_total,
        incumbent_successes=i_success,
        incumbent_total=i_total,
    )


def _default_authority_path() -> Path:
    import os

    override = os.environ.get("AURA_ONTOGENY_AUTHORITY")
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "ontogeny" / "authority.json"
    except (ImportError, AttributeError, RuntimeError, OSError):
        return state_root() / "data" / "ontogeny" / "authority.json"


_ledger: AuthorityLedger | None = None
_ledger_lock = checked_lock("core.ontogeny.authority._ledger_lock")


def get_authority_ledger() -> AuthorityLedger:
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = AuthorityLedger()
    return _ledger


def reset_authority_for_test(ledger: AuthorityLedger | None = None) -> None:
    global _ledger
    with _ledger_lock:
        _ledger = ledger


__all__ = [
    "AUTHORITY_SCHEMA",
    "MAX_ECE_FOR_ADVISORY",
    "MIN_MARGIN",
    "MIN_TRIALS",
    "AuthorityLedger",
    "AuthorityStage",
    "Comparison",
    "Grant",
    "compare",
    "get_authority_ledger",
    "reset_authority_for_test",
]
