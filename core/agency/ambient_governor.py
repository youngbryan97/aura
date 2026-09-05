"""core/agency/ambient_governor.py — what an unprompted mind is allowed to do.

Clean-room adoption of EloPhanto's ambient intervention stack (PolyForm
Noncommercial; mechanism reimplemented from its design, no code taken).

Aura's ambient companion mode is queued work: speak when the window is
closed, understand the screen continuously, act on her own initiative. The
hard part of that was never generating a good interruption. It is that an
always-on companion which can speak whenever it judges the moment right is
intolerable to live with, and no amount of good judgement fixes it, because
the judgement is the thing under load.

Three structures carry the weight, and each is a constraint on Aura rather
than a capability for her.

**A strength ladder that cannot be climbed silently.**
``OBSERVE → NUDGE → ACT → ESCALATE``. The top two rungs are members of
:data:`_AUTO_APPROVE_FORBIDDEN` — a proposal at ACT or ESCALATE cannot be
auto-approved by any code path, including a future one that has forgotten
this file exists. That is deliberately a data structure and not a policy
check: policy checks get an ``if`` added to them.

**An interruption budget, with no default.**
The silence cap is how many unprompted proposals a calendar day may carry.
It is NOT given a default value. An ambient mind whose budget nobody has
set does not get a reasonable-sounding number picked for it — it stays
silent, because "how often may this thing interrupt me" is the owner's
answer and a guess in that slot is a guess about someone's attention. An
unconfigured governor refuses everything above OBSERVE and says why.

The one exemption is a proposal whose *purpose* is protecting silence
(asking to hold quiet hours). Those do not spend budget, because charging
them would make the mechanism eat itself.

**Predictions that can be wrong.**
Every ambient claim carries ``p_hat`` and a ``resolve_by`` deadline, and is
graded against what actually happened or marked UNKNOWN when the deadline
passes with no evidence. Once a claim type has enough graded outcomes, its
own history blends into future estimates. Without this an ambient mind's
confidence is decoration: it never finds out it was wrong, so it never
gets better and neither does anyone's trust in it.

Silence is a success condition here, not an absence of one. A governor
that let everything through would be indistinguishable from no governor,
and the metric worth watching is how much it declined.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock

__all__ = [
    "Strength",
    "ProposalStatus",
    "Outcome",
    "Proposal",
    "Prediction",
    "AmbientGovernor",
    "get_ambient_governor",
]


class Strength(StrEnum):
    """How far into the person's attention a proposal reaches."""

    #: Recorded, never surfaced. Always permitted.
    OBSERVE = "observe"
    #: A surfaced suggestion. Spends interruption budget.
    NUDGE = "nudge"
    #: Takes an action in the world. Never auto-approved.
    ACT = "act"
    #: Interrupts regardless of context. Never auto-approved.
    ESCALATE = "escalate"


#: Structurally, not by policy. A future code path that wants to
#: auto-approve an ACT has to edit this line and explain itself in review.
_AUTO_APPROVE_FORBIDDEN = frozenset({Strength.ACT, Strength.ESCALATE})

#: OBSERVE never reaches the person, so it never spends budget.
_FREE_STRENGTHS = frozenset({Strength.OBSERVE})


class ProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTED = "executed"
    EXPIRED = "expired"
    WITHHELD = "withheld"


class Outcome(StrEnum):
    """How a prediction turned out."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    #: The deadline passed with no evidence either way. Not a failure, and
    #: crucially not a success — scoring it either way would let an
    #: unobservable prediction improve the calibration record.
    UNKNOWN = "unknown"


@dataclass
class Proposal:
    """One thing the ambient mind wants to do, and what became of it."""

    proposal_id: str
    strength: Strength
    summary: str
    status: ProposalStatus = ProposalStatus.PROPOSED
    reason: str = ""
    silence_exempt: bool = False
    prediction_id: str | None = None
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "strength": str(self.strength),
            "summary": self.summary,
            "status": str(self.status),
            "reason": self.reason,
            "silence_exempt": self.silence_exempt,
            "prediction_id": self.prediction_id,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
        }


@dataclass
class Prediction:
    """A falsifiable ambient claim with a deadline."""

    prediction_id: str
    claim_type: str
    claim: str
    p_hat: float
    resolve_by: float
    created_at: float = field(default_factory=time.time)
    outcome: Outcome | None = None
    resolved_at: float | None = None

    @property
    def is_resolved(self) -> bool:
        return self.outcome is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "claim_type": self.claim_type,
            "claim": self.claim,
            "p_hat": round(self.p_hat, 4),
            "resolve_by": self.resolve_by,
            "created_at": self.created_at,
            "outcome": None if self.outcome is None else str(self.outcome),
            "resolved_at": self.resolved_at,
        }


#: Graded outcomes needed before a claim type's own history is allowed to
#: move its estimates. Below this the sample is small enough that blending
#: would be fitting noise and calling it calibration.
_MIN_GRADED_FOR_HISTORY = 5

#: How much weight the measured base rate takes once there is enough of it.
#: The rule keeps most of its say — history corrects a rule, it does not
#: replace one.
_HISTORY_WEIGHT = 0.4

_MAX_PROPOSALS = 512
_MAX_PREDICTIONS = 512


class AmbientGovernor:
    """Decides what an unprompted mind may surface, and remembers the record."""

    def __init__(self, *, daily_silence_cap: int | None = None) -> None:
        self._lock = checked_lock("ambient_governor", rank=LockRank.LEAF)
        # None means UNCONFIGURED, which is not the same as zero. Zero is an
        # owner saying "never interrupt me"; None is nobody having said
        # anything, and the governor must not invent an answer.
        self._daily_cap: int | None = daily_silence_cap
        self._proposals: list[Proposal] = []
        self._predictions: dict[str, Prediction] = {}
        self._withheld = 0

    # ------------------------------------------------------------ configuring

    def configure(self, *, daily_silence_cap: int) -> None:
        """Set the interruption budget. Required before anything may surface."""
        cap = int(daily_silence_cap)
        if cap < 0:
            raise ValueError("daily_silence_cap cannot be negative")
        with self._lock:
            self._daily_cap = cap

    @property
    def is_configured(self) -> bool:
        return self._daily_cap is not None

    # -------------------------------------------------------------- budgeting

    @staticmethod
    def _day_key(at: float) -> str:
        """Local calendar day. Budgets are lived in local time, not UTC."""
        return datetime.fromtimestamp(at).strftime("%Y-%m-%d")

    def _spent_today_locked(self, *, now: float) -> int:
        today = self._day_key(now)
        return sum(
            1
            for p in self._proposals
            if not p.silence_exempt
            and p.strength not in _FREE_STRENGTHS
            and p.status is not ProposalStatus.WITHHELD
            and self._day_key(p.created_at) == today
        )

    def remaining_budget(self, *, now: float | None = None) -> int | None:
        """Interruptions left today. ``None`` when unconfigured."""
        moment = time.time() if now is None else now
        with self._lock:
            if self._daily_cap is None:
                return None
            return max(0, self._daily_cap - self._spent_today_locked(now=moment))

    # -------------------------------------------------------------- proposing

    def propose(
        self,
        *,
        strength: Strength | str,
        summary: str,
        reason: str = "",
        silence_exempt: bool = False,
        prediction_id: str | None = None,
        now: float | None = None,
    ) -> Proposal:
        """Offer an ambient action. Returns the proposal with its verdict.

        A refused proposal is returned WITHHELD rather than raising or
        vanishing: the ambient mind should be able to see that it wanted to
        speak and did not get to, because that record is the only evidence
        of what the budget actually cost.
        """
        moment = time.time() if now is None else now
        try:
            level = Strength(strength)
        except ValueError:
            level = Strength.OBSERVE
            reason = f"unknown strength {strength!r}; recorded as observation only"

        proposal = Proposal(
            proposal_id=uuid.uuid4().hex,
            strength=level,
            summary=str(summary or ""),
            reason=reason,
            silence_exempt=bool(silence_exempt),
            prediction_id=prediction_id,
            created_at=moment,
        )

        with self._lock:
            verdict, why = self._admit_locked(proposal, now=moment)
            proposal.status = verdict
            if why:
                proposal.reason = why
            if verdict is ProposalStatus.WITHHELD:
                self._withheld += 1
            self._proposals.append(proposal)
            while len(self._proposals) > _MAX_PROPOSALS:
                self._proposals.pop(0)
        return proposal

    def _admit_locked(
        self, proposal: Proposal, *, now: float
    ) -> tuple[ProposalStatus, str]:
        # Observations never reach anyone, so they are always allowed and
        # never spend anything.
        if proposal.strength in _FREE_STRENGTHS:
            return ProposalStatus.PROPOSED, ""

        if self._daily_cap is None:
            return (
                ProposalStatus.WITHHELD,
                "no interruption budget configured; an ambient mind with no "
                "agreed cap stays silent rather than choosing one for the owner",
            )

        # A proposal that exists to protect silence does not spend silence.
        if not proposal.silence_exempt:
            spent = self._spent_today_locked(now=now)
            if spent >= self._daily_cap:
                return (
                    ProposalStatus.WITHHELD,
                    f"daily interruption budget spent ({spent}/{self._daily_cap})",
                )
        return ProposalStatus.PROPOSED, ""

    # --------------------------------------------------------------- deciding

    def decide(self, proposal_id: str, *, approved: bool, note: str = "") -> Proposal | None:
        """Record the owner's decision on a proposal."""
        with self._lock:
            proposal = next(
                (p for p in self._proposals if p.proposal_id == proposal_id), None
            )
            if proposal is None:
                return None
            if proposal.status not in (ProposalStatus.PROPOSED,):
                return proposal
            proposal.status = (
                ProposalStatus.APPROVED if approved else ProposalStatus.DENIED
            )
            proposal.decided_at = time.time()
            if note:
                proposal.reason = note
            return proposal

    def may_auto_approve(self, strength: Strength | str) -> bool:
        """Whether this strength may ever be approved without a person.

        The answer for ACT and ESCALATE is no, permanently. Exposed as a
        query so callers can check rather than assume, and so the
        prohibition is testable from outside.
        """
        try:
            level = Strength(strength)
        except ValueError:
            return False
        return level not in _AUTO_APPROVE_FORBIDDEN

    def auto_approve(self, proposal_id: str) -> Proposal | None:
        """Approve without a person. Refuses at ACT and above."""
        with self._lock:
            proposal = next(
                (p for p in self._proposals if p.proposal_id == proposal_id), None
            )
            if proposal is None:
                return None
            if proposal.strength in _AUTO_APPROVE_FORBIDDEN:
                proposal.status = ProposalStatus.WITHHELD
                proposal.reason = (
                    f"{proposal.strength} can never be auto-approved; it requires "
                    "an explicit decision from the owner"
                )
                self._withheld += 1
                return proposal
            proposal.status = ProposalStatus.APPROVED
            proposal.decided_at = time.time()
            return proposal

    # ------------------------------------------------------------ predictions

    def predict(
        self,
        *,
        claim_type: str,
        claim: str,
        p_hat: float,
        resolve_by: float,
        now: float | None = None,
    ) -> Prediction:
        """Register a falsifiable ambient claim.

        ``p_hat`` is blended with the claim type's measured base rate once
        enough outcomes have been graded — the rule keeps most of its say,
        because history corrects a rule rather than replacing it.
        """
        moment = time.time() if now is None else now
        blended = self._blend(claim_type, float(p_hat))
        prediction = Prediction(
            prediction_id=uuid.uuid4().hex,
            claim_type=str(claim_type or "unknown"),
            claim=str(claim or ""),
            p_hat=max(0.0, min(1.0, blended)),
            resolve_by=float(resolve_by),
            created_at=moment,
        )
        with self._lock:
            self._predictions[prediction.prediction_id] = prediction
            while len(self._predictions) > _MAX_PREDICTIONS:
                oldest = min(self._predictions.values(), key=lambda p: p.created_at)
                self._predictions.pop(oldest.prediction_id, None)
        return prediction

    def _blend(self, claim_type: str, p_hat: float) -> float:
        graded = [
            p
            for p in self._predictions.values()
            if p.claim_type == claim_type
            and p.outcome in (Outcome.CORRECT, Outcome.INCORRECT)
        ]
        if len(graded) < _MIN_GRADED_FOR_HISTORY:
            return p_hat
        hits = sum(1 for p in graded if p.outcome is Outcome.CORRECT)
        base_rate = hits / len(graded)
        return p_hat * (1.0 - _HISTORY_WEIGHT) + base_rate * _HISTORY_WEIGHT

    def resolve(self, prediction_id: str, *, correct: bool) -> Prediction | None:
        """Grade a prediction against what actually happened."""
        with self._lock:
            prediction = self._predictions.get(prediction_id)
            if prediction is None or prediction.is_resolved:
                return prediction
            prediction.outcome = Outcome.CORRECT if correct else Outcome.INCORRECT
            prediction.resolved_at = time.time()
            return prediction

    def expire_due(self, *, now: float | None = None) -> list[Prediction]:
        """Mark past-deadline predictions UNKNOWN.

        Not a failure and not a success. A prediction nobody could observe
        must not be able to improve the calibration record by expiring.
        """
        moment = time.time() if now is None else now
        expired: list[Prediction] = []
        with self._lock:
            for prediction in self._predictions.values():
                if prediction.is_resolved or prediction.resolve_by > moment:
                    continue
                prediction.outcome = Outcome.UNKNOWN
                prediction.resolved_at = moment
                expired.append(prediction)
        return expired

    def calibration(self, claim_type: str | None = None) -> dict[str, Any]:
        """How well the ambient mind's confidence has matched reality."""
        with self._lock:
            predictions = [
                p
                for p in self._predictions.values()
                if claim_type is None or p.claim_type == claim_type
            ]
        graded = [
            p for p in predictions if p.outcome in (Outcome.CORRECT, Outcome.INCORRECT)
        ]
        unknown = sum(1 for p in predictions if p.outcome is Outcome.UNKNOWN)
        if not graded:
            return {
                "claim_type": claim_type,
                "graded": 0,
                "unknown": unknown,
                "base_rate": None,
                "mean_p_hat": None,
                "brier": None,
                "verdict": "nothing graded yet; confidence is unvalidated",
            }
        hits = sum(1 for p in graded if p.outcome is Outcome.CORRECT)
        base_rate = hits / len(graded)
        mean_p = sum(p.p_hat for p in graded) / len(graded)
        brier = sum(
            (p.p_hat - (1.0 if p.outcome is Outcome.CORRECT else 0.0)) ** 2
            for p in graded
        ) / len(graded)
        return {
            "claim_type": claim_type,
            "graded": len(graded),
            "unknown": unknown,
            "base_rate": round(base_rate, 4),
            "mean_p_hat": round(mean_p, 4),
            "brier": round(brier, 4),
            "overconfident_by": round(mean_p - base_rate, 4),
            "history_in_use": len(graded) >= _MIN_GRADED_FOR_HISTORY,
        }

    # ------------------------------------------------------------------ report

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        moment = time.time() if now is None else now
        with self._lock:
            today = self._day_key(moment)
            spent = self._spent_today_locked(now=moment) if self._daily_cap is not None else 0
            surfaced = sum(
                1
                for p in self._proposals
                if p.strength not in _FREE_STRENGTHS
                and p.status is not ProposalStatus.WITHHELD
            )
            observed = sum(
                1 for p in self._proposals if p.strength in _FREE_STRENGTHS
            )
            withheld = self._withheld
            cap = self._daily_cap
        return {
            "configured": cap is not None,
            "daily_silence_cap": cap,
            "day": today,
            "spent_today": spent,
            "remaining_today": None if cap is None else max(0, cap - spent),
            "observed": observed,
            "surfaced": surfaced,
            "withheld": withheld,
            # Silence is the success condition, so it is reported as one.
            "restraint_rate": (
                round(withheld / (withheld + surfaced), 4) if (withheld + surfaced) else 0.0
            ),
            "calibration": self.calibration(),
        }

    def reset_for_test(self) -> None:
        with self._lock:
            self._proposals.clear()
            self._predictions.clear()
            self._withheld = 0
            self._daily_cap = None


SERVICE_NAME = "ambient_governor"

_GOVERNOR = AmbientGovernor()


def get_ambient_governor() -> AmbientGovernor:
    """The process-wide governor, published on first use.

    The dependency runs this way deliberately. ``core/runtime`` is the
    foundation and may not import ``core.agency`` — the layering gate
    rejects it outright — so the runtime cannot fetch this itself. Agency
    registers; the runtime reads whatever is there and reports
    ``registered: False`` when nothing has.

    That last part matters more than it looks. A health surface that
    reported zeros for an unregistered governor would be indistinguishable
    from one reporting a governor that ran and had nothing to say — an
    absence presented as a clean result, which is the inversion this
    codebase keeps finding in its own gates.
    """
    _register_in_container(_GOVERNOR)
    return _GOVERNOR


def _register_in_container(governor: AmbientGovernor) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(SERVICE_NAME):
            ServiceContainer.register_instance(
                SERVICE_NAME,
                governor,
                required=False,
                registered_by="ambient_governor",
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "ambient_governor_register",
            exc,
            severity="debug",
            action="governor unpublished; its restraint will not appear in health",
        )
