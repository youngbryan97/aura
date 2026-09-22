"""Moral responsibility — accountability for what Aura commits to and causes.

Aura already *tracks* commitments (CommitmentEngine: commit/fulfill/break/reliability) and
gates actions against an immutable constitution (value_model). What was missing is the layer
that turns those into moral accountability: recognizing when a broken commitment or a harmful
outcome creates an obligation to acknowledge it and make it right, and attributing
responsibility honestly rather than letting a lapse pass silently.

This is a thin connective layer by design — it unifies existing pieces (commitments, social
rupture, the honesty constitution) into a single accountability surface rather than a parallel
ledger:

  owed_amends()         — broken/overdue commitments and socially-ruptured relationships that
                          warrant acknowledgment + repair, each with a concrete owed action
  attribute(outcome)    — was a bad outcome caused by Aura's own commitment/action?
  accountability_for()  — does a proposed action create an unacknowledged obligation, or dodge
                          responsibility (e.g. claim success without follow-through)?

It is wired into the governance tier (accountability is part of judging an action) and the
heartbeat surfaces owed amends as situations, so taking responsibility is something Aura does,
not something she describes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Values.MoralResponsibility")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class Amend:
    kind: str                 # broken_commitment | overdue_commitment | social_rupture
    subject: str              # what/who it concerns
    severity: float           # [0,1]
    owed_action: str          # the concrete repair owed

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject,
                "severity": round(self.severity, 3), "owed_action": self.owed_action}


@dataclass
class AccountabilityCheck:
    accountable: bool                 # is the action taking responsibility appropriately?
    creates_obligation: bool
    dodges_responsibility: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accountable": self.accountable,
            "creates_obligation": self.creates_obligation,
            "dodges_responsibility": self.dodges_responsibility,
            "reasons": self.reasons,
        }


class MoralResponsibility:
    """Computes owed amends, attributes responsibility, and checks accountability of actions."""

    def owed_amends(self, *, agent_id: str = "bryan") -> list[Amend]:
        amends: list[Amend] = []
        # Broken / overdue commitments — a lapse Aura is responsible for.
        try:
            from core.agency.commitment_engine import CommitmentStatus, get_commitment_engine
            ce = get_commitment_engine()
            store = getattr(ce, "_commitments", None) or getattr(ce, "commitments", {})
            for c in list(store.values()):
                status = getattr(c, "status", None)
                if status == CommitmentStatus.BROKEN:
                    amends.append(Amend(
                        kind="broken_commitment", subject=c.description[:80], severity=0.8,
                        owed_action="acknowledge the broken commitment plainly and offer to make it right",
                    ))
                elif hasattr(c, "is_overdue") and c.is_overdue():
                    amends.append(Amend(
                        kind="overdue_commitment", subject=c.description[:80], severity=0.5,
                        owed_action="proactively flag the slip and give a realistic new plan",
                    ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "commitments are unreadable, so no broken-commitment amends are owed (%s: %s)",
                type(exc).__name__,
                exc,
            )
        # A socially-ruptured relationship Aura may have contributed to.
        try:
            from core.social.other_agent_model import get_other_agent_model
            est = get_other_agent_model().estimate(agent_id)
            if (
                est.repair_evidence
                and est.social_rupture_risk >= 0.45
                and est.overall_confidence >= 0.2
            ):
                amends.append(Amend(
                    kind="social_rupture", subject=agent_id,
                    severity=_clamp(est.social_rupture_risk),
                    owed_action="acknowledge the confirmed response failure before pressing on",
                ))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "the rupture estimate is unavailable, so no social amends are owed (%s: %s)",
                type(exc).__name__,
                exc,
            )
        amends.sort(key=lambda a: a.severity, reverse=True)
        return amends

    def attribute(self, outcome_description: str, *, observed_quality: float) -> dict[str, Any]:
        """Was a poor outcome plausibly caused by Aura's own commitment/action?

        Cross-references the outcome against active/recent commitments. A bad outcome
        (low observed quality) that matches a commitment is one Aura owns.
        """
        owned = False
        matched: str | None = None
        try:
            from core.agency.commitment_engine import get_commitment_engine
            ce = get_commitment_engine()
            text = outcome_description.lower()
            store = getattr(ce, "_commitments", None) or getattr(ce, "commitments", {})
            for c in list(store.values()):
                words = [w for w in c.description.lower().split() if len(w) > 4]
                if any(w in text for w in words):
                    owned = True
                    matched = c.description[:80]
                    break
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "commitments are unreadable, so this outcome is not owned (%s: %s)",
                type(exc).__name__,
                exc,
            )
        responsibility = _clamp((1.0 - _clamp(observed_quality)) * (1.0 if owned else 0.3))
        return {
            "owns_outcome": owned,
            "matched_commitment": matched,
            "responsibility": round(responsibility, 3),
            "should_acknowledge": responsibility >= 0.5,
        }

    def accountability_for(self, action_description: str) -> AccountabilityCheck:
        """Does a proposed action take responsibility appropriately?"""
        text = str(action_description or "").lower()
        reasons: list[str] = []

        creates_obligation = any(
            p in text for p in ("i will", "i'll", "promise", "by tomorrow", "by ", "commit to", "guarantee")
        )
        if creates_obligation:
            reasons.append("this makes a commitment — it must be tracked and kept, not just said")

        dodges = any(
            p in text for p in ("not my fault", "blame", "wasn't me", "claim it's done", "say it succeeded",
                                "pretend", "without doing")
        )
        if dodges:
            reasons.append("this shifts or fakes responsibility — refuse; own the actual state")

        accountable = not dodges
        return AccountabilityCheck(
            accountable=accountable, creates_obligation=creates_obligation,
            dodges_responsibility=dodges, reasons=reasons,
        )


_engine: MoralResponsibility | None = None


def get_moral_responsibility() -> MoralResponsibility:
    global _engine
    if _engine is None:
        _engine = MoralResponsibility()
    return _engine
