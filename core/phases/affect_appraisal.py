"""The affect phase's appraisal of a turn: her checks read, and felt.

The checks and the feelings they make are in core/affect/appraisal.py. This
reads each check from the ledger or model of hers that measures it, and moves
each emotion by the share of its room the appraisal gives it. The gain is the
one a percept of the same strength has, so a turn's meaning and a turn's events
move her feelings on one scale.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.affect.appraisal import Appraisal, OutcomeReader, coping_of, feelings
from core.phases.affect_readings import AFFECT_UPDATE_ERRORS, RecordsADegradation, _her_agency
from core.state.aura_state import AffectVector, AuraState

logger = logging.getLogger(__name__)


def _novelty() -> float | None:
    """Her world model's latest surprise among its recent ones; None before it has seen anything."""
    from core.runtime.service_registry import get_runtime_service

    model = get_runtime_service("unified_world_model", default=None)
    learned = getattr(model, "learned", None)
    if learned is None or not int(learned.get_status().get("step_count", 0) or 0):
        return None
    rank = model.surprise_rank()
    return None if rank is None else float(rank)


def _what_the_turn_named(state: AuraState) -> str:
    """The objective and the last thing said to her, the text her likes are read in."""
    cognition = state.cognition
    said = ""
    for item in reversed(list(getattr(cognition, "working_memory", []) or [])):
        if isinstance(item, dict) and item.get("role") == "user":
            said = str(item.get("content") or "")
            break
    return f"{getattr(cognition, 'current_objective', '') or ''} {said}".strip()


def _liking(state: AuraState) -> float | None:
    text = _what_the_turn_named(state)
    if not text:
        return None
    from core.conation.what_she_wants import get_what_she_wants

    return float(get_what_she_wants().regard_for_text(text))


def _social_need(state: AuraState) -> float | None:
    """How far her social drive is below its capacity, in [0, 1]; None when she has none.

    The drive, not the affect field called social hunger: that field held its
    default of 0.5 for every turn of a 24-turn offline life on 22 October,
    which made belonging a constant push whenever somebody spoke.
    """
    budgets = getattr(getattr(state, "motivation", None), "budgets", {}) or {}
    social = budgets.get("social")
    if not isinstance(social, dict):
        return None
    capacity = float(social.get("capacity", 0.0) or 0.0)
    if capacity <= 0.0:
        return None
    level = float(social.get("level", capacity) or 0.0)
    return max(0.0, min(1.0, (capacity - level) / capacity))


class TurnAppraisal:
    """Reads a turn's appraisal and lets it move her feelings."""

    def __init__(self, record: RecordsADegradation, *, gain: float, bump: Callable[[dict, str, float], None]) -> None:
        self._record = record
        self._gain = gain
        self._bump = bump
        self._outcomes = OutcomeReader()
        #: The warmth ledger's message count when it was last read.
        self._acts_read = 0

    def read(self, state: AuraState, affect: AffectVector, *, company: bool) -> Appraisal:
        """This turn's checks, each None where its source cannot be read."""
        checks: dict[str, Any] = {
            "company": 1.0 if company else 0.0,
            "social_need": _social_need(state),
            "fear": float(affect.emotions.get("fear", 0.0) or 0.0),
            "hope": float(affect.emotions.get("hope", 0.0) or 0.0),
        }
        try:
            from core.affect.what_it_was_worth import get_worth_ledger

            checks.update(self._outcomes.read(get_worth_ledger().read()))
        except AFFECT_UPDATE_ERRORS as exc:
            logger.debug("the turn's outcome could not be appraised: %s", exc)
        try:
            from core.social.warmth import get_warmth_ledger

            warmth = get_warmth_ledger()
            # A message heard since the last appraisal: what the person did is
            # that message, warm or objecting. Otherwise the worth channel of
            # what being met returned stands for their doing.
            if warmth.heard_count != self._acts_read:
                self._acts_read = warmth.heard_count
                checks["their_doing"] = warmth.act()
        except AFFECT_UPDATE_ERRORS as exc:
            logger.debug("what the person did could not be read: %s", exc)
        ledger = _her_agency()
        if ledger is not None:
            checks["coping"] = coping_of(int(ledger.acted), int(ledger.succeeded))
        for name, reader in (("novelty", _novelty), ("liking", lambda: _liking(state))):
            try:
                checks[name] = reader()
            except AFFECT_UPDATE_ERRORS as exc:
                logger.debug("the %s check could not be read: %s", name, exc)
        return Appraisal(**checks)

    # A method rather than `__call__`: the subject fork carries a phase's
    # attributes but skips callable ones, so a callable appraisal left its
    # reader out of every snapshot, and two untouched forks of one anchor
    # parted at the ninth frame (22 October).
    def feel(self, state: AuraState, affect: AffectVector, *, company: bool) -> dict[str, float]:
        """Appraise the turn and move each emotion it names; returns what it felt."""
        try:
            appraisal = self.read(state, affect, company=company)
            felt = {name: value for name, value in feelings(appraisal).items() if name in affect.emotions}
            for name, value in felt.items():
                self._bump(affect.emotions, name, value * self._gain)
            strongest = sorted(felt.items(), key=lambda item: -abs(item[1]))
            affect.markers["appraisal"] = {
                "checks": appraisal.as_dict(),
                "felt": {name: round(value, 4) for name, value in strongest if value},
            }
            return felt
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="appraisal",
                action="kept the feelings the turn's events gave her; its meaning was not appraised",
                severity="warning",
            )
            return {}
