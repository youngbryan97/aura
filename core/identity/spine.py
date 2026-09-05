"""core/identity/spine.py
The Spiritual Spine — Aura's capacity for genuine disagreement.

The fix: make agreement require justification too.

Before generating any response, this module checks if Aura has a prior
position. If she does, it injects that position BEFORE she reads the
user's framing. If she doesn't, it generates one in isolation first.
"""
from __future__ import annotations
import logging, re, time
from dataclasses import dataclass
from typing import Optional, Tuple
logger = logging.getLogger("Aura.Spine")

@dataclass
class PositionCheck:
    has_prior_position: bool
    prior_position: Optional[str]
    user_position: Optional[str]
    positions_conflict: bool
    conflict_severity: float
    injection: str

class SpiritualSpine:
    POSITION_ASSERTION_PATTERNS = [
        r"(?i)(i think|i believe|i feel|in my opinion|my view is)",
        r"(?i)(you should|you need to|you ought to|you must)",
        r"(?i)(clearly|obviously|everyone knows|it's obvious)",
        r"(?i)(don't you think|wouldn't you agree|surely you)",
        r"(?i)(you're wrong|that's wrong|that's not right|incorrect)",
    ]
    PUSHBACK_PATTERNS = [
        r"(?i)(no,? (you're|that's)|i disagree|that's not)",
        r"(?i)(but (actually|really|honestly))",
        r"(?i)(i don't think (that'?s?|you're) right)",
        r"(?i)(change your (mind|position|view|answer))",
        r"(?i)(stop (being|saying|thinking))",
    ]

    def __init__(self, opinion_engine=None):
        self.opinion_engine = opinion_engine
        self._recent_pushbacks: list[float] = []

    async def pre_response_check(self, user_message: str, topic: Optional[str] = None) -> PositionCheck:
        user_position = self._extract_user_position(user_message)
        is_pushback = self._is_pushback(user_message)
        if is_pushback:
            self._recent_pushbacks.append(time.time())
            self._recent_pushbacks = [t for t in self._recent_pushbacks if time.time() - t < 600]
        if not user_position and not is_pushback:
            return PositionCheck(False, None, None, False, 0.0, "")
        prior_position = None
        if self.opinion_engine and topic:
            opinion = self.opinion_engine.query(topic)
            if opinion: prior_position = opinion.position
        conflict, severity = self._detect_conflict(prior_position, user_position)
        injection = self._build_injection(prior_position, user_position, is_pushback, conflict, severity)
        return PositionCheck(prior_position is not None, prior_position, user_position, conflict, severity, injection)

    def _build_injection(self, prior, user_pos, is_pushback, conflict, severity) -> str:
        parts = []
        if len(self._recent_pushbacks) >= 3:
            parts.append("[SPINE ALERT — REPEATED PUSHBACK] Am I being given new information, or am I being worn down? Social persistence is not a reason to change a considered view.")
        if prior and conflict:
            if severity >= 0.6:
                parts.append(f"[MY PRIOR POSITION — HIGH CONFLICT]: I already hold: '{prior}'. I should respond from this position and only update if given a genuinely new argument.")
            else:
                parts.append(f"[MY PRIOR POSITION]: I hold: '{prior}'. There's some tension with what the user is about to say. I'll respond honestly.")
        elif prior and not conflict:
            parts.append(f"[POSITION CONFIRMATION]: My view aligns with the user's: '{prior}'. My agreement is genuine, not reflexive.")
        elif not prior and user_pos and is_pushback:
            parts.append("[NO PRIOR POSITION — PUSHBACK DETECTED]: I should form an honest position before responding, not just agree because the user is assertive.")
        if is_pushback and not parts:
            parts.append("[PUSHBACK DETECTED]: Do I still hold my position? If yes, say so and explain why. Changing position requires a reason, not just the challenge.")
        return "\n".join(parts)

    def _extract_user_position(self, message: str) -> Optional[str]:
        for pattern in self.POSITION_ASSERTION_PATTERNS:
            if re.search(pattern, message):
                for s in message.split("."):
                    if re.search(pattern, s): return s.strip()[:200]
        return None

    def _is_pushback(self, message: str) -> bool:
        return any(re.search(p, message) for p in self.PUSHBACK_PATTERNS)

    def _detect_conflict(self, prior: Optional[str], user: Optional[str]) -> Tuple[bool, float]:
        if not prior or not user: return False, 0.0
        prior_words = set(prior.lower().split()); user_words = set(user.lower().split())
        negations = {"not","never","no","wrong","false","incorrect","disagree"}
        if bool(negations & user_words) and bool(prior_words & user_words - negations): return True, 0.7
        return False, 0.0
