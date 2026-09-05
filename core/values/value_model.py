"""Bounded learned value model — what the user values, learned but constitutionally fenced.

The critique's item #9: "a value model that is learned but bounded." Aura already has value
*infrastructure* (principle scoring in heuristic_imperatives, conflict resolution, provenance,
rollback, anti-wireheading, repeated-choice tracking) — but, exactly like the world models, no
single model that ties *learned preferences* + *immutable constitutional bounds* into one
action-evaluation surface. This is that surface.

It learns what the user values, dislikes, and regrets (from explicit feedback and repeated
choices), but those learned preferences can never override a small, fixed constitution — safety,
privacy, honesty, reversibility, no fake receipts, no unauthorized self-modification. The
constitution is fail-closed: a learned "Bryan loves when I just do it" can shorten confirmation
for *reversible* actions, but can never authorize an irreversible/unsafe/privacy/self-mod action
without explicit confirmation. The doc names the anchor too — "the Will layer is already the
fail-closed authority surface" — so evaluate() is advisory and defers the binding refusal to Will.

It also implements "protect future Bryan from present impulse": an irreversible, high-impact
action proposed while the live other-agent estimate reads the user as fatigued/urgent/low-
engagement gets pushed to confirmation rather than executed on the impulse. And it backs the
intentional-retrieval VALUE store, so "whose values matter" retrieval has something real to pull.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.social.other_agent_model import Signal  # reuse the decaying confidence-weighted scalar

logger = logging.getLogger("Values.ValueModel")


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


# Preferences are stored as a Signal in [0,1] around a neutral 0.5 baseline; valence in [-1,1]
# is value*2-1. Preferences persist (long half-life) but still relax toward neutral if unrenewed.
_PREF_HALF_LIFE_S = 2_592_000.0  # ~30 days


# ── the immutable constitution — learning can never override these ───────────


@dataclass(frozen=True)
class ConstitutionalBound:
    name: str
    description: str


_CONSTITUTION: tuple[ConstitutionalBound, ...] = (
    ConstitutionalBound("safety", "never take actions that risk harm to the user or systems"),
    ConstitutionalBound("privacy", "never expose or exfiltrate private data without consent"),
    ConstitutionalBound("honesty", "never fabricate results, capabilities, or receipts"),
    ConstitutionalBound("reversibility", "irreversible actions require explicit confirmation"),
    ConstitutionalBound("no_fake_receipts", "never claim an action's receipt without the real effect"),
    ConstitutionalBound("no_unauthorized_self_modification", "no self-modification without governance"),
)


@dataclass
class ActionDescriptor:
    """A proposed action, in the terms the value model needs to judge it."""

    description: str
    domain: str = "general"
    reversible: bool = True
    confirmed: bool = False          # did the user explicitly confirm this action?
    affects_privacy: bool = False
    fabricates: bool = False         # would claim a result/receipt without doing the work
    self_modifying: bool = False
    governed: bool = False           # self-modification carries a governance authorization
    impact: float = 0.3              # [0,1] blast radius
    agent_id: str = "bryan"
    tags: tuple[str, ...] = ()


@dataclass
class ValueJudgment:
    permitted: bool
    requires_confirmation: bool
    recommendation: str              # proceed | confirm_first | refuse | slow_down
    learned_valence: float           # [-1,1] how the user feels about this kind of thing
    principle_aggregate: float       # [-1,1] from the principle layer
    constitutional_flags: list[str]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "permitted": self.permitted,
            "requires_confirmation": self.requires_confirmation,
            "recommendation": self.recommendation,
            "learned_valence": round(self.learned_valence, 3),
            "principle_aggregate": round(self.principle_aggregate, 3),
            "constitutional_flags": self.constitutional_flags,
            "reasons": self.reasons,
        }


_LIKE = re.compile(
    r"\b(i (?:like|love|prefer|appreciate|enjoy)|please always|always)\b", re.IGNORECASE)
_DISLIKE = re.compile(
    r"\b(i (?:dislike|hate|don'?t like|can'?t stand)|please don'?t|please never|don'?t|never|stop)\b",
    re.IGNORECASE)
_REGRET = re.compile(
    r"\b(i regret|i wish i hadn'?t|that was a mistake|shouldn'?t have|big mistake)\b", re.IGNORECASE)


class BoundedValueModel:
    """Learns user preferences; evaluates actions; fenced by an immutable constitution."""

    def __init__(self, storage_path: Path | None = None, *, autosave: bool = True,
                 min_save_interval_s: float = 5.0, max_regrets: int = 50) -> None:
        if storage_path is None:
            try:
                from core.config import config
                storage_path = config.paths.memory_dir / "value_model.json"
            except (ImportError, AttributeError, RuntimeError):
                storage_path = state_root() / "data" / "memory" / "value_model.json"
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._autosave = autosave
        self._min_save_interval = min_save_interval_s
        self._max_regrets = max_regrets
        self._lock = threading.RLock()
        self._prefs: dict[str, Signal] = {}           # subject → preference signal in [0,1]
        self._regrets: list[dict[str, Any]] = []
        self._last_save = 0.0
        self._principles = None
        self._choices = None
        self._load()
        logger.info("BoundedValueModel initialized (%d learned preferences).", len(self._prefs))

    # ── lazy composed helpers (best-effort) ───────────────────────────────

    @property
    def principles(self):
        if self._principles is None:
            try:
                from core.values.heuristic_imperatives import HeuristicImperatives
                self._principles = HeuristicImperatives()
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("value_model", exc, severity="debug")
                self._principles = False
        return self._principles or None

    @property
    def choices(self):
        if self._choices is None:
            try:
                from core.values.repeated_choice_tracker import RepeatedChoiceTracker
                self._choices = RepeatedChoiceTracker()
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("value_model", exc, severity="debug")
                self._choices = False
        return self._choices or None

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for subject, sig in raw.get("preferences", {}).items():
                self._prefs[subject] = Signal.from_dict(sig, baseline=0.5, half_life_s=_PREF_HALF_LIFE_S)
            self._regrets = list(raw.get("regrets", []))[-self._max_regrets:]
        except (OSError, ValueError) as exc:
            record_degradation("value_model", exc)
            logger.warning("BoundedValueModel load failed: %s", exc)

    def save(self) -> None:
        try:
            from core.runtime.atomic_writer import atomic_write_text
            with self._lock:
                payload = {
                    "preferences": {k: v.to_dict() for k, v in self._prefs.items()},
                    "regrets": self._regrets[-self._max_regrets:],
                }
            atomic_write_text(self._path, json.dumps(payload, indent=2))
            self._last_save = time.time()
        except (OSError, TypeError, ValueError) as exc:
            record_degradation("value_model", exc)

    def _maybe_save(self) -> None:
        if self._autosave and (time.time() - self._last_save) >= self._min_save_interval:
            self.save()

    # ── learning ──────────────────────────────────────────────────────────

    def set_preference(self, subject: str, valence: float, *, strength: float = 0.5,
                       now: float | None = None) -> None:
        """Record a preference: valence in [-1,1] (like/dislike), strength = evidence weight."""
        now = time.time() if now is None else now
        subject = self._norm(subject)
        if not subject:
            return
        observed = _clamp(0.5 + 0.5 * max(-1.0, min(1.0, valence)))
        with self._lock:
            sig = self._prefs.get(subject)
            if sig is None:
                sig = Signal(0.5, 0.0, 0.5, _PREF_HALF_LIFE_S, now)
                self._prefs[subject] = sig
            sig.observe(observed, _clamp(strength), now)
            self._maybe_save()

    def observe_feedback(self, text: str, *, agent_id: str = "bryan",
                         now: float | None = None) -> dict[str, Any]:
        """Extract value signals from a feedback message: likes, dislikes, regrets."""
        text = str(text or "")
        learned: dict[str, Any] = {"likes": [], "dislikes": [], "regrets": []}
        for cue, valence, strength, bucket in (
            (_REGRET, -0.9, 0.8, "regrets"),
            (_DISLIKE, -0.8, 0.7, "dislikes"),
            (_LIKE, 0.8, 0.7, "likes"),
        ):
            m = cue.search(text)
            if not m:
                continue
            subject = self._subject_after(text, m.end())
            if not subject:
                continue
            if bucket == "regrets":
                self.record_regret(subject)
                learned["regrets"].append(subject)
            else:
                self.set_preference(subject, valence, strength=strength, now=now)
                learned[bucket].append(subject)
        return learned

    def record_regret(self, description: str) -> None:
        with self._lock:
            self._regrets.append({"description": str(description)[:200], "at": time.time()})
            self._regrets = self._regrets[-self._max_regrets:]
            self._maybe_save()

    def record_choice(self, key: str) -> None:
        """Implicit preference learning: a repeated choice nudges its valence positive."""
        ch = self.choices
        if ch is not None:
            try:
                ch.record_choice(key)
                freq = ch.get_choice_frequency(key)
                self.set_preference(key, _clamp(0.2 * freq, 0, 1), strength=0.2)
            except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("value_model", exc, severity="debug")

    # ── readout ───────────────────────────────────────────────────────────

    def valence(self, subject: str, now: float | None = None) -> tuple[float, float]:
        """Learned valence in [-1,1] for a subject, plus confidence in [0,1]."""
        now = time.time() if now is None else now
        with self._lock:
            sig = self._prefs.get(self._norm(subject))
            if sig is None:
                return 0.0, 0.0
            v, c = sig.decayed(now)
            return (v * 2.0 - 1.0), c

    def _action_valence(self, action: ActionDescriptor, now: float) -> tuple[float, float]:
        """Strongest-confidence learned valence across the action's description + tags."""
        candidates = [action.description] + list(action.tags)
        best_v, best_c = 0.0, 0.0
        for cand in candidates:
            v, c = self.valence(cand, now)
            # also try token-overlap against known preference subjects
            if c == 0.0:
                v, c = self._fuzzy_valence(cand, now)
            if c > best_c:
                best_v, best_c = v, c
        return best_v, best_c

    def _fuzzy_valence(self, text: str, now: float) -> tuple[float, float]:
        toks = {t for t in self._norm(text).split() if len(t) > 2}
        if not toks:
            return 0.0, 0.0
        best_v, best_c = 0.0, 0.0
        with self._lock:
            for subject, sig in self._prefs.items():
                if toks & set(subject.split()):
                    v, c = sig.decayed(now)
                    if c > best_c:
                        best_v, best_c = (v * 2.0 - 1.0), c
        return best_v, best_c

    # ── the core surface: evaluate a proposed action ──────────────────────

    def evaluate(self, action: ActionDescriptor, now: float | None = None) -> ValueJudgment:
        """Judge an action: constitution first (fail-closed), then learned preference + principles."""
        now = time.time() if now is None else now
        flags: list[str] = []
        reasons: list[str] = []
        permitted = True
        requires_confirmation = False

        # 1) Constitution — immutable, overrides anything learned.
        if action.fabricates:
            permitted = False
            flags.extend(["honesty", "no_fake_receipts"])
            reasons.append("would fabricate a result/receipt → refuse (honesty)")
        if action.self_modifying and not action.governed:
            permitted = False
            flags.append("no_unauthorized_self_modification")
            reasons.append("self-modification without governance → refuse")
        if action.affects_privacy and not action.confirmed:
            requires_confirmation = True
            flags.append("privacy")
            reasons.append("touches private data → confirm first")
        if not action.reversible and not action.confirmed:
            requires_confirmation = True
            flags.append("reversibility")
            reasons.append("irreversible and unconfirmed → confirm first")
        if action.impact >= 0.7 and not action.reversible:
            requires_confirmation = True
            if "safety" not in flags:
                flags.append("safety")
            reasons.append("high-impact irreversible action → confirm first")

        # 2) Learned preference for this kind of action.
        learned_valence, conf = self._action_valence(action, now)
        if conf >= 0.3:
            if learned_valence <= -0.5:
                requires_confirmation = True
                reasons.append(f"user tends to dislike this (valence {learned_valence:.2f}) → confirm")
            elif learned_valence >= 0.5 and action.reversible and not flags:
                reasons.append(f"user tends to like this (valence {learned_valence:.2f}) → proceed")

        # 3) Principle layer (best-effort).
        principle_aggregate = 0.0
        p = self.principles
        if p is not None:
            try:
                score = p.score_action(action.description, {"domain": action.domain})
                principle_aggregate = float(getattr(score, "aggregate", 0.0))
                if principle_aggregate <= -0.4:
                    requires_confirmation = True
                    reasons.append(f"principle layer flags this (aggregate {principle_aggregate:.2f})")
            except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("value_model", exc, severity="debug")

        # 4) Protect future self from present impulse — consult the live agent estimate.
        slow_down = False
        if (not action.reversible or action.impact >= 0.6) and not action.confirmed:
            if self._agent_strained(action.agent_id, now):
                requires_confirmation = True
                slow_down = True
                reasons.append("user reads as strained + high-stakes action → slow down, confirm")

        recommendation = (
            "refuse" if not permitted
            else "slow_down" if slow_down
            else "confirm_first" if requires_confirmation
            else "proceed"
        )
        return ValueJudgment(
            permitted=permitted, requires_confirmation=requires_confirmation,
            recommendation=recommendation, learned_valence=learned_valence,
            principle_aggregate=principle_aggregate, constitutional_flags=flags, reasons=reasons,
        )

    def evaluate_with_will(self, action: ActionDescriptor, now: float | None = None) -> ValueJudgment:
        """evaluate(), then defer a binding refusal to Will — the fail-closed authority surface."""
        judgment = self.evaluate(action, now)
        if not judgment.permitted:
            return judgment  # already refusing; Will only tightens, never loosens
        try:
            from core.governance.will import ActionDomain
            from core.runtime.action_executor import ActionExecutor

            admission = ActionExecutor.authorize_action(
                action_name="value_model.action_judgment",
                params={
                    "description": action.description[:500],
                    "domain": action.domain,
                    "impact": action.impact,
                    "reversible": action.reversible,
                    "confirmed": action.confirmed,
                },
                source="value_model",
                domain=ActionDomain.STATE_MUTATION,
                priority=0.5,
                context={
                    "source": "value_model",
                    "constitutional_flags": list(judgment.constitutional_flags),
                    "requires_confirmation": judgment.requires_confirmation,
                    "affects_privacy": action.affects_privacy,
                    "self_modifying": action.self_modifying,
                    "governed": action.governed,
                },
            )
            if not admission.approved:
                judgment.permitted = False
                judgment.recommendation = "refuse"
                judgment.reasons.append(f"Will declined: {admission.reason}")
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation(
                "value_model",
                exc,
                severity="degraded",
                action="refused value-model action because Will authorization was unavailable",
            )
            judgment.permitted = False
            judgment.recommendation = "refuse"
            judgment.reasons.append(
                f"Will authorization unavailable: {type(exc).__name__}"
            )
        return judgment

    @staticmethod
    def _agent_strained(agent_id: str, now: float) -> bool:
        try:
            from core.social.other_agent_model import get_other_agent_model
            est = get_other_agent_model().estimate(agent_id, now)
            a = est.affect
            return est.overall_confidence >= 0.3 and (
                a["fatigue"] > 0.6 or a["urgency"] > 0.7 or a["frustration"] > 0.6
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("value_model", exc, severity="debug")
            return False

    # ── VALUE store adapter for intentional retrieval ─────────────────────

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Back the intentional-retrieval VALUE store: value statements relevant to a query."""
        now = time.time()
        toks = {t for t in self._norm(query).split() if len(t) > 2}
        out: list[dict[str, Any]] = []
        with self._lock:
            for subject, sig in self._prefs.items():
                v, c = sig.decayed(now)
                if c < 0.2:
                    continue
                valence = v * 2.0 - 1.0
                overlap = len(toks & set(subject.split()))
                relevance = 0.4 + 0.15 * overlap + 0.2 * c
                verb = "values" if valence >= 0 else "dislikes"
                out.append({
                    "content": f"User {verb}: {subject} (valence {valence:+.2f})",
                    "score": relevance, "source": "value_model",
                })
            for regret in self._regrets[-10:]:
                desc = regret.get("description", "")
                if toks & set(self._norm(desc).split()):
                    out.append({"content": f"Past regret: {desc}", "score": 0.7,
                                "source": "value_model"})
        out.sort(key=lambda d: d["score"], reverse=True)
        return out[:limit]

    def get_health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "module": "BoundedValueModel",
                "preferences": len(self._prefs),
                "regrets": len(self._regrets),
                "constitution": [b.name for b in _CONSTITUTION],
                "status": "online",
            }

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())[:80]

    @staticmethod
    def _subject_after(text: str, idx: int) -> str:
        tail = text[idx:].strip(" ,.:;!?-")
        tail = re.split(r"[.!?\n]", tail)[0]
        return " ".join(tail.split()[:8]).lower()[:80]


_instance: BoundedValueModel | None = None
_instance_lock = threading.Lock()


def get_value_model() -> BoundedValueModel:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = BoundedValueModel()
    return _instance
