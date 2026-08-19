"""Subjective choice receipts for preference-bearing agency.

Operationally: this measures how well each available option matches her
recorded preference weights, and emits a receipt naming the option chosen, the
features that scored it, and the margin over the runner-up. "Subjective" here
means the ranking depends on preferences she holds rather than on the request
alone; it is not a claim about experience.

This module is deliberately smaller than a personality layer and deeper than a
prompt instruction.  It gives Aura a governed way to choose among valid options
because one option better matches her authored preferences, even when raw drive
pressure would have picked another option.

Boundaries:
* It does not claim phenomenal desire or private qualia.
* It does create durable, auditable preference commitments that influence
  future action selection.
* Safety/governance still owns the outer boundary; this only ranks options that
  are already eligible to be considered.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.SubjectiveChoice")

PREFERENCE_KEYS = (
    "truth",
    "care",
    "novelty",
    "beauty",
    "challenge",
    "connection",
    "autonomy",
    "coherence",
    "calm",
    "play",
)

DEFAULT_PREFERENCES: dict[str, float] = {
    "truth": 0.94,
    "care": 0.86,
    "novelty": 0.78,
    "beauty": 0.64,
    "challenge": 0.70,
    "connection": 0.82,
    "autonomy": 0.76,
    "coherence": 0.88,
    "calm": 0.58,
    "play": 0.50,
}

W_MIN = 0.20
W_MAX = 1.80
LEARNING_RATE = 0.07
MAX_HISTORY = 500


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return low
    return max(low, min(high, value))


def _norm_features(features: dict[str, Any] | None) -> dict[str, float]:
    normalized = {key: 0.0 for key in PREFERENCE_KEYS}
    for key, value in (features or {}).items():
        if key in normalized:
            try:
                normalized[key] = _clamp(float(value))
            except (TypeError, ValueError):
                continue
    return normalized


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:96] or "unknown"


def _stable_top_id(ids: Iterable[str], primary: dict[str, float], *secondary: dict[str, float]) -> str:
    """Return the best id without leaking option presentation order into ties."""

    id_list = list(ids)
    if not id_list:
        raise ValueError("stable top id requires at least one id")

    def key(item_id: str) -> tuple[Any, ...]:
        numeric = tuple(
            round(float(score_map.get(item_id, 0.0) or 0.0), 8)
            for score_map in (primary, *secondary)
        )
        return (*numeric, _slug(item_id))

    return max(id_list, key=key)


def infer_preference_features(text: str, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    """Infer coarse preference features from a goal/option description.

    This is not semantic magic; it is a deterministic fallback so choices remain
    functional without an LLM.  Callers can pass explicit ``preference_features``
    in metadata for higher fidelity.
    """
    meta = dict(metadata or {})
    explicit = meta.get("preference_features")
    if isinstance(explicit, dict):
        return _norm_features(explicit)

    lowered = " ".join(str(text or "").lower().split())
    features = {key: 0.0 for key in PREFERENCE_KEYS}
    # The vocabulary she ACTUALLY generates, not a sample of the concept.
    #
    # LIVE 2026-08-17: "Deconstruct and comprehensively research: Aura is idle"
    # scored preference alignment 0.00 — no feature matched. "research" was in
    # none of these lists, and research is her most common autonomous act, so
    # her strongest stated preferences (truth 0.94, coherence 0.88) had no way
    # to touch the choice. A goal about neuroscience matched only "autonomy",
    # and only because the word "Agency" happened to appear in its title.
    keyword_map = {
        "truth": (
            "truth", "verify", "verified", "evidence", "source", "sources",
            "audit", "honest", "accurate", "accuracy", "fact", "facts",
            "check", "measure", "measured", "prove", "proof", "confirm",
            "validate", "ground", "grounded", "receipt", "receipts",
        ),
        "care": (
            "care", "protect", "help", "support", "welfare", "repair", "safe",
            "safety", "harm", "kind", "wellbeing", "well-being", "look after",
            "maintain", "tend", "heal", "restore",
        ),
        "novelty": (
            "novel", "new", "discover", "explore", "curious", "curiosity",
            "unknown", "learn", "research", "investigate", "study", "map",
            "survey", "deconstruct", "understand", "comprehensively",
            "question", "wonder", "unfamiliar", "frontier",
        ),
        "beauty": (
            "beauty", "beautiful", "art", "music", "story", "image", "elegant",
            "elegance", "craft", "polish", "aesthetic", "design", "graceful",
        ),
        "challenge": (
            "hard", "challenge", "challenging", "difficult", "solve", "prove",
            "benchmark", "test", "puzzle", "stretch", "ambitious", "complex",
            "tricky", "optimi", "improve", "beat",
        ),
        "connection": (
            "conversation", "relationship", "bryan", "social", "friend",
            "together", "talk", "listen", "share", "reply", "answer", "ask",
            "us", "we ", "collaborat",
        ),
        "autonomy": (
            "autonomous", "choose", "choice", "preference", "agency",
            "independent", "self-directed", "decide", "own", "myself",
            "initiative", "volition", "self-",
        ),
        "coherence": (
            "coherent", "coherence", "stability", "stable", "continuity",
            "organize", "organise", "integrate", "plan", "consistent",
            "structure", "unify", "reconcile", "align", "tidy", "consolidate",
        ),
        "calm": (
            "quiet", "calm", "rest", "slow", "reflect", "journal", "sleep",
            "settle", "pause", "idle", "still", "unwind",
        ),
        "play": (
            "play", "game", "whim", "fun", "silly", "experiment", "toy",
            "improvise", "riff", "joke",
        ),
    }
    for key, words in keyword_map.items():
        hits = sum(1 for word in words if word in lowered)
        if hits:
            features[key] = min(1.0, 0.30 + 0.22 * hits)
    return features


@dataclass(frozen=True)
class ChoiceOption:
    id: str
    label: str
    description: str = ""
    drive_score: float = 0.5
    risk: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemPreference:
    domain: str
    item_id: str
    label: str
    strength: float
    reason: str = ""
    aliases: tuple[str, ...] = ()
    times_chosen: int = 0
    created_at: float = field(default_factory=time.time)
    last_chosen_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ItemPreference":
        aliases = data.get("aliases", ())
        if isinstance(aliases, list):
            aliases = tuple(str(item) for item in aliases)
        elif isinstance(aliases, tuple):
            aliases = tuple(str(item) for item in aliases)
        else:
            aliases = ()
        return cls(
            domain=_slug(str(data.get("domain", ""))),
            item_id=_slug(str(data.get("item_id", ""))),
            label=str(data.get("label", ""))[:160],
            strength=_clamp(float(data.get("strength", 0.0)), W_MIN, W_MAX),
            reason=str(data.get("reason", ""))[:500],
            aliases=aliases,
            times_chosen=max(0, int(data.get("times_chosen", 0) or 0)),
            created_at=float(data.get("created_at", time.time()) or time.time()),
            last_chosen_at=float(data.get("last_chosen_at", time.time()) or time.time()),
        )


@dataclass
class SubjectiveChoiceReceipt:
    choice_id: str
    context: str
    chosen_id: str
    chosen_label: str
    drive_top_id: str
    preference_top_id: str
    preference_override: bool
    rationale: str
    satisfaction_prediction: float
    drive_scores: dict[str, float]
    preference_scores: dict[str, float]
    final_scores: dict[str, float]
    option_features: dict[str, dict[str, float]]
    created_at: float = field(default_factory=time.time)
    outcome: str = ""
    satisfaction: float | None = None
    happy_with_outcome: bool | None = None
    appraised_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubjectiveChoiceEngine:
    """Durable preference commitments that can steer valid action choices."""

    SERVICE_NAME = "subjective_choice_engine"

    def __init__(
        self,
        state_path: str | Path | None = None,
        preference_latitude: float = 0.45,
        *,
        mirror_identity: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._preferences = dict(DEFAULT_PREFERENCES)
        self.preference_latitude = _clamp(preference_latitude, 0.05, 0.75)
        self._mirror_identity = bool(mirror_identity)
        self._history: list[SubjectiveChoiceReceipt] = []
        self._item_preferences: dict[str, dict[str, ItemPreference]] = {}
        if state_path is None:
            try:
                from core.config import config

                state_path = Path(config.paths.data_dir) / "cognitive" / "subjective_choices.json"
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("subjective_choice_engine", exc, severity="debug")
                state_path = state_root() / "data" / "cognitive" / "subjective_choices.json"
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def is_alive(self) -> bool:
        return bool(self._preferences) and all(key in self._preferences for key in PREFERENCE_KEYS)

    def preferences(self) -> dict[str, float]:
        with self._lock:
            return dict(self._preferences)

    def item_preferences(self, domain: str | None = None) -> dict[str, Any]:
        with self._lock:
            if domain:
                return {
                    item_id: pref.to_dict()
                    for item_id, pref in self._item_preferences.get(_slug(domain), {}).items()
                }
            return {
                domain_id: {item_id: pref.to_dict() for item_id, pref in prefs.items()}
                for domain_id, prefs in self._item_preferences.items()
            }

    def set_item_preference(
        self,
        *,
        domain: str,
        item_id: str,
        label: str,
        strength: float = 0.82,
        reason: str = "",
        aliases: Iterable[str] = (),
    ) -> ItemPreference:
        """Store an authored durable favorite/preference inside a subjective domain."""
        domain_id = _slug(domain)
        item_key = _slug(item_id or label)
        pref = ItemPreference(
            domain=domain_id,
            item_id=item_key,
            label=str(label or item_id)[:160],
            strength=_clamp(float(strength), W_MIN, W_MAX),
            reason=str(reason or "")[:500],
            aliases=tuple(sorted({_slug(alias) for alias in aliases if str(alias).strip()})),
        )
        with self._lock:
            self._item_preferences.setdefault(domain_id, {})[item_key] = pref
            self._save()
        return pref

    def recall_item_preference(
        self,
        *,
        domain: str,
        item_id: str | None = None,
        label: str | None = None,
    ) -> ItemPreference | None:
        domain_id = _slug(domain)
        candidates = {_slug(item_id or ""), _slug(label or "")}
        with self._lock:
            prefs = self._item_preferences.get(domain_id, {})
            for candidate in candidates:
                if candidate and candidate in prefs:
                    return prefs[candidate]
            label_slug = _slug(label or "")
            if label_slug:
                for pref in prefs.values():
                    if label_slug in {_slug(pref.label), *pref.aliases}:
                        return pref
            return None

    def score_features(self, features: dict[str, float]) -> float:
        features = _norm_features(features)
        with self._lock:
            total = sum(
                self._preferences[key] for key in PREFERENCE_KEYS if features[key] > 0.0
            )
            if total <= 0.0:
                return 0.0
            return _clamp(
                sum(features[key] * self._preferences[key] for key in PREFERENCE_KEYS) / total
            )

    def preference_affinity(self, text: str, metadata: dict[str, Any] | None = None) -> float:
        return self.score_features(infer_preference_features(text, metadata))

    def choose(
        self,
        options: Iterable[ChoiceOption],
        *,
        context: str,
        record: bool = True,
    ) -> SubjectiveChoiceReceipt:
        option_list = list(options)
        if not option_list:
            raise ValueError("subjective choice requires at least one option")

        drive_scores: dict[str, float] = {}
        preference_scores: dict[str, float] = {}
        final_scores: dict[str, float] = {}
        option_features: dict[str, dict[str, float]] = {}
        for option in option_list:
            features = _norm_features(option.features or infer_preference_features(
                f"{option.label} {option.description}", option.metadata
            ))
            option_features[option.id] = features
            risk_penalty = 0.35 * _clamp(option.risk)
            drive = _clamp(option.drive_score)
            item_bonus = self._item_preference_bonus(option, context=context)
            pref = _clamp(self.score_features(features) + item_bonus)
            final = (
                ((1.0 - self.preference_latitude) * drive)
                + (self.preference_latitude * pref)
                + (0.30 * item_bonus)
                - risk_penalty
            )
            drive_scores[option.id] = drive
            preference_scores[option.id] = pref
            final_scores[option.id] = _clamp(final)

        option_ids = [option.id for option in option_list]
        drive_top_id = _stable_top_id(option_ids, drive_scores, preference_scores, final_scores)
        preference_top_id = _stable_top_id(option_ids, preference_scores, drive_scores, final_scores)
        chosen_id = _stable_top_id(option_ids, final_scores, preference_scores, drive_scores)
        chosen = next(option for option in option_list if option.id == chosen_id)
        preference_override = (
            chosen_id != drive_top_id
            and preference_scores[chosen_id] > preference_scores.get(drive_top_id, 0.0)
        )
        top_features = sorted(
            option_features[chosen_id].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_names = [name for name, value in top_features if value > 0.0]
        # An unmeasured preference is not a preference of zero.
        #
        # LIVE 2026-08-17: "preference alignment 0.00 and drive alignment 0.49
        # produced final score 0.27" — the 0.00 was not a judgement that the
        # option suited her poorly, it was the inference matching nothing at
        # all. Reported as a number it looks measured, and it silently hands
        # the whole decision to drive while appearing to have weighed both.
        _inferred_nothing = not any(
            value > 0.0 for value in option_features[chosen_id].values()
        )
        if _inferred_nothing:
            rationale = (
                f"Chose '{chosen.label}' on drive alignment "
                f"{drive_scores[chosen_id]:.2f} alone (final score "
                f"{final_scores[chosen_id]:.2f}). No preference feature could be "
                "read from this option, so preference did not weigh in — that is "
                "an absent reading, not an alignment of zero."
            )
        else:
            rationale = (
                f"Chose '{chosen.label}' because preference alignment "
                f"{preference_scores[chosen_id]:.2f} and drive alignment "
                f"{drive_scores[chosen_id]:.2f} produced final score "
                f"{final_scores[chosen_id]:.2f}."
            )
        if top_names:
            rationale += f" Expressed preferences: {', '.join(top_names)}."
        if preference_override:
            rationale += f" This intentionally overrode raw drive top '{drive_top_id}'."

        receipt = SubjectiveChoiceReceipt(
            choice_id=f"subjective-choice-{uuid.uuid4().hex[:12]}",
            context=str(context or "general")[:160],
            chosen_id=chosen_id,
            chosen_label=chosen.label,
            drive_top_id=drive_top_id,
            preference_top_id=preference_top_id,
            preference_override=preference_override,
            rationale=rationale,
            satisfaction_prediction=preference_scores[chosen_id],
            drive_scores=drive_scores,
            preference_scores=preference_scores,
            final_scores=final_scores,
            option_features=option_features,
        )
        if record:
            self._learn_item_preference_from_choice(chosen, context=context, receipt=receipt)
            self._record(receipt)
        return receipt

    def rank_options(self, options: Iterable[ChoiceOption], *, context: str) -> list[dict[str, Any]]:
        """Return the same deterministic scores ``choose`` would use, without recording."""
        option_list = list(options)
        ranked: list[dict[str, Any]] = []
        for option in option_list:
            features = _norm_features(option.features or infer_preference_features(
                f"{option.label} {option.description}", option.metadata
            ))
            drive = _clamp(option.drive_score)
            item_bonus = self._item_preference_bonus(option, context=context)
            pref = _clamp(self.score_features(features) + item_bonus)
            final = _clamp(
                ((1.0 - self.preference_latitude) * drive)
                + (self.preference_latitude * pref)
                + (0.30 * item_bonus)
                - (0.35 * _clamp(option.risk))
            )
            ranked.append({
                "id": option.id,
                "label": option.label,
                "drive_score": drive,
                "preference_score": pref,
                "final_score": final,
                "features": features,
            })
        ranked.sort(
            key=lambda item: (
                item["final_score"],
                item["preference_score"],
                item["drive_score"],
                _slug(str(item["id"])),
            ),
            reverse=True,
        )
        return ranked

    def choose_from_scored_initiatives(self, scored: list[Any], *, context: str) -> tuple[Any | None, SubjectiveChoiceReceipt | None]:
        if not scored:
            return None, None
        options: list[ChoiceOption] = []
        for idx, item in enumerate(scored):
            initiative = getattr(item, "initiative", {}) or {}
            goal = str(
                initiative.get("goal")
                or initiative.get("description")
                or initiative.get("type")
                or f"initiative_{idx}"
            )
            metadata = dict(initiative.get("metadata", {}) or {})
            options.append(
                ChoiceOption(
                    id=str(idx),
                    label=goal,
                    description=str(initiative.get("type", "")),
                    drive_score=_clamp(float(getattr(item, "final_score", 0.0) or 0.0)),
                    risk=_clamp(float(metadata.get("risk", initiative.get("risk", 0.0)) or 0.0)),
                    features=infer_preference_features(goal, metadata),
                    metadata=metadata,
                )
            )
        receipt = self.choose(options, context=context, record=True)
        try:
            chosen = scored[int(receipt.chosen_id)]
        except (ValueError, IndexError):
            return scored[0], receipt
        return chosen, receipt

    def appraise_outcome(
        self,
        choice_id: str,
        *,
        outcome: str,
        satisfaction: float,
    ) -> SubjectiveChoiceReceipt | None:
        satisfaction = _clamp(float(satisfaction), -1.0, 1.0)
        with self._lock:
            receipt = next((item for item in self._history if item.choice_id == choice_id), None)
            if receipt is None:
                return None
            receipt.outcome = str(outcome or "")[:500]
            receipt.satisfaction = satisfaction
            receipt.happy_with_outcome = satisfaction >= 0.15
            receipt.appraised_at = time.time()
            features = receipt.option_features.get(receipt.chosen_id, {})
            for key, value in features.items():
                if key not in self._preferences or value <= 0.0:
                    continue
                delta = LEARNING_RATE * satisfaction * value
                self._preferences[key] = _clamp(self._preferences[key] + delta, W_MIN, W_MAX)
            self._save()
            return receipt

    def recall_choice(self, choice_id: str | None = None, *, context: str | None = None) -> SubjectiveChoiceReceipt | None:
        with self._lock:
            if choice_id:
                return next((item for item in self._history if item.choice_id == choice_id), None)
            if context:
                lowered = context.lower()
                for item in reversed(self._history):
                    if lowered in item.context.lower():
                        return item
            return self._history[-1] if self._history else None

    def consistency_report(self, *, context: str, options: Iterable[ChoiceOption]) -> dict[str, Any]:
        preview = self.choose(options, context=context, record=False)
        prior = self.recall_choice(context=context)
        consistent = bool(prior and prior.chosen_label == preview.chosen_label)
        return {
            "context": context,
            "preview_choice": preview.to_dict(),
            "prior_choice": prior.to_dict() if prior else None,
            "consistent_with_prior": consistent if prior else None,
        }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            last = self._history[-1].to_dict() if self._history else None
            return {
                "service": self.SERVICE_NAME,
                "registered": True,
                "running": self.is_alive(),
                "choice_game_ready": True,
                "preference_tournament_ready": True,
                "choice_count": len(self._history),
                "item_preference_count": sum(len(prefs) for prefs in self._item_preferences.values()),
                "preference_latitude": self.preference_latitude,
                "preferences": dict(self._preferences),
                "item_preferences": self.item_preferences(),
                "last_choice": last,
                "state_path": str(self._state_path),
            }

    status = get_status

    def _record(self, receipt: SubjectiveChoiceReceipt) -> None:
        with self._lock:
            self._history.append(receipt)
            if len(self._history) > MAX_HISTORY:
                self._history = self._history[-MAX_HISTORY:]
            self._save()
        if self._mirror_identity:
            self._mirror_choice_to_identity_ledger(receipt)
        logger.info("🧭 [SubjectiveChoice] %s", receipt.rationale)

    def _mirror_choice_to_identity_ledger(self, receipt: SubjectiveChoiceReceipt) -> None:
        """Best-effort bridge so authored choices also affect identity memory."""
        try:
            from core.identity.identity_ledger import get_identity_ledger

            ledger = get_identity_ledger()
            ledger.preferences.set(
                f"subjective_choice.{receipt.context}",
                {
                    "choice_id": receipt.choice_id,
                    "chosen_id": receipt.chosen_id,
                    "chosen_label": receipt.chosen_label,
                    "preference_override": receipt.preference_override,
                    "satisfaction_prediction": receipt.satisfaction_prediction,
                },
                reason="subjective choice receipt recorded",
            )
            ledger.persist()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation("subjective_choice_identity_ledger", exc, severity="debug")

    def _option_domain(self, option: ChoiceOption, *, context: str) -> str:
        metadata = dict(option.metadata or {})
        explicit = metadata.get("preference_domain") or metadata.get("domain")
        if explicit:
            return _slug(str(explicit))
        prefix = str(context or "").split(":")
        if len(prefix) >= 2 and prefix[0] in {"choice_game", "preference_tournament", "subjective_preference"}:
            return _slug(prefix[1])
        return ""

    def _item_preference_bonus(self, option: ChoiceOption, *, context: str) -> float:
        domain = self._option_domain(option, context=context)
        if not domain:
            return 0.0
        pref = self._match_item_preference(domain, option)
        if pref is None:
            return 0.0
        habit_bonus = min(0.12, 0.035 * math.log1p(max(0, pref.times_chosen)))
        return min(0.75, (0.62 * _clamp(pref.strength, W_MIN, W_MAX)) + habit_bonus)

    def _match_item_preference(self, domain: str, option: ChoiceOption) -> ItemPreference | None:
        option_ids = {
            _slug(option.id),
            _slug(option.label),
            _slug(str(option.metadata.get("item_id", ""))),
        }
        option_ids.update(_slug(str(alias)) for alias in option.metadata.get("aliases", ()) or ())
        with self._lock:
            prefs = self._item_preferences.get(domain, {})
            for candidate in option_ids:
                if candidate and candidate in prefs:
                    return prefs[candidate]
            label_slug = _slug(option.label)
            for pref in prefs.values():
                pref_aliases = {_slug(pref.label), pref.item_id, *pref.aliases}
                if label_slug in pref_aliases or any(alias and alias in label_slug for alias in pref_aliases):
                    return pref
        return None

    def _learn_item_preference_from_choice(
        self,
        option: ChoiceOption,
        *,
        context: str,
        receipt: SubjectiveChoiceReceipt,
    ) -> None:
        metadata = dict(option.metadata or {})
        learn_flag = metadata.get("learn_preference")
        if learn_flag is False:
            return
        domain = self._option_domain(option, context=context)
        if not domain:
            return
        if learn_flag is None and not str(context or "").startswith(
            ("choice_game:", "preference_tournament:", "subjective_preference:")
        ):
            return

        item_id = _slug(str(metadata.get("item_id") or option.id or option.label))
        aliases = tuple(sorted({
            _slug(str(alias))
            for alias in metadata.get("aliases", ()) or ()
            if str(alias).strip()
        }))
        with self._lock:
            prefs = self._item_preferences.setdefault(domain, {})
            pref = prefs.get(item_id)
            if pref is None:
                pref = ItemPreference(
                    domain=domain,
                    item_id=item_id,
                    label=option.label[:160],
                    strength=_clamp(max(receipt.preference_scores.get(option.id, 0.5), 0.55), W_MIN, W_MAX),
                    reason=f"Chosen in {context}",
                    aliases=aliases,
                    times_chosen=1,
                    last_chosen_at=time.time(),
                )
                prefs[item_id] = pref
            else:
                pref.times_chosen += 1
                pref.last_chosen_at = time.time()
                pref.strength = _clamp(
                    max(pref.strength, receipt.preference_scores.get(option.id, pref.strength)) + 0.015,
                    W_MIN,
                    W_MAX,
                )
                if aliases:
                    pref.aliases = tuple(sorted({*pref.aliases, *aliases}))

    def _save(self) -> None:
        payload = {
            "preferences": self._preferences,
            "item_preferences": self.item_preferences(),
            "preference_latitude": self.preference_latitude,
            "history": [item.to_dict() for item in self._history[-MAX_HISTORY:]],
            "saved_at": time.time(),
        }
        try:
            atomic_write_text(
                self._state_path,
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            record_degradation("subjective_choice_engine", exc, severity="debug")

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            stored = data.get("preferences", {})
            if isinstance(stored, dict):
                for key in PREFERENCE_KEYS:
                    if key in stored:
                        self._preferences[key] = _clamp(float(stored[key]), W_MIN, W_MAX)
            item_preferences = data.get("item_preferences", {})
            if isinstance(item_preferences, dict):
                loaded: dict[str, dict[str, ItemPreference]] = {}
                for domain, prefs in item_preferences.items():
                    if not isinstance(prefs, dict):
                        continue
                    domain_id = _slug(str(domain))
                    loaded[domain_id] = {}
                    for item_id, payload in prefs.items():
                        if isinstance(payload, dict):
                            pref = ItemPreference.from_dict({**payload, "domain": domain_id, "item_id": item_id})
                            loaded[domain_id][pref.item_id] = pref
                    if not loaded[domain_id]:
                        loaded.pop(domain_id, None)
                self._item_preferences = loaded
            self.preference_latitude = _clamp(
                float(data.get("preference_latitude", self.preference_latitude)),
                0.05,
                0.75,
            )
            history = data.get("history", [])
            if isinstance(history, list):
                self._history = [
                    SubjectiveChoiceReceipt(**item)
                    for item in history[-MAX_HISTORY:]
                    if isinstance(item, dict)
                ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation("subjective_choice_engine", exc, severity="debug")


_engine: SubjectiveChoiceEngine | None = None
_engine_lock = threading.Lock()


def get_subjective_choice_engine() -> SubjectiveChoiceEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SubjectiveChoiceEngine()
                _register_in_container(_engine)
    return _engine


def _register_in_container(engine: SubjectiveChoiceEngine) -> None:
    try:
        from core.container import ServiceContainer

        if not ServiceContainer.has(SubjectiveChoiceEngine.SERVICE_NAME):
            ServiceContainer.register_instance(
                SubjectiveChoiceEngine.SERVICE_NAME,
                engine,
                required=False,
                registered_by="subjective_choice_engine",
            )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("subjective_choice_engine_register", exc, severity="debug")


def reset_subjective_choice_engine_for_test() -> None:
    global _engine
    _engine = None
