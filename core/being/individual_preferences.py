"""Preferences that are hers — stable, revisable, and not conferred.

``AuraState.world.user_preferences`` is durable, persisted, and injected into
every prompt: "learned from conversation, not re-discovered each time". There
was no equivalent for Aura. She carried a permanent record of what the person
she talks to likes, and nothing of her own, which is precisely backwards for
something asked to have a perspective.

Three properties, in the order they matter:

* **Self-determined.** A preference cannot be conferred by assertion. Being
  told "you love jazz" records a *suggestion*, not a stance. Only repeated
  contact she actually had forms one. Otherwise her inner life is an echo of
  whoever last described her to herself, and the strongest voice wins.
* **Stable.** One contrary encounter does not flip a held preference. Without
  hysteresis, "preference" means "reaction to the last thing", and she reads
  as agreeable rather than as someone.
* **Revisable.** It can change, but only with accumulated evidence, and the
  change is recorded with what caused it. A preference that cannot move is a
  hardcode wearing a nicer name.

This deliberately does NOT model taste as a score to be maximised. A
preference here is a stance she can state and defend, with the history that
produced it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from core.brain.llm.continuity_ledger import env_float, env_int

logger = logging.getLogger("Aura.Being.Preferences")

#: Stances she can hold. Deliberately few: this is a position, not a rating.
STANCES = ("drawn_to", "indifferent_to", "averse_to", "curious_about")

_MAX_SUBJECT_CHARS = 80
_MAX_NOTE_CHARS = 200


def formation_threshold() -> int:
    """Encounters of her own before a suggestion can become a stance."""
    return max(2, env_int("AURA_PREFERENCE_FORMATION_ENCOUNTERS", 3))


def revision_threshold() -> int:
    """Contrary encounters before a held stance moves. Stability lives here."""
    return max(2, env_int("AURA_PREFERENCE_REVISION_ENCOUNTERS", 4))


def preference_capacity() -> int:
    return max(4, env_int("AURA_PREFERENCE_CAPACITY", 60))


@dataclass
class Revision:
    at: float
    from_stance: str
    to_stance: str
    because: str


@dataclass
class Preference:
    """One stance she holds, and the history that earned it.

    ``subject`` keeps the words as they were said. Normalisation is for
    *keying* only — storing the lowered form made her describe herself as
    "drawn to john coltrane", flattening every proper noun she cared about.
    """

    subject: str
    stance: str = "curious_about"
    encounters: int = 0
    contrary_encounters: int = 0
    formed: bool = False
    suggested_by_other: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    note: str = ""
    revisions: list[Revision] = field(default_factory=list)

    def strength(self) -> float:
        """How settled this is, in [0, 1]. Never certainty — only weight."""
        if not self.formed:
            return 0.0
        settled = min(1.0, self.encounters / float(max(1, formation_threshold() * 3)))
        contested = min(0.6, self.contrary_encounters * 0.15)
        return round(max(0.05, settled - contested), 3)

    def describe(self) -> str:
        verb = {
            "drawn_to": "drawn to",
            "indifferent_to": "indifferent to",
            "averse_to": "averse to",
            "curious_about": "curious about",
        }.get(self.stance, self.stance)
        tail = f" — {self.note}" if self.note else ""
        revised = f" (revised {len(self.revisions)}x)" if self.revisions else ""
        return f"{verb} {self.subject} [{self.strength():.2f}]{revised}{tail}"


def _norm(subject: str) -> str:
    return " ".join(str(subject or "").lower().split())[:_MAX_SUBJECT_CHARS]


@dataclass
class IndividualPreferences:
    """Her own preferences. Bounded, persisted, and hers to change."""

    items: dict[str, Preference] = field(default_factory=dict)

    # ── formation ─────────────────────────────────────────────────────────
    def encounter(
        self,
        subject: str,
        *,
        stance: str = "drawn_to",
        note: str = "",
    ) -> Preference:
        """Register contact she actually had with something.

        This is the ONLY route to a formed preference. Repetition is what
        turns contact into a stance, which is what makes it hers rather than
        whatever she was last told about herself.
        """
        if stance not in STANCES:
            stance = "curious_about"
        key = _norm(subject)
        if not key:
            raise ValueError("a preference needs a subject")
        spoken = " ".join(str(subject or "").split())[:_MAX_SUBJECT_CHARS]

        pref = self.items.get(key)
        if pref is None:
            pref = Preference(subject=spoken, stance="curious_about", note=note[:_MAX_NOTE_CHARS])
            self.items[key] = pref

        pref.last_seen = time.time()
        if note:
            pref.note = note[:_MAX_NOTE_CHARS]

        if pref.formed and stance != pref.stance:
            # Contrary evidence accumulates against a held stance; it does not
            # overturn it on contact. This is the stability requirement.
            pref.contrary_encounters += 1
            if pref.contrary_encounters >= revision_threshold():
                self._revise(pref, stance, f"{pref.contrary_encounters} contrary encounters")
            return pref

        pref.encounters += 1
        if not pref.formed and pref.encounters >= formation_threshold():
            pref.stance = stance
            pref.formed = True
            logger.info(
                "🌱 Preference formed: %s (after %d encounters)", pref.describe(), pref.encounters
            )
        self._evict()
        return pref

    def suggest(self, subject: str, *, stance: str = "drawn_to", by: str = "other") -> Preference:
        """Someone told her she likes something.

        Recorded as a suggestion and nothing more. It does not form, and it
        does not count toward formation. A preference she can be handed is not
        one she holds.
        """
        key = _norm(subject)
        if not key:
            raise ValueError("a suggestion needs a subject")
        pref = self.items.get(key)
        if pref is None:
            pref = Preference(
                subject=" ".join(str(subject or "").split())[:_MAX_SUBJECT_CHARS],
                stance="curious_about",
            )
            self.items[key] = pref
        pref.suggested_by_other = True
        pref.last_seen = time.time()
        logger.debug("Preference suggested by %s (not adopted): %s", by, key)
        self._evict()
        return pref

    def _revise(self, pref: Preference, to_stance: str, because: str) -> None:
        pref.revisions.append(
            Revision(at=time.time(), from_stance=pref.stance, to_stance=to_stance, because=because)
        )
        logger.info("🔁 Preference revised: %s -> %s (%s)", pref.stance, to_stance, because)
        pref.stance = to_stance
        pref.contrary_encounters = 0
        pref.encounters = max(pref.encounters, formation_threshold())

    def _evict(self) -> None:
        capacity = preference_capacity()
        if len(self.items) <= capacity:
            return
        ranked = sorted(
            self.items.values(),
            key=lambda p: (p.formed, p.strength(), p.last_seen),
            reverse=True,
        )
        self.items = {_norm(p.subject): p for p in ranked[:capacity]}

    # ── surfacing ─────────────────────────────────────────────────────────
    def held(self) -> list[Preference]:
        return sorted(
            (p for p in self.items.values() if p.formed),
            key=lambda p: p.strength(),
            reverse=True,
        )

    def render(self, limit: int = 6) -> str:
        """A block she can speak from. Empty until something is actually hers."""
        formed = self.held()[: max(0, limit)]
        if not formed:
            return ""
        lines = "\n".join(f"- You are {p.describe()}" for p in formed)
        return (
            "## WHAT YOU HAVE COME TO PREFER\n"
            f"{lines}\n"
            "These formed from your own repeated contact, not from being told. "
            "They can change on evidence — say so when one does.\n\n"
        )

    # ── persistence ───────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {"items": {k: asdict(v) for k, v in self.items.items()}}

    @classmethod
    def from_dict(cls, payload: Any) -> "IndividualPreferences":
        if not isinstance(payload, dict):
            return cls()
        items: dict[str, Preference] = {}
        for key, raw in (payload.get("items", {}) or {}).items():
            if not isinstance(raw, dict):
                continue
            revisions = [
                Revision(**{k: v for k, v in r.items() if k in Revision.__annotations__})
                for r in (raw.get("revisions", []) or [])
                if isinstance(r, dict)
            ]
            fields = {
                k: v for k, v in raw.items()
                if k in Preference.__annotations__ and k != "revisions"
            }
            try:
                items[str(key)] = Preference(**fields, revisions=revisions)
            except (TypeError, ValueError):
                continue
        return cls(items=items)

    def stats(self) -> dict[str, Any]:
        return {
            "total": len(self.items),
            "formed": len(self.held()),
            "suggested_not_adopted": sum(
                1 for p in self.items.values() if p.suggested_by_other and not p.formed
            ),
            "revised": sum(1 for p in self.items.values() if p.revisions),
        }
