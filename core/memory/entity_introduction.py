"""core/memory/entity_introduction.py — how Aura meets someone new, and how she keeps track of who "he" is.

The first version of entity discovery looked for capitalised words. That is
orthography, not evidence: it misses ``my dog rex``, invents entities out of
sentence-initial words, and — worst — it has no opinion about *what kind of
thing* it found or *how sure it is*. So the memory could only ever recognise
entities somebody had introduced by hand.

This module replaces that with two mechanisms.

**Introduction by licensed pattern.** Humans introduce entities into discourse
with recognisable constructions: "my friend Sarah", "this is Dave", "the
workshop", "Rex is my dog". Each pattern here carries three things a regex over
capital letters cannot: the **kind** it implies (a thing introduced as "my
friend X" is a person), a **reliability** reflecting how often that construction
really does introduce an entity, and a **receipt** — the matched text — so Aura
can later say *how she came to know someone*. The pattern is the evidence.

**Coreference by salience.** "What is he working on?" contains no name. A
memory that cannot resolve that pronoun has to fall back on stuffing keywords
into a query and hoping. Recently-active entities are tracked with decaying
salience, and pronouns bind to the most salient entity of a compatible kind, so
the pronoun resolves to an actual entity id before any retrieval happens.

Two guards, both deliberate:

* **Aura's own output cannot introduce entities.** Only text from the user or
  from a verified observation may create one. Without that rule a single
  confabulated name in her own generation becomes a permanent member of her
  social graph, and every later mention "confirms" it.
* **Introduction is bounded per utterance.** A pasted document cannot mint
  hundreds of entities in one turn.

The pattern lexicon is English-only and says so. That is a real limitation, not
a hidden one: :func:`detect_introductions` reports what it matched, and an
entity Aura never introduces is simply one she does not claim to know.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from core.memory.associative_entity_memory import (
    Entity,
    EntityKind,
    Provenance,
    get_associative_entity_memory,
    normalize_name,
)
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.EntityIntroduction")

_INTRO_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError)

#: Reliability below which a candidate is reported but never introduced.
#: A construction that is right only half the time should not be able to add a
#: permanent member to Aura's world.
MIN_INTRODUCTION_RELIABILITY = 0.6

#: Most entities one utterance may introduce. A pasted document should not be
#: able to mint a hundred people.
MAX_INTRODUCTIONS_PER_UTTERANCE = 4

#: Sources trusted to introduce. Aura's own generated text is deliberately
#: absent: a confabulated name must not become a permanent entity that later
#: mentions then "confirm".
TRUSTED_INTRODUCTION_SOURCES = frozenset({
    "user", "conversation", "observation", "verified_observation", "operator",
})


# ── the lexicon that turns a construction into a kind ────────────────────────

_PERSON_ROLES = (
    "friend", "brother", "sister", "mother", "father", "mom", "dad", "parent",
    "wife", "husband", "partner", "spouse", "son", "daughter", "child", "kid",
    "colleague", "coworker", "boss", "manager", "teacher", "student", "neighbor",
    "neighbour", "cousin", "aunt", "uncle", "grandmother", "grandfather",
    "roommate", "landlord", "doctor", "therapist", "client", "mentor",
)
_ANIMAL_ROLES = (
    "dog", "cat", "puppy", "kitten", "pet", "bird", "horse", "rabbit", "fish",
    "hamster", "snake", "lizard", "turtle", "parrot",
)
_PLACE_NOUNS = (
    "workshop", "office", "kitchen", "garage", "studio", "lab", "laboratory",
    "bedroom", "basement", "attic", "garden", "yard", "shop", "cabin", "house",
    "apartment", "room", "warehouse", "server room", "living room",
)
_THING_NOUNS = (
    "laptop", "computer", "phone", "car", "truck", "bike", "bicycle", "guitar",
    "camera", "printer", "server", "machine", "desk", "chair", "app",
    "application", "project", "script", "tool",
)
_ORG_NOUNS = (
    "company", "team", "startup", "firm", "agency", "school", "university",
    "college", "hospital", "clinic", "band", "club",
)

_ROLE_TO_KIND: dict[str, EntityKind] = {}
for _r in _PERSON_ROLES:
    _ROLE_TO_KIND[_r] = EntityKind.PERSON
for _r in _ANIMAL_ROLES:
    # No ANIMAL kind exists and inventing one would ripple through identity;
    # a pet is closer to a person than to an object in how Aura relates to it,
    # so it is filed as OTHER with its role recorded as a trait rather than
    # being flattened into THING.
    _ROLE_TO_KIND[_r] = EntityKind.OTHER
for _r in _PLACE_NOUNS:
    _ROLE_TO_KIND[_r] = EntityKind.PLACE
for _r in _THING_NOUNS:
    _ROLE_TO_KIND[_r] = EntityKind.THING
for _r in _ORG_NOUNS:
    _ROLE_TO_KIND[_r] = EntityKind.ORGANIZATION

_ALL_ROLES = sorted(_ROLE_TO_KIND, key=len, reverse=True)
_ROLE_ALT = "(?i:" + "|".join(re.escape(r) for r in _ALL_ROLES) + ")"
_THING_ALT = "(?i:" + "|".join(
    re.escape(r) for r in sorted(_THING_NOUNS, key=len, reverse=True)) + ")"
_PLACE_ALT = "(?i:" + "|".join(
    re.escape(r) for r in sorted(_PLACE_NOUNS, key=len, reverse=True)) + ")"

# A proper name is CASE-SENSITIVE on purpose. The patterns below therefore must
# NOT be compiled with re.IGNORECASE — doing so silently erases this
# distinction, and "my friend Sarah helped me" starts capturing "Sarah helped"
# as the name while "my laptop died" captures "died". Fixed keywords carry
# their own scoped (?i:...) flag instead.
_NAME = r"([A-Z][\w'-]{1,30}(?:\s+[A-Z][\w'-]{1,30})?)"
_MY = r"(?i:my|our)"


@dataclass(frozen=True)
class IntroductionPattern:
    """One construction that licenses meeting something new."""

    name: str
    regex: re.Pattern[str]
    #: Which capture group holds the entity's name.
    name_group: int
    #: Which group (if any) holds the role noun that implies the kind.
    role_group: int | None
    #: Kind used when no role group is present.
    default_kind: EntityKind
    #: How often this construction really does introduce an entity. These are
    #: ordered by specificity — an explicit "this is X" is far more reliable
    #: than a bare capitalised token.
    reliability: float
    gloss: str


_PATTERNS: tuple[IntroductionPattern, ...] = (
    IntroductionPattern(
        "possessive_role_name",
        re.compile(rf"\b{_MY}\s+({_ROLE_ALT})\s+{_NAME}"),
        name_group=2, role_group=1, default_kind=EntityKind.PERSON,
        reliability=0.95,
        gloss="introduced as a named relation ('my friend Sarah')",
    ),
    IntroductionPattern(
        "name_appositive_role",
        re.compile(rf"\b{_NAME}\s*,\s*{_MY}\s+({_ROLE_ALT})\b"),
        name_group=1, role_group=2, default_kind=EntityKind.PERSON,
        reliability=0.92,
        gloss="introduced by apposition ('Sarah, my sister')",
    ),
    IntroductionPattern(
        "copula_role",
        re.compile(rf"\b{_NAME}\s+(?i:is)\s+(?i:my|our|a|an|the)\s+({_ROLE_ALT})\b"),
        name_group=1, role_group=2, default_kind=EntityKind.OTHER,
        reliability=0.88,
        gloss="introduced by predication ('Rex is my dog')",
    ),
    IntroductionPattern(
        "explicit_presentation",
        re.compile(rf"\b(?i:this is|meet|say hello to)\s+{_NAME}"),
        name_group=1, role_group=None, default_kind=EntityKind.PERSON,
        reliability=0.90,
        gloss="explicitly presented ('this is Dave')",
    ),
    # For these two the role noun IS the entity ("the workshop"), so they are
    # restricted to place/thing vocabularies — a person role must never mint an
    # entity literally named "friend".
    IntroductionPattern(
        "definite_place",
        re.compile(rf"\b(?i:in|at|to|from)\s+(?i:the|my|our)\s+({_PLACE_ALT})\b"),
        name_group=1, role_group=1, default_kind=EntityKind.PLACE,
        reliability=0.70,
        gloss="referred to as a definite location ('in the workshop')",
    ),
    IntroductionPattern(
        "possessive_thing",
        re.compile(rf"\b{_MY}\s+({_THING_ALT})\b"),
        name_group=1, role_group=1, default_kind=EntityKind.THING,
        reliability=0.65,
        gloss="referred to as a possession ('my laptop')",
    ),
)


@dataclass
class Introduction:
    """A candidate entity, the construction that licensed it, and the receipt."""

    surface: str
    kind: EntityKind
    reliability: float
    pattern: str
    gloss: str
    matched_text: str
    role: str = ""

    @property
    def admissible(self) -> bool:
        return self.reliability >= MIN_INTRODUCTION_RELIABILITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "kind": self.kind.value,
            "reliability": round(self.reliability, 3),
            "pattern": self.pattern,
            "gloss": self.gloss,
            "matched_text": self.matched_text[:160],
            "role": self.role,
            "admissible": self.admissible,
        }


_NOT_A_NAME = {
    "i", "it", "he", "she", "they", "we", "you", "the", "a", "an", "this",
    "that", "there", "here", "what", "who", "when", "where", "how", "why",
    "own", "new", "old", "other", "same", "next", "last", "first",
}


def detect_introductions(text: str) -> list[Introduction]:
    """Find constructions in which something is being introduced.

    Reports every match with its reliability, including inadmissible ones, so a
    caller can see what was considered and rejected rather than only what
    survived.
    """
    found: dict[tuple[str, str], Introduction] = {}
    body = str(text or "")
    for pattern in _PATTERNS:
        for match in pattern.regex.finditer(body):
            try:
                surface = (match.group(pattern.name_group) or "").strip()
            except IndexError:
                continue
            norm = normalize_name(surface)
            if not norm or norm in _NOT_A_NAME:
                continue
            role = ""
            kind = pattern.default_kind
            if pattern.role_group is not None:
                try:
                    role = (match.group(pattern.role_group) or "").strip().lower()
                except IndexError:
                    role = ""
                if role:
                    kind = _ROLE_TO_KIND.get(role, pattern.default_kind)
            # When the role IS the name ("in the workshop"), the entity is the
            # place itself rather than something called by that role.
            if pattern.name_group == pattern.role_group:
                surface = role or surface
                norm = normalize_name(surface)
            key = (norm, kind.value)
            candidate = Introduction(
                surface=surface, kind=kind, reliability=pattern.reliability,
                pattern=pattern.name, gloss=pattern.gloss,
                matched_text=match.group(0), role=role,
            )
            # Most reliable construction wins when several match one name.
            existing = found.get(key)
            if existing is None or candidate.reliability > existing.reliability:
                found[key] = candidate
    return sorted(found.values(), key=lambda i: i.reliability, reverse=True)


def introduce_entities(
    text: str,
    *,
    source: str = "user",
    evidence_id: str = "",
    memory: Any = None,
    max_new: int = MAX_INTRODUCTIONS_PER_UTTERANCE,
) -> list[Entity]:
    """Meet whatever this utterance introduces, and record how.

    Returns only entities that were newly learned. Each carries a trait naming
    the role it was introduced under and a receipt of the construction that
    licensed it, so Aura can answer "how do you know Sarah?" with the actual
    moment rather than a guess.
    """
    if source not in TRUSTED_INTRODUCTION_SOURCES:
        # Aura's own output is not evidence that a person exists.
        logger.debug("Refused introduction from untrusted source %r.", source)
        return []

    mem = memory or get_associative_entity_memory()
    if not getattr(mem, "available", False):
        return []

    learned: list[Entity] = []
    for intro in detect_introductions(text):
        if len(learned) >= max(0, int(max_new)):
            break
        if not intro.admissible:
            continue
        try:
            if mem.resolve(intro.surface, kind=intro.kind, create=False) is not None:
                continue                      # already known; not a new meeting
            entity = mem.resolve(intro.surface, kind=intro.kind, create=True)
            if entity is None:
                continue
            provenance = Provenance(
                source=source,
                evidence_id=evidence_id,
                detail=f"{intro.gloss}: {intro.matched_text}"[:500],
            )
            # The introduction is itself the first thing known about them.
            mem.observe(
                entity, kind="fact", key="was introduced as",
                value=intro.matched_text[:200],
                strength=1.0, evidence_weight=1.0,
                provenance=provenance,
            )
            if intro.role:
                mem.note_trait(
                    entity, intro.role, strength=intro.reliability,
                    evidence_weight=1.0, provenance=provenance,
                )
            learned.append(entity)
            logger.info("🧠 Met a new %s: %r (%s)",
                        intro.kind.value, entity.canonical_name, intro.pattern)
        except _INTRO_ERRORS as exc:
            record_degradation("entity_introduction", exc, severity="warning",
                               action="entity not introduced")
    return learned


# ── coreference: keeping track of who "he" is ───────────────────────────────

_PRONOUN_KINDS: dict[str, tuple[EntityKind, ...]] = {
    "he": (EntityKind.PERSON, EntityKind.OTHER),
    "him": (EntityKind.PERSON, EntityKind.OTHER),
    "his": (EntityKind.PERSON, EntityKind.OTHER),
    "she": (EntityKind.PERSON, EntityKind.OTHER),
    "her": (EntityKind.PERSON, EntityKind.OTHER),
    "hers": (EntityKind.PERSON, EntityKind.OTHER),
    "they": (EntityKind.PERSON, EntityKind.ORGANIZATION, EntityKind.OTHER),
    "them": (EntityKind.PERSON, EntityKind.ORGANIZATION, EntityKind.OTHER),
    "their": (EntityKind.PERSON, EntityKind.ORGANIZATION, EntityKind.OTHER),
    "it": (EntityKind.THING, EntityKind.PLACE, EntityKind.ORGANIZATION),
    "its": (EntityKind.THING, EntityKind.PLACE, EntityKind.ORGANIZATION),
    "there": (EntityKind.PLACE,),
}

_PRONOUN_RE = re.compile(
    r"\b(" + "|".join(sorted(_PRONOUN_KINDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


@dataclass
class _SalientEntity:
    entity_id: str
    kind: EntityKind
    name: str
    last_at: float
    activations: int = 1


class DiscourseSalience:
    """Which entities are currently 'in the room', and how strongly.

    Salience decays: an entity mentioned twenty turns ago should not capture a
    pronoun from an entity mentioned in the last sentence. The half-life is in
    *turns* rather than seconds because discourse reference tracks conversation
    structure, not wall-clock time.
    """

    def __init__(self, *, turn_half_life: float = 3.0, capacity: int = 32) -> None:
        self._turn_half_life = max(0.5, float(turn_half_life))
        self._capacity = max(4, int(capacity))
        self._entities: dict[str, _SalientEntity] = {}
        self._turn = 0
        self._lock = checked_lock("entity_introduction.instance", reentrant=True)

    def begin_turn(self) -> None:
        with self._lock:
            self._turn += 1

    def note(self, entity: Entity) -> None:
        """Mark an entity as active in the current turn."""
        with self._lock:
            existing = self._entities.get(entity.entity_id)
            if existing is None:
                self._entities[entity.entity_id] = _SalientEntity(
                    entity_id=entity.entity_id, kind=entity.kind,
                    name=entity.canonical_name, last_at=float(self._turn),
                )
            else:
                existing.last_at = float(self._turn)
                existing.activations += 1
            if len(self._entities) > self._capacity:
                stalest = min(self._entities.values(), key=lambda s: s.last_at)
                self._entities.pop(stalest.entity_id, None)

    def salience(self, entity_id: str) -> float:
        with self._lock:
            item = self._entities.get(entity_id)
            if item is None:
                return 0.0
            age = max(0.0, self._turn - item.last_at)
            return 0.5 ** (age / self._turn_half_life)

    def resolve_pronouns(self, text: str) -> dict[str, _SalientEntity]:
        """Bind pronouns in this text to the most salient compatible entity.

        Returns only bindings that could actually be made. An unbound pronoun
        is reported by its absence rather than guessed at — resolving "he" to
        whoever happens to be in memory would be worse than not resolving it.
        """
        bindings: dict[str, _SalientEntity] = {}
        with self._lock:
            if not self._entities:
                return bindings
            for match in _PRONOUN_RE.finditer(str(text or "")):
                pronoun = match.group(1).lower()
                if pronoun in bindings:
                    continue
                compatible = _PRONOUN_KINDS.get(pronoun, ())
                best: _SalientEntity | None = None
                best_score = 0.0
                for item in self._entities.values():
                    if item.kind not in compatible:
                        continue
                    age = max(0.0, self._turn - item.last_at)
                    score = 0.5 ** (age / self._turn_half_life)
                    if score > best_score:
                        best, best_score = item, score
                # A binding this weak is a guess; leaving the pronoun unresolved
                # is the more honest outcome.
                if best is not None and best_score >= 0.25:
                    bindings[pronoun] = best
        return bindings

    def active(self, limit: int = 8) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(self._entities.values(),
                           key=lambda s: s.last_at, reverse=True)[:limit]
            return [{
                "entity_id": i.entity_id,
                "name": i.name,
                "kind": i.kind.value,
                "salience": round(self.salience(i.entity_id), 4),
                "activations": i.activations,
            } for i in items]

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()


_SALIENCE: DiscourseSalience | None = None
_SALIENCE_LOCK = checked_lock("entity_introduction.salience")


def get_discourse_salience() -> DiscourseSalience:
    global _SALIENCE
    if _SALIENCE is not None:
        return _SALIENCE
    with _SALIENCE_LOCK:
        if _SALIENCE is None:
            _SALIENCE = DiscourseSalience()
        return _SALIENCE


__all__ = [
    "DiscourseSalience",
    "Introduction",
    "IntroductionPattern",
    "MAX_INTRODUCTIONS_PER_UTTERANCE",
    "MIN_INTRODUCTION_RELIABILITY",
    "TRUSTED_INTRODUCTION_SOURCES",
    "detect_introductions",
    "get_discourse_salience",
    "introduce_entities",
]
