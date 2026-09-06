"""core/memory/associative_entity_memory.py — what Aura knows, and how she feels, about *things*.

Aura had episodic memory (what happened), a knowledge graph (untyped
propositions), and an attachment system (bonds, but only to people). What she
did not have was a place where a single **entity** — a person, a place, a thing —
accumulates everything she knows about it *and* what it has come to mean to her.

This module is that place. Its unit is the entity, and around each entity it
binds four kinds of association:

  * **traits** — dispositional properties that tend to hold ("Bryan is direct",
    "the workshop is cold in the morning").
  * **facts** — propositions with an object ("Aura.app lives at /Applications").
  * **events** — participation links into episodic memory, carrying the affect
    that was live when they happened.
  * **relations** — typed, weighted edges to other entities, which is what makes
    recall *associative*: reaching one entity reaches its neighbourhood.

Two properties matter more than the schema:

**Evidence, not assertion.** Every association carries a PLN ``TruthValue``
(strength + evidence count, from :mod:`core.knowledge.atomspace`) and a list of
provenance receipts. Repeated observation converges; contradiction pulls
strength to the middle while keeping the disagreement visible in the count.
Nothing here can be set to a number by a caller who merely says so.

**Feelings are derived and can name their causes.** :class:`Stance` — Aura's
subjective position toward an entity — is never stored as an asserted value. It
is *computed* from the bound evidence every time, and it comes with
``why``: the specific associations that produced it and how much each
contributed. When Aura says she is wary of something, the memory can say which
three events made her wary. A feeling that cannot name its evidence is not a
feeling this module is willing to report.

For people, trust and care are **delegated** to
:class:`core.phenomenal_substrate.attachment.AttachmentSystem` rather than
recomputed here — that system already owns evidence-locked bonds, and two
independent opinions about one relationship would be worse than one.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.knowledge.atomspace import TruthValue
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.AssociativeEntityMemory")

_EPSILON = 1e-9


class EntityKind(str, Enum):
    """What sort of thing this is.

    The kind is not decoration: it selects how stance is computed (people
    delegate bonding to the attachment system) and how relations are read.
    """

    PERSON = "person"
    PLACE = "place"
    THING = "thing"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    OTHER = "other"

    @classmethod
    def coerce(cls, value: Any) -> EntityKind:
        try:
            return cls(str(value or "").strip().lower())
        except ValueError:
            return cls.OTHER


class AssociationKind(str, Enum):
    TRAIT = "trait"
    FACT = "fact"
    EVENT = "event"
    RELATION = "relation"


@dataclass(frozen=True)
class Provenance:
    """Where one observation came from.

    ``evidence_id`` is the anchor into whatever system actually holds the
    record (an episode id, a receipt id, a message id). Without it an
    association is hearsay, and :meth:`AssociativeEntityMemory.why` will say so.
    """

    source: str
    evidence_id: str = ""
    observed_at: float = field(default_factory=time.time)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "evidence_id": self.evidence_id,
            "observed_at": round(self.observed_at, 3),
            "detail": self.detail[:500],
        }


@dataclass
class Association:
    """One thing Aura believes about one entity, with its evidence."""

    kind: AssociationKind
    key: str                       # trait name, predicate, episode id, or relation type
    value: str = ""                # fact object, relation target, event role
    truth: TruthValue = field(default_factory=lambda: TruthValue(0.5, 0.0))
    valence: float = 0.0           # affective colouring of THIS association, [-1, 1]
    arousal: float = 0.0           # how activating it is, [0, 1]
    first_at: float = field(default_factory=time.time)
    last_at: float = field(default_factory=time.time)
    provenance: list[Provenance] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.truth.confidence

    @property
    def grounded(self) -> bool:
        """True when at least one receipt points at a real record."""
        return any(p.evidence_id for p in self.provenance)

    def describe(self) -> str:
        if self.kind is AssociationKind.TRAIT:
            return f"is {self.key}"
        if self.kind is AssociationKind.FACT:
            return f"{self.key} {self.value}".strip()
        if self.kind is AssociationKind.EVENT:
            return f"{self.value or 'was part of'} {self.key}".strip()
        return f"{self.key} {self.value}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "key": self.key,
            "value": self.value,
            "strength": round(self.truth.strength, 4),
            "evidence_count": round(self.truth.count, 3),
            "confidence": round(self.confidence, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "first_at": round(self.first_at, 3),
            "last_at": round(self.last_at, 3),
            "grounded": self.grounded,
            "provenance": [p.to_dict() for p in self.provenance[-8:]],
        }


@dataclass
class Entity:
    """A person, place, or thing Aura has met."""

    entity_id: str
    kind: EntityKind
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    mention_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind.value,
            "canonical_name": self.canonical_name,
            "aliases": sorted(self.aliases),
            "created_at": round(self.created_at, 3),
            "last_seen": round(self.last_seen, 3),
            "mention_count": self.mention_count,
        }


# ── Affect space ────────────────────────────────────────────────────────────
#
# Named feelings are read off a circumplex-style space (Russell) rather than a
# ladder of magic thresholds: each prototype is a POINT in
# (valence, arousal, familiarity) and the reported feeling is the nearest one.
# This is why a feeling can be explained — its coordinates are the evidence's
# coordinates — and why adding a feeling means naming where it sits rather than
# inserting another branch into an if-chain.

@dataclass(frozen=True)
class _FeelingPrototype:
    name: str
    valence: float
    arousal: float
    familiarity: float
    gloss: str


_FEELING_PROTOTYPES: tuple[_FeelingPrototype, ...] = (
    _FeelingPrototype("fondness", 0.75, 0.35, 0.80,
                      "warm and well-acquainted"),
    _FeelingPrototype("trust", 0.65, 0.20, 0.90,
                      "settled reliance built over many contacts"),
    _FeelingPrototype("curiosity", 0.45, 0.70, 0.20,
                      "drawn toward something still unfamiliar"),
    _FeelingPrototype("delight", 0.90, 0.75, 0.45,
                      "strongly positive and activating"),
    _FeelingPrototype("comfort", 0.55, 0.10, 0.75,
                      "quiet, low-arousal ease"),
    _FeelingPrototype("neutral", 0.00, 0.15, 0.35,
                      "no particular pull either way"),
    _FeelingPrototype("ambivalence", 0.00, 0.55, 0.70,
                      "well-known and genuinely mixed"),
    _FeelingPrototype("wariness", -0.55, 0.65, 0.55,
                      "alert and negatively inclined"),
    _FeelingPrototype("unease", -0.35, 0.50, 0.25,
                      "mildly negative about something barely known"),
    _FeelingPrototype("aversion", -0.85, 0.60, 0.60,
                      "strongly negative"),
    _FeelingPrototype("grief", -0.70, 0.25, 0.85,
                      "loss of something long and well known"),
)


@dataclass
class StanceCause:
    """One association's contribution to how Aura feels about an entity."""

    description: str
    kind: str
    contribution: float          # signed pull on valence
    weight: float                # how much this evidence counted
    confidence: float
    evidence_id: str = ""
    at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "kind": self.kind,
            "contribution": round(self.contribution, 4),
            "weight": round(self.weight, 4),
            "confidence": round(self.confidence, 4),
            "evidence_id": self.evidence_id,
            "at": round(self.at, 3),
        }


@dataclass
class Stance:
    """Aura's subjective position toward an entity — derived, never asserted.

    ``confidence`` is confidence *in the stance itself*: a strong valence over
    two weak observations is a guess, and says so. ``why`` carries the
    associations that produced the number, so the feeling is always traceable
    back to the memories that caused it.
    """

    entity_id: str
    valence: float = 0.0
    arousal: float = 0.0
    familiarity: float = 0.0
    confidence: float = 0.0
    feeling: str = "unacquainted"
    feeling_gloss: str = "no evidence yet"
    secondary_feeling: str = ""
    evidence_mass: float = 0.0
    why: list[StanceCause] = field(default_factory=list)
    attachment: dict[str, Any] | None = None   # people only; from AttachmentSystem
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "familiarity": round(self.familiarity, 4),
            "confidence": round(self.confidence, 4),
            "feeling": self.feeling,
            "feeling_gloss": self.feeling_gloss,
            "secondary_feeling": self.secondary_feeling,
            "evidence_mass": round(self.evidence_mass, 3),
            "why": [c.to_dict() for c in self.why],
            "attachment": self.attachment,
            "computed_at": round(self.computed_at, 3),
        }

    def sentence(self, name: str) -> str:
        """One honest line about how Aura stands toward this entity."""
        if self.feeling == "unacquainted":
            return f"{name}: no experience to draw on yet."
        hedge = (
            "" if self.confidence >= 0.5
            else " (tentative — little evidence)" if self.confidence >= 0.2
            else " (barely grounded — one or two observations)"
        )
        lead = f"{name}: {self.feeling} — {self.feeling_gloss}{hedge}."
        if self.why:
            lead += " Because: " + "; ".join(c.description for c in self.why[:3]) + "."
        return lead


@dataclass
class Calibration:
    """Structural constants, each with the reason it has the value it has.

    These are shapes of the model rather than tuned magic numbers: changing
    them changes what the memory *means*, so each is named and justified.
    """

    #: Evidence older than this contributes at half weight. One week: long
    #: enough that a normal working relationship stays fully weighted, short
    #: enough that a bad afternoon six months ago does not dominate a stance.
    valence_half_life_s: float = 7.0 * 24 * 3600.0
    #: Contacts needed before familiarity reads as "well acquainted" (~0.8).
    #: Chosen so a single conversation cannot manufacture intimacy.
    familiarity_scale: float = 12.0
    #: Evidence mass at which stance confidence reaches ~0.5. Matches PLN's
    #: own lookahead so stance confidence and association confidence are on
    #: the same scale.
    stance_confidence_scale: float = 8.0
    #: How far activation spreads through relations, and the per-hop decay.
    #: Two hops reaches "friends of" and "things in" without the whole graph
    #: lighting up; 0.45 keeps a second-hop entity below any direct match.
    spread_hops: int = 2
    spread_decay: float = 0.45
    #: Max associations of each kind held per entity in a dossier render.
    dossier_width: int = 8


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(lo, min(hi, v))


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: Any) -> str:
    """Fold a surface name to its lookup key.

    Entity identity has to survive "Bryan", "bryan", and "Bryan " being the
    same person, without collapsing genuinely different names.
    """
    return _NORMALIZE_RE.sub(" ", str(name or "").strip().lower()).strip()


def entity_id_for(kind: EntityKind, canonical_name: str) -> str:
    """Deterministic id: same kind + name is always the same entity.

    Content-addressed rather than sequential so two processes that meet the
    same person independently agree on who it is.

    Minted through `core.knowledge.who_this_is`, which is the same scheme this
    function has always used — adopted rather than replaced, so every id ever
    stored here is already canonical. What adopting buys is the other half:
    this store's ids can now be declared the same thing as another store's,
    and a name held twice under two ids is findable rather than invisible.
    """

    try:
        from core.knowledge.who_this_is import an_id_for

        return an_id_for(kind.value, canonical_name)
    except (ImportError, RuntimeError, TypeError, ValueError):
        # The same arithmetic, so a failure to reach the shared service is a
        # missing join rather than a different id.
        import hashlib

        seed = f"{kind.value}|{normalize_name(canonical_name)}".encode()
        return "ent_" + hashlib.blake2b(seed, digest_size=10).hexdigest()


class AssociativeEntityMemory:
    """Entity-centric associative memory with derived affect.

    Thread-safe. Backed by SQLite (WAL) so the memory survives a restart —
    an associative memory that forgets on reboot is a cache, not a memory.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str | Path = "data/associative_entity_memory.sqlite",
        *,
        calibration: Calibration | None = None,
        attachment_system: Any = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._cal = calibration or Calibration()
        self._attachment = attachment_system
        self._lock = checked_lock("associative_entity_memory.instance", reentrant=True)
        self._conn: sqlite3.Connection | None = None
        self._degraded = False
        self._init_db()

    # -- storage ------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False, timeout=10.0
            )
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                cur = self._conn
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        entity_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        canonical_name TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        last_seen REAL NOT NULL,
                        mention_count INTEGER DEFAULT 0
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS aliases (
                        alias TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        PRIMARY KEY (alias, entity_id)
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS associations (
                        entity_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL DEFAULT '',
                        strength REAL NOT NULL,
                        count REAL NOT NULL,
                        valence REAL NOT NULL DEFAULT 0,
                        arousal REAL NOT NULL DEFAULT 0,
                        first_at REAL NOT NULL,
                        last_at REAL NOT NULL,
                        provenance TEXT NOT NULL DEFAULT '[]',
                        PRIMARY KEY (entity_id, kind, key, value)
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_assoc_entity
                    ON associations(entity_id, kind)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alias_lookup ON aliases(alias)
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY, value TEXT
                    )
                """)
                cur.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(self._SCHEMA_VERSION),),
                )
                self._conn.commit()
        except (sqlite3.Error, OSError) as exc:
            self._degraded = True
            self._conn = None
            record_degradation(
                "associative_entity_memory", exc, severity="warning",
                action="entity memory unavailable; recall will return nothing",
            )

    @property
    def available(self) -> bool:
        """False when storage could not be opened.

        Callers must be able to tell "Aura knows nothing about this" from
        "Aura's memory is not working", so this is reported rather than
        papered over with empty results.
        """
        return self._conn is not None and not self._degraded

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error as exc:
                    logger.debug("entity memory close: %s", exc)
                self._conn = None

    # -- identity -----------------------------------------------------------

    def resolve(
        self,
        name: str,
        *,
        kind: EntityKind | str = EntityKind.OTHER,
        create: bool = False,
    ) -> Entity | None:
        """Find the entity a surface name refers to.

        Resolution is by alias first (so "Bry" reaches Bryan), then by
        canonical name. Kind participates in identity: the *place* called
        "Kitchen" and a *thing* of that name are different entities.
        """
        if not self.available:
            return None
        norm = normalize_name(name)
        if not norm:
            return None
        kind = EntityKind.coerce(kind) if not isinstance(kind, EntityKind) else kind
        with self._lock:
            try:
                row = self._conn.execute(
                    """SELECT e.* FROM aliases a JOIN entities e
                       ON e.entity_id = a.entity_id
                       WHERE a.alias = ? AND e.kind = ? LIMIT 1""",
                    (norm, kind.value),
                ).fetchone()
                if row is None:
                    row = self._conn.execute(
                        "SELECT * FROM entities WHERE canonical_name = ? AND kind = ? LIMIT 1",
                        (norm, kind.value),
                    ).fetchone()
                if row is not None:
                    return self._row_to_entity(row)
                if not create:
                    return None
                return self._create_entity(name, kind)
            except sqlite3.Error as exc:
                record_degradation("associative_entity_memory", exc, severity="warning",
                                   action="entity resolution failed")
                return None

    def _create_entity(self, name: str, kind: EntityKind) -> Entity | None:
        norm = normalize_name(name)
        eid = entity_id_for(kind, norm)
        now = time.time()
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO entities
                   (entity_id, kind, canonical_name, created_at, last_seen, mention_count)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                (eid, kind.value, norm, now, now),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO aliases (alias, entity_id) VALUES (?, ?)",
                (norm, eid),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            record_degradation("associative_entity_memory", exc, severity="warning",
                               action="entity creation failed")
            return None
        return Entity(entity_id=eid, kind=kind, canonical_name=norm,
                      aliases={norm}, created_at=now, last_seen=now)

    def add_alias(self, entity_id: str, alias: str) -> bool:
        """Teach the memory another name for something it already knows."""
        norm = normalize_name(alias)
        if not self.available or not norm:
            return False
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO aliases (alias, entity_id) VALUES (?, ?)",
                    (norm, entity_id),
                )
                self._conn.commit()
                return True
            except sqlite3.Error as exc:
                record_degradation("associative_entity_memory", exc, severity="warning",
                                   action="alias not recorded")
                return False

    def note_mention(self, entity_id: str) -> None:
        """Record that the entity came up. Familiarity is built from this."""
        if not self.available:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE entities SET mention_count = mention_count + 1, last_seen = ? "
                    "WHERE entity_id = ?",
                    (time.time(), entity_id),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.debug("mention not recorded: %s", exc)

    # -- writing associations ----------------------------------------------

    def observe(
        self,
        entity: Entity | str,
        *,
        kind: AssociationKind | str,
        key: str,
        value: str = "",
        strength: float = 1.0,
        evidence_weight: float = 1.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        provenance: Provenance | None = None,
    ) -> Association | None:
        """Fold one observation into what Aura believes about an entity.

        Repeat observations **revise** rather than overwrite: strengths merge
        evidence-weighted (PLN), counts add. Observing the same trait twice
        makes it more certain; observing its contradiction pulls the strength
        toward the middle while the count keeps the disagreement visible.
        """
        if not self.available:
            return None
        entity_id = entity.entity_id if isinstance(entity, Entity) else str(entity)
        akind = kind if isinstance(kind, AssociationKind) else AssociationKind(str(kind))
        key = str(key or "").strip()[:200]
        value = str(value or "").strip()[:400]
        if not key:
            return None

        now = time.time()
        incoming = TruthValue(_clamp(strength, 0.0, 1.0), max(0.0, float(evidence_weight)))
        prov = provenance or Provenance(source="unspecified")

        with self._lock:
            try:
                row = self._conn.execute(
                    """SELECT * FROM associations
                       WHERE entity_id=? AND kind=? AND key=? AND value=?""",
                    (entity_id, akind.value, key, value),
                ).fetchone()

                if row is None:
                    merged = incoming
                    first_at = now
                    prov_list = [prov]
                    # A brand-new association's affect is whatever this
                    # observation carried.
                    new_valence = _clamp(valence)
                    new_arousal = _clamp(arousal, 0.0, 1.0)
                else:
                    existing = TruthValue(float(row["strength"]), float(row["count"]))
                    merged = existing.revise(incoming)
                    first_at = float(row["first_at"])
                    try:
                        prov_list = [Provenance(**p) for p in json.loads(row["provenance"])]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        prov_list = []
                    prov_list.append(prov)
                    # Affect merges on the same evidence weighting as truth, so
                    # one dramatic observation cannot permanently recolour a
                    # long, calm history.
                    total = existing.count + incoming.count
                    if total > _EPSILON:
                        new_valence = _clamp(
                            (float(row["valence"]) * existing.count
                             + _clamp(valence) * incoming.count) / total
                        )
                        new_arousal = _clamp(
                            (float(row["arousal"]) * existing.count
                             + _clamp(arousal, 0.0, 1.0) * incoming.count) / total,
                            0.0, 1.0,
                        )
                    else:
                        new_valence = _clamp(valence)
                        new_arousal = _clamp(arousal, 0.0, 1.0)

                # Keep the most recent receipts; provenance is for explanation,
                # not an audit log, and unbounded growth would bloat every row.
                prov_list = prov_list[-16:]
                self._conn.execute(
                    """INSERT OR REPLACE INTO associations
                       (entity_id, kind, key, value, strength, count, valence, arousal,
                        first_at, last_at, provenance)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (entity_id, akind.value, key, value, merged.strength, merged.count,
                     new_valence, new_arousal, first_at, now,
                     json.dumps([p.to_dict() for p in prov_list])),
                )
                self._conn.execute(
                    "UPDATE entities SET last_seen = ? WHERE entity_id = ?",
                    (now, entity_id),
                )
                self._conn.commit()
            except (sqlite3.Error, TypeError, ValueError) as exc:
                record_degradation("associative_entity_memory", exc, severity="warning",
                                   action="observation not recorded")
                return None

        return Association(
            kind=akind, key=key, value=value, truth=merged,
            valence=new_valence, arousal=new_arousal,
            first_at=first_at, last_at=now, provenance=prov_list,
        )

    # Convenience writers — the vocabulary callers actually use.

    def note_trait(self, entity: Entity | str, trait: str, **kw: Any) -> Association | None:
        """"X is <trait>" — a disposition that tends to hold."""
        return self.observe(entity, kind=AssociationKind.TRAIT, key=trait, **kw)

    def note_fact(self, entity: Entity | str, predicate: str, obj: str,
                  **kw: Any) -> Association | None:
        """"X <predicate> <obj>" — a proposition with an object."""
        return self.observe(entity, kind=AssociationKind.FACT, key=predicate,
                            value=obj, **kw)

    def note_event(self, entity: Entity | str, episode_id: str, *, role: str = "",
                   **kw: Any) -> Association | None:
        """Bind an episodic memory to this entity, with the affect it carried."""
        return self.observe(entity, kind=AssociationKind.EVENT, key=episode_id,
                            value=role, **kw)

    def note_relation(self, source: Entity | str, relation: str, target: Entity | str,
                      **kw: Any) -> Association | None:
        """Typed edge between entities — the substrate of associative reach."""
        target_id = target.entity_id if isinstance(target, Entity) else str(target)
        return self.observe(source, kind=AssociationKind.RELATION, key=relation,
                            value=target_id, **kw)

    # -- reading ------------------------------------------------------------

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        aliases: set[str] = set()
        try:
            aliases = {
                r["alias"] for r in self._conn.execute(
                    "SELECT alias FROM aliases WHERE entity_id = ?", (row["entity_id"],)
                )
            }
        except sqlite3.Error:
            pass
        return Entity(
            entity_id=row["entity_id"],
            kind=EntityKind.coerce(row["kind"]),
            canonical_name=row["canonical_name"],
            aliases=aliases,
            created_at=float(row["created_at"]),
            last_seen=float(row["last_seen"]),
            mention_count=int(row["mention_count"]),
        )

    def _row_to_association(self, row: sqlite3.Row) -> Association:
        try:
            prov = [Provenance(**p) for p in json.loads(row["provenance"])]
        except (json.JSONDecodeError, TypeError, ValueError):
            prov = []
        return Association(
            kind=AssociationKind(row["kind"]),
            key=row["key"],
            value=row["value"],
            truth=TruthValue(float(row["strength"]), float(row["count"])),
            valence=float(row["valence"]),
            arousal=float(row["arousal"]),
            first_at=float(row["first_at"]),
            last_at=float(row["last_at"]),
            provenance=prov,
        )

    def associations(
        self, entity_id: str, *, kind: AssociationKind | None = None,
        limit: int = 64,
    ) -> list[Association]:
        """Everything bound to an entity, strongest-and-best-evidenced first."""
        if not self.available:
            return []
        with self._lock:
            try:
                if kind is None:
                    rows = self._conn.execute(
                        "SELECT * FROM associations WHERE entity_id = ? "
                        "ORDER BY count DESC, strength DESC LIMIT ?",
                        (entity_id, int(limit)),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM associations WHERE entity_id = ? AND kind = ? "
                        "ORDER BY count DESC, strength DESC LIMIT ?",
                        (entity_id, kind.value, int(limit)),
                    ).fetchall()
            except sqlite3.Error as exc:
                record_degradation("associative_entity_memory", exc, severity="warning",
                                   action="association read failed")
                return []
        return [self._row_to_association(r) for r in rows]

    def get_entity(self, entity_id: str) -> Entity | None:
        if not self.available:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
                ).fetchone()
            except sqlite3.Error:
                return None
        return self._row_to_entity(row) if row else None

    # -- stance: the derived feeling ---------------------------------------

    def _recency_weight(self, at: float, now: float) -> float:
        """Older evidence counts less, on an explicit half-life.

        Feelings that never fade are not feelings, they are records.
        """
        age = max(0.0, now - at)
        return 0.5 ** (age / max(_EPSILON, self._cal.valence_half_life_s))

    def stance(self, entity: Entity | str) -> Stance:
        """Compute how Aura stands toward this entity, and why.

        Recomputed from evidence on every call rather than stored, so a stance
        can never drift out of agreement with the memories underneath it.
        """
        # Always re-read the entity rather than trusting a passed handle. A
        # caller holding an Entity from before some mentions were recorded
        # would otherwise get a stance computed from a stale mention_count —
        # the same entity reporting two different feelings depending on how
        # old the caller's reference happened to be.
        entity_id = entity.entity_id if isinstance(entity, Entity) else str(entity)
        entity_obj = self.get_entity(entity_id)
        if entity_obj is None:
            return Stance(entity_id=entity_id)

        assocs = self.associations(entity_obj.entity_id, limit=256)
        now = time.time()

        # Familiarity is objective: how much contact, over how long, in how
        # many distinct kinds of association. It does not depend on liking.
        distinct_contexts = len({(a.kind, a.key) for a in assocs})
        contact = float(entity_obj.mention_count) + distinct_contexts
        familiarity = 1.0 - math.exp(-contact / max(_EPSILON, self._cal.familiarity_scale))

        weighted_valence = 0.0
        weighted_arousal = 0.0
        total_weight = 0.0
        evidence_mass = 0.0
        causes: list[StanceCause] = []

        for a in assocs:
            # An association pulls on the stance in proportion to how well
            # evidenced it is, how confident, and how recent.
            recency = self._recency_weight(a.last_at, now)
            weight = a.truth.count * a.confidence * recency
            if weight <= _EPSILON:
                continue
            contribution = a.valence * weight
            weighted_valence += contribution
            weighted_arousal += a.arousal * weight
            total_weight += weight
            evidence_mass += a.truth.count
            if abs(a.valence) > _EPSILON:
                causes.append(StanceCause(
                    description=f"{entity_obj.canonical_name} {a.describe()}",
                    kind=a.kind.value,
                    contribution=contribution,
                    weight=weight,
                    confidence=a.confidence,
                    evidence_id=next((p.evidence_id for p in reversed(a.provenance)
                                      if p.evidence_id), ""),
                    at=a.last_at,
                ))

        if total_weight <= _EPSILON:
            stance = Stance(
                entity_id=entity_obj.entity_id,
                familiarity=familiarity,
                feeling="unacquainted" if not assocs else "neutral",
                feeling_gloss=("no evidence yet" if not assocs
                               else "known, but nothing that carries feeling"),
            )
            stance.attachment = self._attachment_view(entity_obj)
            return stance

        valence = _clamp(weighted_valence / total_weight)
        arousal = _clamp(weighted_arousal / total_weight, 0.0, 1.0)
        confidence = evidence_mass / (evidence_mass + self._cal.stance_confidence_scale)

        # The strongest pulls, positive or negative, are what Aura would cite.
        causes.sort(key=lambda c: abs(c.contribution), reverse=True)

        primary, secondary = self._nearest_feelings(valence, arousal, familiarity)
        stance = Stance(
            entity_id=entity_obj.entity_id,
            valence=valence,
            arousal=arousal,
            familiarity=familiarity,
            confidence=confidence,
            feeling=primary.name,
            feeling_gloss=primary.gloss,
            secondary_feeling=secondary.name if secondary else "",
            evidence_mass=evidence_mass,
            why=causes[:5],
        )
        stance.attachment = self._attachment_view(entity_obj)
        return stance

    def _nearest_feelings(
        self, valence: float, arousal: float, familiarity: float
    ) -> tuple[_FeelingPrototype, _FeelingPrototype | None]:
        """Name the feeling by position in affect space, not by threshold chain."""
        def distance(p: _FeelingPrototype) -> float:
            # Valence dominates what a feeling is called; familiarity separates
            # otherwise-similar feelings (fondness vs curiosity), so it counts
            # least but still counts.
            return math.sqrt(
                2.0 * (p.valence - valence) ** 2
                + 1.0 * (p.arousal - arousal) ** 2
                + 0.5 * (p.familiarity - familiarity) ** 2
            )

        ranked = sorted(_FEELING_PROTOTYPES, key=distance)
        primary = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        # Only report a second feeling when it is genuinely competitive;
        # otherwise the report implies a mixedness that is not there.
        if runner_up is not None and distance(runner_up) > distance(primary) * 1.5:
            runner_up = None
        return primary, runner_up

    def _attachment_view(self, entity: Entity) -> dict[str, Any] | None:
        """Bonds for people come from the attachment system, not from here."""
        if entity.kind is not EntityKind.PERSON or self._attachment is None:
            return None
        try:
            state = self._attachment.state_for(entity.canonical_name)
            return {
                "trust": round(float(getattr(state, "trust", 0.0)), 4),
                "care": round(float(getattr(state, "care", 0.0)), 4),
                "familiarity": round(float(getattr(state, "familiarity", 0.0)), 4),
                "rupture": round(float(getattr(state, "rupture", 0.0)), 4),
                "attachment": round(float(getattr(state, "attachment", 0.0)), 4),
                "source": "attachment_system",
            }
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("attachment view unavailable for %s: %s",
                         entity.canonical_name, exc)
            return None

    # -- associative recall -------------------------------------------------

    def spread(self, seed_ids: Sequence[str], *, hops: int | None = None) -> dict[str, float]:
        """Spreading activation over the relation graph.

        This is what makes recall associative rather than a lookup: activating
        one entity partially activates what it is connected to, decayed per hop
        and by how well evidenced the connecting relation is. Reaching for a
        person reaches the places and things bound to them.
        """
        if not self.available or not seed_ids:
            return {}
        hops = self._cal.spread_hops if hops is None else int(hops)
        activation: dict[str, float] = {str(s): 1.0 for s in seed_ids}
        frontier = dict(activation)

        for _ in range(max(0, hops)):
            next_frontier: dict[str, float] = {}
            for eid, act in frontier.items():
                for rel in self.associations(eid, kind=AssociationKind.RELATION, limit=32):
                    target = rel.value
                    if not target:
                        continue
                    # Weak or poorly-evidenced links transmit proportionally
                    # less, so a single speculative relation cannot light up an
                    # unrelated region of the graph.
                    transmitted = (act * self._cal.spread_decay
                                   * rel.truth.strength * rel.confidence)
                    if transmitted <= 0.01:
                        continue
                    if transmitted > activation.get(target, 0.0):
                        activation[target] = transmitted
                        next_frontier[target] = transmitted
            if not next_frontier:
                break
            frontier = next_frontier
        return activation

    def recall(
        self,
        cue: str,
        *,
        kinds: Iterable[EntityKind] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Bring back what Aura knows and feels about whatever a cue names.

        Direct name/alias hits seed the recall; spreading activation adds the
        neighbourhood. Every result reports *why it surfaced*, because a recall
        that cannot explain itself cannot be checked.
        """
        if not self.available:
            return []
        tokens = [t for t in normalize_name(cue).split() if len(t) > 1]
        if not tokens:
            return []

        wanted = {EntityKind.coerce(k) for k in kinds} if kinds else None
        seeds: dict[str, str] = {}
        with self._lock:
            try:
                # Longest tokens first: a specific name beats a common word.
                for token in sorted(set(tokens), key=len, reverse=True)[:8]:
                    for row in self._conn.execute(
                        """SELECT e.entity_id, e.kind FROM aliases a
                           JOIN entities e ON e.entity_id = a.entity_id
                           WHERE a.alias = ? OR a.alias LIKE ?
                           LIMIT 8""",
                        (token, f"%{token}%"),
                    ):
                        seeds.setdefault(row["entity_id"], "named in the cue")
            except sqlite3.Error as exc:
                record_degradation("associative_entity_memory", exc, severity="warning",
                                   action="recall lookup failed")
                return []

        if not seeds:
            return []

        activation = self.spread(list(seeds))
        results: list[dict[str, Any]] = []
        for entity_id, act in sorted(activation.items(), key=lambda kv: kv[1], reverse=True):
            entity = self.get_entity(entity_id)
            if entity is None:
                continue
            if wanted and entity.kind not in wanted:
                continue
            reason = seeds.get(entity_id) or "associated with something named in the cue"
            results.append({
                "entity": entity.to_dict(),
                "activation": round(act, 4),
                "why_surfaced": reason,
                "stance": self.stance(entity).to_dict(),
            })
            if len(results) >= max(1, int(limit)):
                break
        return results

    def dossier(self, entity: Entity | str) -> dict[str, Any] | None:
        """Everything Aura has on one entity: what she knows, and how she feels."""
        entity_obj = entity if isinstance(entity, Entity) else self.get_entity(str(entity))
        if entity_obj is None:
            return None
        width = self._cal.dossier_width
        return {
            "entity": entity_obj.to_dict(),
            "traits": [a.to_dict() for a in
                       self.associations(entity_obj.entity_id,
                                         kind=AssociationKind.TRAIT, limit=width)],
            "facts": [a.to_dict() for a in
                      self.associations(entity_obj.entity_id,
                                        kind=AssociationKind.FACT, limit=width)],
            "events": [a.to_dict() for a in
                       self.associations(entity_obj.entity_id,
                                         kind=AssociationKind.EVENT, limit=width)],
            "relations": [a.to_dict() for a in
                          self.associations(entity_obj.entity_id,
                                            kind=AssociationKind.RELATION, limit=width)],
            "stance": self.stance(entity_obj).to_dict(),
        }

    def status(self) -> dict[str, Any]:
        """Health that can actually report a problem."""
        if not self.available:
            return {"available": False, "reason": "storage unavailable",
                    "entities": 0, "associations": 0}
        with self._lock:
            try:
                entities = self._conn.execute(
                    "SELECT COUNT(*) c FROM entities").fetchone()["c"]
                assocs = self._conn.execute(
                    "SELECT COUNT(*) c FROM associations").fetchone()["c"]
                grounded = self._conn.execute(
                    "SELECT COUNT(*) c FROM associations WHERE provenance LIKE '%evidence_id%'"
                ).fetchone()["c"]
            except sqlite3.Error as exc:
                return {"available": False, "reason": str(exc),
                        "entities": 0, "associations": 0}
        return {
            "available": True,
            "entities": int(entities),
            "associations": int(assocs),
            "grounded_associations": int(grounded),
            "db_path": str(self._db_path),
        }


# ── singleton ───────────────────────────────────────────────────────────────

_INSTANCE: AssociativeEntityMemory | None = None
_INSTANCE_LOCK = checked_lock("associative_entity_memory.instance2")


def reset_associative_entity_memory_for_test() -> None:
    """Close the process-wide entity memory and drop it.

    The instance opens its sqlite connection on construction and keeps it for
    the life of the process, which is correct for the runtime. In a suite it
    means the first test to touch entity memory leaves the handle open, and the
    hermetic guard then fails whichever test is running when it notices —
    reporting associative_entity_memory.sqlite, -wal and -shm against tests that
    never went near a memory.
    """

    global _INSTANCE
    with _INSTANCE_LOCK:
        instance, _INSTANCE = _INSTANCE, None
    if instance is None:
        return
    try:
        instance.close()
    except (sqlite3.Error, OSError, AttributeError) as exc:
        logger.debug("entity memory test reset close failed: %s", exc)


def get_associative_entity_memory() -> AssociativeEntityMemory:
    """Process-wide entity memory, constructed once.

    The attachment system is wired in on construction so people's bonds come
    from the one place that owns them.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            return _INSTANCE
        attachment = None
        try:
            from core.runtime.service_access import optional_service

            engine = optional_service("phenomenal_engine", default=None)
            attachment = getattr(engine, "attachments", None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("attachment system not wired into entity memory: %s", exc)
        _INSTANCE = AssociativeEntityMemory(attachment_system=attachment)
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance(
                "associative_entity_memory", _INSTANCE, required=False
            )
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("entity memory not registered in ServiceContainer: %s", exc)
        return _INSTANCE


__all__ = [
    "Association",
    "AssociationKind",
    "AssociativeEntityMemory",
    "Calibration",
    "Entity",
    "EntityKind",
    "Provenance",
    "Stance",
    "StanceCause",
    "get_associative_entity_memory",
    "reset_associative_entity_memory_for_test",
    "normalize_name",
]
