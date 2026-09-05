"""What she knows about a person — typed, sourced, and correctable.

The first version of this stored one kind of thing: a claim with a frequency
count. That was enough to stop prose summarisation flattening a hedge into a
trait, but it had a flaw of its own, and it is the same flaw one level up.

If "frustrated" is noticed across five separate stressful weeks, a
frequency-only model reads five confirmations of a *disposition*. It was five
**states** with five different causes. Counting transient things manufactures a
trait nobody observed — which is the caricature problem again, arriving through
arithmetic instead of adjectives.

And a relationship is not a list of dispositions. It is also what you did
together, what he said outright versus what she guessed, what he corrected her
about, what is unresolved between you, and what she still does not know. Those
are different kinds of knowledge with genuinely different rules, and flattening
them into one shape loses the distinctions that make them useful.

So knowledge here is **typed**:

* ``TRAIT`` accumulates and needs counter-evidence to be honest.
* ``STATE`` expires. It can never be counted toward a trait — that is enforced,
  not conventional, because the arithmetic above is what makes it necessary.
* ``PREFERENCE`` and ``VALUE`` are about what he wants and what is at stake.
* ``EVENT`` happened once, between them. Frequency is meaningless for it and it
  never decays — shared history is not evidence for a proposition.
* ``COMMITMENT`` is open or discharged.
* ``RUPTURE`` is trust damaged, and carries whether it was repaired. A model of
  someone that cannot represent having hurt them is not a model of a
  relationship.
* ``QUESTION`` is what she does not know and wants to. Recording ignorance is
  the part most systems skip, and it is where curiosity comes from.

And **sourced**: ``STATED`` (he told her) outranks ``OBSERVED`` (she saw it)
outranks ``INFERRED`` (she reasoned it). An inference rendered as though it were
a fact is how she ends up confidently wrong about someone, so inferences say so
in the text. He can also overrule anything outright with ``correct()``, and a
correction is permanent — it is the one input that no amount of later observing
can outvote, because being told you are wrong about someone is not evidence to
be weighed against other evidence.

Consolidation remains aggregation, never summarisation. No code path here hands
anything to a language model; the tests assert that structurally.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger("Aura.InterpersonalModel")

__all__ = [
    "Facet",
    "Subject",
    "Provenance",
    "Valence",
    "Occurrence",
    "Observation",
    "Dynamic",
    "PersonModel",
    "DAY_SECONDS",
    "DEFAULT_STATE_TTL",
]

DAY_SECONDS = 86400.0

#: How long a state stands before it stops being current. States are the things
#: that are true *now* — tired, under deadline, unwell — and the failure mode of
#: keeping them forever is treating a bad week as a personality.
DEFAULT_STATE_TTL = 3 * DAY_SECONDS


class Facet(StrEnum):
    TRAIT = "trait"
    STATE = "state"
    PREFERENCE = "preference"
    VALUE = "value"
    EVENT = "event"
    COMMITMENT = "commitment"
    RUPTURE = "rupture"
    QUESTION = "question"
    #: Something lived through together, carrying how it felt rather than only
    #: that it happened. An event is a fact; an experience is a fact you were
    #: both inside of.
    EXPERIENCE = "experience"
    #: How an interaction landed. Neither a trait nor an event — the felt
    #: quality, which is most of what people actually remember.
    AFFECT = "affect"
    #: Something now grasped about each other, and where that grasp failed.
    UNDERSTANDING = "understanding"
    MISUNDERSTANDING = "misunderstanding"
    #: The counterpart to RUPTURE. A record that can only hold the damage is a
    #: ledger of grievances, not a relationship.
    TRUST_BUILT = "trust_built"


class Subject(StrEnum):
    """Who a piece of knowledge is about.

    Everything here used to be implicitly about *him*, which cannot represent
    "we understand each other" or "I make him feel X". A relationship has three
    sides: her view of him, her sense of herself in it, and the thing between
    them that belongs to neither.
    """

    THEM = "them"
    SELF = "self"
    BETWEEN = "between"


class Provenance(StrEnum):
    """Where a piece of knowledge came from. Ordered by authority."""

    INFERRED = "inferred"
    OBSERVED = "observed"
    STATED = "stated"


class Valence(StrEnum):
    """How something felt.

    One field rather than a facet per feeling. Enjoyment, warmth and difficulty
    are not different *kinds of record* needing different storage rules — they
    are the same record with a different quality, and inventing a facet for
    each would multiply the taxonomy without adding a single new rule.
    """

    WARM = "warm"
    NEUTRAL = "neutral"
    DIFFICULT = "difficult"


_AUTHORITY = {Provenance.INFERRED: 0, Provenance.OBSERVED: 1, Provenance.STATED: 2}

#: Facets where seeing it again means "more true". Everywhere else a repeat is
#: either meaningless (an event happened when it happened) or actively
#: misleading (five bad weeks are not one disposition).
_ACCUMULATING = frozenset({Facet.TRAIT, Facet.PREFERENCE, Facet.VALUE})

#: Facets that stop applying on their own.
_EXPIRING = frozenset({Facet.STATE})


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _key(facet: Facet, subject: Subject, claim: str, conditions: str) -> tuple[str, str, str, str]:
    return (str(facet), str(subject), _normalize(claim), _normalize(conditions))


@dataclass(frozen=True)
class Dynamic:
    """A derived reading of the relationship, with the evidence behind it.

    Qualitative on purpose. "trust: 0.72" is a summary of the evidence — the
    same lossy move this module exists to refuse — and a number nobody can
    argue with is a number nobody can correct.
    """

    name: str
    standing: str
    basis: tuple[str, ...] = ()

    def render(self) -> str:
        if not self.basis:
            return f"{self.name}: {self.standing}"
        return f"{self.name}: {self.standing} ({'; '.join(self.basis)})"


@dataclass(frozen=True)
class Occurrence:
    """One time something was noticed, said, or inferred."""

    episode_id: str
    provenance: Provenance = Provenance.OBSERVED
    at: float = field(default_factory=time.time)
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "provenance": str(self.provenance),
            "at": self.at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Occurrence:
        return cls(
            episode_id=str(data["episode_id"]),
            provenance=Provenance(str(data.get("provenance", Provenance.OBSERVED))),
            at=float(data.get("at", 0.0)),
            note=str(data.get("note", "")),
        )


@dataclass
class Observation:
    """One piece of knowledge about a person, with its kind and evidence."""

    claim: str
    facet: Facet = Facet.TRAIT
    subject: Subject = Subject.THEM
    valence: Valence = Valence.NEUTRAL
    conditions: str = ""
    occurrences: list[Occurrence] = field(default_factory=list)
    counter_examples: list[Occurrence] = field(default_factory=list)
    ttl: float | None = None
    #: Set when the person overruled this. Permanent, and unrenderable.
    corrected_by: Occurrence | None = None
    #: RUPTURE: how it was repaired. QUESTION/COMMITMENT: how it was closed.
    resolved_by: Occurrence | None = None

    @property
    def key(self) -> tuple[str, str, str, str]:
        # Facet is part of the identity: "tired" the state and "tired" the
        # trait are different claims, and merging them is the arithmetic that
        # turns a bad week into a personality. Subject too — "I feel understood"
        # and "he feels understood" are not the same fact.
        return (
            str(self.facet),
            str(self.subject),
            _normalize(self.claim),
            _normalize(self.conditions),
        )

    @property
    def support(self) -> int:
        return len(self.occurrences)

    @property
    def contradictions(self) -> int:
        return len(self.counter_examples)

    @property
    def corrected(self) -> bool:
        return self.corrected_by is not None

    @property
    def resolved(self) -> bool:
        return self.resolved_by is not None

    @property
    def authority(self) -> Provenance:
        """The strongest source behind this. Stated beats observed beats guessed."""
        if not self.occurrences:
            return Provenance.INFERRED
        return max((o.provenance for o in self.occurrences), key=lambda p: _AUTHORITY[p])

    @property
    def accumulates(self) -> bool:
        return self.facet in _ACCUMULATING

    def last_seen(self) -> float | None:
        return max((o.at for o in self.occurrences), default=None)

    def first_seen(self) -> float | None:
        return min((o.at for o in self.occurrences), default=None)

    def expires_at(self) -> float | None:
        if self.facet not in _EXPIRING:
            return None
        last = self.last_seen()
        if last is None:
            return None
        return last + (self.ttl if self.ttl is not None else DEFAULT_STATE_TTL)

    def is_current(self, now: float | None = None) -> bool:
        """Whether this still applies.

        Corrections are permanent. A resolved rupture or answered question is
        history rather than current knowledge. A lapsed state is simply no
        longer true.
        """
        if self.corrected:
            return False
        if self.facet in {Facet.RUPTURE, Facet.QUESTION, Facet.COMMITMENT} and self.resolved:
            return False
        expiry = self.expires_at()
        if expiry is None:
            return True
        return (time.time() if now is None else now) < expiry

    def episodes(self) -> list[str]:
        return [o.episode_id for o in self.occurrences]

    def to_dict(self) -> dict[str, object]:
        """Every field, because a serialiser that drops one is this module's
        own failure arriving through a different door.

        Losing ``counter_examples`` across a restart would leave a record that
        can only accumulate confirmations, which is the grudge this store was
        built to refuse. ``tests/test_interpersonal_persistence.py`` derives the
        expected keys from the dataclass, so a new field that is not written
        here fails rather than being silently forgotten at the next restart.
        """
        return {
            "claim": self.claim,
            "facet": str(self.facet),
            "subject": str(self.subject),
            "valence": str(self.valence),
            "conditions": self.conditions,
            "occurrences": [o.to_dict() for o in self.occurrences],
            "counter_examples": [o.to_dict() for o in self.counter_examples],
            "ttl": self.ttl,
            "corrected_by": self.corrected_by.to_dict() if self.corrected_by else None,
            "resolved_by": self.resolved_by.to_dict() if self.resolved_by else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Observation:
        ttl = data.get("ttl")
        corrected = data.get("corrected_by")
        resolved = data.get("resolved_by")
        return cls(
            claim=str(data["claim"]),
            facet=Facet(str(data.get("facet", Facet.TRAIT))),
            subject=Subject(str(data.get("subject", Subject.THEM))),
            valence=Valence(str(data.get("valence", Valence.NEUTRAL))),
            conditions=str(data.get("conditions", "")),
            occurrences=[
                Occurrence.from_dict(o) for o in data.get("occurrences", [])  # type: ignore[union-attr]
            ],
            counter_examples=[
                Occurrence.from_dict(o) for o in data.get("counter_examples", [])  # type: ignore[union-attr]
            ],
            ttl=None if ttl is None else float(ttl),  # type: ignore[arg-type]
            corrected_by=Occurrence.from_dict(corrected) if corrected else None,  # type: ignore[arg-type]
            resolved_by=Occurrence.from_dict(resolved) if resolved else None,  # type: ignore[arg-type]
        )

    def render(self, *, now: float | None = None) -> str:
        """As context text, stating kind, evidence and how she knows it."""
        now = time.time() if now is None else now
        parts = [self.claim.strip()]
        if self.conditions:
            parts.append(f"({self.conditions.strip()})")

        evidence: list[str] = []
        if self.authority is Provenance.STATED:
            evidence.append("he told me this")
        elif self.authority is Provenance.INFERRED:
            # Said out loud, because an inference rendered as fact is how she
            # ends up confidently wrong about someone.
            evidence.append("my inference, not something I was told or saw")

        if self.accumulates:
            evidence.append(f"noticed {_times(self.support)}")
        elif self.facet is Facet.EVENT:
            first = self.first_seen()
            if first is not None:
                evidence.append(_ago(now - first))

        last = self.last_seen()
        if last is not None and self.accumulates:
            evidence.append(f"most recently {_ago(now - last)}")
        if self.contradictions:
            evidence.append(f"{_times(self.contradictions)} it did not hold")
        if self.facet is Facet.STATE:
            evidence.append("a current state, not a trait")
        if self.facet is Facet.RUPTURE and not self.resolved:
            evidence.append("not yet repaired")

        if evidence:
            parts.append("— " + ", ".join(evidence))
        return " ".join(parts)


def _times(count: int) -> str:
    if count == 1:
        return "once"
    if count == 2:
        return "twice"
    return f"{count} times"


def _ago(seconds: float) -> str:
    if seconds < DAY_SECONDS:
        return "today"
    days = int(seconds // DAY_SECONDS)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    return "a month ago" if months == 1 else f"{months} months ago"


#: Headings, in the order a person is worth describing: what they want and care
#: about before what they are like, because the former explains the latter.
_FACET_HEADINGS: tuple[tuple[Facet, str], ...] = (
    (Facet.VALUE, "What matters to them"),
    (Facet.PREFERENCE, "What they prefer"),
    (Facet.UNDERSTANDING, "What we understand about each other"),
    (Facet.MISUNDERSTANDING, "Where we have missed each other"),
    (Facet.TRAIT, "Patterns I have noticed"),
    (Facet.STATE, "True right now"),
    (Facet.AFFECT, "How this has felt"),
    (Facet.EXPERIENCE, "What we have been through together"),
    (Facet.EVENT, "Things that happened"),
    (Facet.COMMITMENT, "Open between us"),
    (Facet.TRUST_BUILT, "Where trust was built"),
    (Facet.RUPTURE, "Where I got it wrong with them"),
    (Facet.QUESTION, "What I still do not know"),
)


class PersonModel:
    """Her structured, typed view of one person."""

    def __init__(self, person: str, *, max_observations: int = 256) -> None:
        if not person or not person.strip():
            raise ValueError("a person model needs someone to be about")
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        self.person = person.strip()
        self.max_observations = max_observations
        self._observations: dict[tuple[str, str, str, str], Observation] = {}

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self):
        return iter(self._observations.values())

    # -- recording ---------------------------------------------------------

    def observe(
        self,
        claim: str,
        *,
        episode_id: str,
        facet: Facet = Facet.TRAIT,
        subject: Subject = Subject.THEM,
        valence: Valence = Valence.NEUTRAL,
        provenance: Provenance = Provenance.OBSERVED,
        conditions: str = "",
        note: str = "",
        ttl: float | None = None,
        at: float | None = None,
    ) -> Observation:
        """Record something. Repeats increment; they never rewrite.

        A repeat of a non-accumulating facet is still recorded — the evidence
        is real — but ``render`` will not describe it as a frequency, because
        five bad weeks are five states rather than one disposition.
        """
        if not claim or not claim.strip():
            raise ValueError("an observation needs a claim")
        if not episode_id:
            raise ValueError(
                "an observation needs the episode that justifies it; a claim "
                "with no evidence is exactly what this module exists to prevent"
            )
        candidate = Observation(
            claim=claim.strip(),
            facet=facet,
            subject=subject,
            valence=valence,
            conditions=conditions.strip(),
            ttl=ttl,
        )
        existing = self._observations.get(candidate.key)
        if existing is None:
            self._observations[candidate.key] = candidate
            existing = candidate
        if any(item.episode_id == episode_id for item in existing.occurrences):
            return existing
        if existing.corrected:
            # He already said this was wrong. Observing it again does not
            # reopen the question; that is what "correction is permanent" means.
            logger.debug("ignoring re-observation of corrected claim: %s", claim)
            return existing
        existing.occurrences.append(
            Occurrence(
                episode_id=episode_id,
                provenance=provenance,
                at=time.time() if at is None else at,
                note=note,
            )
        )
        self._evict_if_needed()
        return existing

    def remove_episode(self, episode_id: str) -> int:
        """Remove every evidence edge owned by one superseded episode."""

        target = str(episode_id or "").strip()
        if not target:
            return 0
        removed = 0
        for key, observation in list(self._observations.items()):
            occurrences = [
                item for item in observation.occurrences if item.episode_id != target
            ]
            counter_examples = [
                item
                for item in observation.counter_examples
                if item.episode_id != target
            ]
            removed += len(observation.occurrences) - len(occurrences)
            removed += len(observation.counter_examples) - len(counter_examples)
            observation.occurrences = occurrences
            observation.counter_examples = counter_examples
            if (
                observation.corrected_by is not None
                and observation.corrected_by.episode_id == target
            ):
                observation.corrected_by = None
                removed += 1
            if (
                observation.resolved_by is not None
                and observation.resolved_by.episode_id == target
            ):
                observation.resolved_by = None
                removed += 1
            if (
                not observation.occurrences
                and not observation.counter_examples
                and observation.corrected_by is None
                and observation.resolved_by is None
            ):
                del self._observations[key]
        return removed

    def contradict(
        self,
        claim: str,
        *,
        episode_id: str,
        facet: Facet = Facet.TRAIT,
        subject: Subject = Subject.THEM,
        conditions: str = "",
        note: str = "",
        at: float | None = None,
    ) -> Observation | None:
        """Record an occasion the claim did not hold."""
        observation = self._observations.get(
            _key(facet, subject, claim, conditions)
        )
        if observation is None:
            return None
        observation.counter_examples.append(
            Occurrence(
                episode_id=episode_id,
                provenance=Provenance.OBSERVED,
                at=time.time() if at is None else at,
                note=note,
            )
        )
        return observation

    def correct(
        self,
        claim: str,
        *,
        episode_id: str,
        facet: Facet = Facet.TRAIT,
        subject: Subject = Subject.THEM,
        conditions: str = "",
        note: str = "",
    ) -> Observation | None:
        """The person says this is wrong. Permanent, and outranks everything.

        Not a counter-example — a counter-example is evidence to be weighed.
        Being told you are wrong about someone is not evidence, it is the
        answer, and no amount of subsequent observing may outvote it.
        """
        observation = self._observations.get(
            _key(facet, subject, claim, conditions)
        )
        if observation is None:
            return None
        observation.corrected_by = Occurrence(
            episode_id=episode_id, provenance=Provenance.STATED, note=note
        )
        return observation

    def resolve(
        self,
        claim: str,
        *,
        episode_id: str,
        facet: Facet,
        subject: Subject = Subject.THEM,
        conditions: str = "",
        note: str = "",
    ) -> Observation | None:
        """Close a rupture (repaired), a question (answered), or a commitment."""
        observation = self._observations.get(
            _key(facet, subject, claim, conditions)
        )
        if observation is None:
            return None
        observation.resolved_by = Occurrence(
            episode_id=episode_id, provenance=Provenance.OBSERVED, note=note
        )
        return observation

    def forget(
        self,
        claim: str,
        *,
        facet: Facet = Facet.TRAIT,
        subject: Subject = Subject.THEM,
        conditions: str = "",
    ) -> bool:
        return (
            self._observations.pop(_key(facet, subject, claim, conditions), None)
            is not None
        )

    # -- consolidation is aggregation --------------------------------------

    def merge(self, other: PersonModel) -> None:
        """Fold another model in by aggregating evidence. Never by rewording."""
        if _normalize(other.person) != _normalize(self.person):
            raise ValueError(
                f"refusing to merge notes about {other.person!r} into "
                f"{self.person!r} — conflating two people is not a compression"
            )
        for observation in other:
            mine = self._observations.get(observation.key)
            if mine is None:
                self._observations[observation.key] = observation
                continue
            mine.occurrences.extend(
                _new_only(observation.occurrences, mine.occurrences)
            )
            mine.counter_examples.extend(
                _new_only(observation.counter_examples, mine.counter_examples)
            )
            # A correction anywhere is a correction everywhere.
            mine.corrected_by = mine.corrected_by or observation.corrected_by
            mine.resolved_by = mine.resolved_by or observation.resolved_by
        self._evict_if_needed()

    def _rank(self, observation: Observation) -> tuple:
        """Ordering within a facet. Authority first, then net evidence.

        Never across facets: an event and a trait are not comparable by count,
        and ranking them together would let shared history be crowded out by a
        much-repeated triviality.
        """
        return (
            _AUTHORITY[observation.authority],
            observation.support - observation.contradictions,
            observation.last_seen() or 0.0,
        )

    def _evict_if_needed(self) -> None:
        if len(self._observations) <= self.max_observations:
            return
        # Never evict what he told her, what is unrepaired, or shared history.
        protected = {Facet.EVENT, Facet.RUPTURE, Facet.COMMITMENT}
        candidates = [
            o for o in self._observations.values()
            if o.facet not in protected and o.authority is not Provenance.STATED
        ]
        ranked = sorted(candidates, key=self._rank)
        for observation in ranked[: len(self._observations) - self.max_observations]:
            logger.debug("evicting weakly-evidenced %s: %s", observation.facet, observation.claim)
            del self._observations[observation.key]

    # -- reading -----------------------------------------------------------

    def current(self, *, facet: Facet | None = None, now: float | None = None) -> list[Observation]:
        """Everything that still applies, best-sourced first."""
        found = [o for o in self._observations.values() if o.is_current(now)]
        if facet is not None:
            found = [o for o in found if o.facet is facet]
        return sorted(found, key=self._rank, reverse=True)

    def open_questions(self, *, now: float | None = None) -> list[Observation]:
        return self.current(facet=Facet.QUESTION, now=now)

    def unrepaired(self, *, now: float | None = None) -> list[Observation]:
        return self.current(facet=Facet.RUPTURE, now=now)

    def render(
        self,
        *,
        per_facet: int = 6,
        now: float | None = None,
        include_dynamics: bool = True,
    ) -> str:
        """The block text, grouped by kind of knowledge.

        ``include_dynamics`` exists for callers working to a character budget.
        The readings are *derived* from the same records printed above them, so
        when something has to go they go first: they can be recomputed from
        what remains, and the evidence cannot be recomputed from them.
        """
        sections: list[str] = []
        for facet, heading in _FACET_HEADINGS:
            observations = self.current(facet=facet, now=now)[:per_facet]
            if not observations:
                continue
            sections.append(heading + ":")
            sections.extend(f"- {o.render(now=now)}" for o in observations)
            sections.append("")
        if not sections:
            return f"I do not know anything about {self.person} yet."

        readings = [d for d in self.dynamics(now=now) if d.basis] if include_dynamics else []
        if readings:
            sections.append("Where this stands:")
            sections.extend(f"- {d.render()}" for d in readings)

        return f"What I know about {self.person}:\n\n" + "\n".join(sections).strip()

    # -- dynamics: read off the record, never asserted ----------------------

    def dynamics(self, *, now: float | None = None) -> list[Dynamic]:
        """Trust, safety, comfort, novelty, connection, enjoyment, investment.

        These are *derived*, and that is the whole point. Storing "trust: 0.7"
        would be a claim nobody could check and everybody would round up. Here
        each reading is a function over the same evidence everything else uses,
        and it carries the basis that produced it — so a low reading can be
        argued with, and a high one can be audited.

        There are more of these than there are facets, deliberately. Facets need
        distinct *storage rules*; dynamics need only a function. That asymmetry
        is what lets the felt side of a relationship be rich without the
        taxonomy becoming unusable.
        """
        now = time.time() if now is None else now
        readings = [
            self._trust(now),
            self._what_it_has_been_through(now),
            self._safety(now),
            self._comfort(now),
            self._connection(now),
            self._enjoyment(now),
            self._novelty(now),
            self._investment(now),
        ]
        # A reading with nothing to say is left out rather than saying so.
        # Every dynamic here carries the evidence that produced it, and one
        # that reports the absence of evidence breaks that and fills the
        # block with the absence of things.
        return [one for one in readings if one.standing]

    def _what_it_has_been_through(self, now: float) -> Dynamic:
        """Trust read in the order it happened, which counts cannot do.

        `_trust` above counts: so many kept, so many repaired, so many not. A
        count has no order, and the order is where the interesting fact
        lives. A bond that broke, was repaired, and then held through four
        more commitments is not the same as one repaired and never tested
        again, and the two produce identical counts.

        The long horizon computes this and had nothing to compute it from —
        it wanted a history and the store held a set. This is the bridge, and
        it reads what living already wrote; it never writes anything to make
        the reading nicer.
        """

        try:
            from core.social.what_the_years_add_up_to import how_they_stand

            where = how_they_stand(self.person, model=self, now=now)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return Dynamic(name="what it has been through", standing="", basis=())
        if not where.tested:
            # Nothing has strained it, and `trust` above already says
            # untested. A second line saying the same absence is noise.
            return Dynamic(name="what it has been through", standing="", basis=())
        if where.unrepaired:
            standing = f"{where.unrepaired} thing(s) still open between us"
        elif where.stronger_for_it:
            standing = "strained, repaired, and held since — stronger for it"
        else:
            standing = "repaired, and not tested since"
        return Dynamic(
            name="what it has been through",
            standing=standing,
            basis=(
                f"{where.breaks} rupture(s), {where.repairs} repaired",
                f"proved {where.proved:.0%} of the way by what followed",
                f"trust from the history: {where.trust:.2f}",
            ),
        )

    def _count(self, facet: Facet, *, resolved: bool | None = None) -> int:
        return sum(
            1 for o in self._observations.values()
            if o.facet is facet
            and not o.corrected
            and (resolved is None or o.resolved is resolved)
        )

    def _trust(self, now: float) -> Dynamic:
        kept = self._count(Facet.COMMITMENT, resolved=True)
        open_ = self._count(Facet.COMMITMENT, resolved=False)
        repaired = self._count(Facet.RUPTURE, resolved=True)
        unrepaired = self._count(Facet.RUPTURE, resolved=False)
        built = self._count(Facet.TRUST_BUILT)

        credit = kept + repaired + built
        if unrepaired:
            standing = "damaged" if unrepaired > credit else "strained but holding"
        elif credit == 0:
            standing = "untested"
        else:
            standing = "well established" if credit >= 3 else "building"
        return Dynamic(
            name="trust",
            standing=standing,
            basis=(
                f"{kept} commitment(s) kept, {open_} open",
                f"{repaired} rupture(s) repaired, {unrepaired} not",
                f"{built} moment(s) that built it",
            ),
        )

    def _safety(self, now: float) -> Dynamic:
        unrepaired = self._count(Facet.RUPTURE, resolved=False)
        corrections = sum(1 for o in self._observations.values() if o.corrected)
        disclosed = sum(
            1 for o in self._observations.values()
            if o.authority is Provenance.STATED and not o.corrected
        )
        if unrepaired > 1:
            standing = "guarded"
        elif disclosed >= 3 or corrections >= 2:
            standing = "safe enough to be corrected and told things"
        elif disclosed:
            standing = "opening"
        else:
            standing = "unknown"
        return Dynamic(
            name="safety",
            standing=standing,
            basis=(
                f"{disclosed} thing(s) he chose to tell me",
                f"{corrections} time(s) he corrected me and it stuck",
                f"{unrepaired} unrepaired rupture(s)",
            ),
        )

    def _comfort(self, now: float) -> Dynamic:
        stamps = [
            o.first_seen() for o in self._observations.values() if o.first_seen()
        ]
        if not stamps:
            return Dynamic("comfort", "no shared time yet", ())
        span_days = int((now - min(stamps)) // DAY_SECONDS)
        recent = sum(1 for s in stamps if now - s < 7 * DAY_SECONDS)
        if span_days >= 30 and recent:
            standing = "familiar"
        elif span_days >= 7:
            standing = "settling"
        else:
            standing = "new"
        return Dynamic(
            name="comfort",
            standing=standing,
            basis=(
                f"known for {span_days} day(s)",
                f"{recent} thing(s) noticed in the last week",
            ),
        )

    def _connection(self, now: float) -> Dynamic:
        mutual = self._count(Facet.UNDERSTANDING)
        missed = self._count(Facet.MISUNDERSTANDING)
        shared = self._count(Facet.EXPERIENCE) + self._count(Facet.EVENT)
        between = sum(
            1 for o in self._observations.values() if o.subject is Subject.BETWEEN
        )
        if mutual == 0 and shared == 0:
            standing = "not yet formed"
        elif missed > mutual:
            standing = "we keep missing each other"
        elif mutual >= 3 or shared >= 3:
            standing = "we understand each other"
        else:
            standing = "forming"
        return Dynamic(
            name="connection",
            standing=standing,
            basis=(
                f"{mutual} thing(s) we understand about each other, {missed} we did not",
                f"{shared} shared experience(s)",
                f"{between} thing(s) that belong to neither of us alone",
            ),
        )

    def _enjoyment(self, now: float) -> Dynamic:
        warm = sum(
            1 for o in self._observations.values()
            if o.valence is Valence.WARM and not o.corrected
        )
        difficult = sum(
            1 for o in self._observations.values()
            if o.valence is Valence.DIFFICULT and not o.corrected
        )
        if warm == 0 and difficult == 0:
            standing = "unmeasured"
        elif difficult > warm:
            standing = "more difficult than easy lately"
        elif warm >= 3:
            standing = "we enjoy this"
        else:
            standing = "some warmth"
        return Dynamic(
            name="enjoyment",
            standing=standing,
            basis=(f"{warm} warm, {difficult} difficult",),
        )

    def _novelty(self, now: float) -> Dynamic:
        """New ground versus familiar ground, recently.

        Measured as first sightings in the last fortnight against repeats in
        the same window: discovering things about someone is what novelty *is*
        here, and it is directly readable off the record rather than guessed at.
        """
        window = 14 * DAY_SECONDS
        firsts = 0
        repeats = 0
        for observation in self._observations.values():
            for index, occurrence in enumerate(sorted(observation.occurrences, key=lambda o: o.at)):
                if now - occurrence.at > window:
                    continue
                if index == 0:
                    firsts += 1
                else:
                    repeats += 1
        if firsts == 0 and repeats == 0:
            standing = "quiet"
        elif firsts > repeats:
            standing = "still discovering each other"
        elif firsts:
            standing = "mostly familiar ground, some new"
        else:
            standing = "well-worn ground"
        return Dynamic(
            name="novelty",
            standing=standing,
            basis=(f"{firsts} new thing(s) and {repeats} familiar in the last fortnight",),
        )

    def _investment(self, now: float) -> Dynamic:
        commitments = self._count(Facet.COMMITMENT)
        questions = self._count(Facet.QUESTION, resolved=False)
        total = sum(o.support for o in self._observations.values())
        if total == 0:
            standing = "none yet"
        elif commitments or questions:
            standing = "actively invested"
        else:
            standing = "present but not committed"
        return Dynamic(
            name="investment",
            standing=standing,
            basis=(
                f"{commitments} commitment(s) between us",
                f"{questions} thing(s) I still want to know",
                f"{total} recorded moment(s)",
            ),
        )

    # -- durability --------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """The whole model, field for field.

        Restoring goes through ``from_dict`` rather than replaying ``observe``:
        replaying would re-stamp every occurrence with the time of the restart,
        turning a year of evidence into a burst of sightings on boot day, and
        would silently drop anything he had corrected — ``observe`` refuses a
        corrected claim, which is right at the door and wrong on reload.
        """
        return {
            "person": self.person,
            "max_observations": self.max_observations,
            "observations": [o.to_dict() for o in self._observations.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PersonModel:
        model = cls(
            str(data["person"]),
            max_observations=int(data.get("max_observations", 256)),  # type: ignore[arg-type]
        )
        for entry in data.get("observations", []):  # type: ignore[union-attr]
            observation = Observation.from_dict(entry)
            model._observations[observation.key] = observation
        return model

    def audit(self) -> list[dict[str, object]]:
        """Every claim with its kind, source and evidence, for a human to check."""
        return [
            {
                "claim": o.claim,
                "facet": str(o.facet),
                "conditions": o.conditions,
                "authority": str(o.authority),
                "support": o.support,
                "contradictions": o.contradictions,
                "corrected": o.corrected,
                "resolved": o.resolved,
                "current": o.is_current(),
                "episodes": o.episodes(),
            }
            for o in sorted(self._observations.values(), key=self._rank, reverse=True)
        ]


def _new_only(incoming: list[Occurrence], existing: list[Occurrence]) -> list[Occurrence]:
    seen = {(o.episode_id, o.at) for o in existing}
    return [o for o in incoming if (o.episode_id, o.at) not in seen]
