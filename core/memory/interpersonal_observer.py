"""Noticing things about someone, without inventing them.

`PersonModel` is the store and the reasoning. Nothing wrote to it, which made
it furniture. This is the writer — and it is the dangerous half, because a
component that decides *"that was a rupture"* or *"he prefers X"* is exactly
where a caricature gets manufactured. Everything the store does to prevent
distortion is downstream of what this admits.

So the rules here are strict at the source.

**Nothing is written directly.** Detection produces ``Proposal`` objects;
``admit`` writes them. Noticing and recording are separate acts because they
have different error costs — a wrong notice is a shrug, a wrong record is a
belief about a person that will be read as established fact for months.

**A trait is never proposed from an exchange.** Not by a rule, not by a model.
A disposition inferred from one conversation is precisely the thing the typed
store exists to make unrepresentable, and the easiest place to reintroduce it
is here. Traits may only enter through someone saying one outright ("I'm a
systems thinker"), which arrives as a stated preference or value. What a single
exchange *can* produce is a state, an affect, something he said, a commitment,
a question, or a misunderstanding — things that are true of a moment.

**Only a first-person statement counts as STATED.** Heuristics fire on surface
text and will over-fire. Anything the detectors merely infer is marked
``INFERRED``, which the store renders as "my inference, not something I was
told or saw" — so an over-eager pattern degrades into a visible hedge rather
than a false certainty.

**A language model may propose, never rewrite.** The store's guarantee is that
existing notes are aggregated and never re-worded by a model. Extraction from
raw conversation is a different operation and a legitimate one — something has
to notice. So an optional ``propose`` callable can suggest observations from an
exchange, but its output arrives as INFERRED proposals subject to the same
gate, and it is never handed anything already stored.

The detectors below are deliberately shallow. They catch explicit, high-signal
phrasing and miss everything subtle, which is the right failure direction: a
missed observation costs nothing, and a confident wrong one costs trust.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from core.memory.interpersonal_model import (
    Facet,
    Observation,
    PersonModel,
    Provenance,
    Subject,
    Valence,
)

logger = logging.getLogger("Aura.InterpersonalObserver")

__all__ = ["Exchange", "Proposal", "InterpersonalObserver"]


@dataclass(frozen=True)
class Exchange:
    """One turn of conversation, as the observer sees it."""

    episode_id: str
    user_text: str = ""
    assistant_text: str = ""
    at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Proposal:
    """Something noticed, not yet believed.

    ``rationale`` carries the text that triggered it, so an admitted proposal
    can be traced back to the words that caused it rather than to a detector's
    say-so.
    """

    claim: str
    facet: Facet
    provenance: Provenance
    episode_id: str
    subject: Subject = Subject.THEM
    valence: Valence = Valence.NEUTRAL
    conditions: str = ""
    rationale: str = ""
    at: float = field(default_factory=time.time)
    #: Set when the proposal is a correction of an existing belief rather than
    #: a new one.
    corrects: str | None = None

    def describe(self) -> str:
        return f"[{self.facet}/{self.provenance}] {self.claim} — from {self.rationale!r}"


#: Explicit statements of preference or value, in the person's own words.
#:
#: Ordered most specific first, because the span each one claims is withheld
#: from the rest — see ``_stated``. A general pattern re-reading words a
#: specific one already read produces a second record of a single statement.
_STATED_PATTERNS: tuple[tuple[re.Pattern[str], Facet], ...] = (
    (re.compile(r"\bI (?:prefer|like|love|enjoy|want|need)\b(?:[^.!?\n]{3,80})", re.I), Facet.PREFERENCE),
    (re.compile(r"\bI (?:don't|do not|dont|never) (?:like|want|need|enjoy)\b(?:[^.!?\n]{3,80})", re.I), Facet.PREFERENCE),
    (re.compile(r"\bI (?:hate|dislike|can't stand|cannot stand)\b(?:[^.!?\n]{3,80})", re.I), Facet.PREFERENCE),
    (re.compile(r"\bI (?:care about|value)\b(?:[^.!?\n]{3,80})", re.I), Facet.VALUE),
    (re.compile(r"\bwhat matters to me is\b(?:[^.!?\n]{3,80})", re.I), Facet.VALUE),
    # A standing instruction about how he wants to be dealt with. It needs an
    # emphasis marker — "please", "ever", "never ever" — because a bare
    # negative imperative is as often a figure of speech ("don't worry about
    # it", "never mind") as an instruction, and this module would rather miss
    # an instruction than record a preference he never expressed.
    (
        re.compile(
            r"\b(?:please\s+(?:don'?t|do not|never)|(?:don'?t|do not)\s+ever|never\s+ever)"
            r"\s+(?:[^.!?\n]{3,80})",
            re.I,
        ),
        Facet.PREFERENCE,
    ),
)

#: Signals that she got something wrong about him.
#:
#: Every one of these is an explicit repair frame. A bare "no, ..." is not
#: here on purpose: it opens agreement as readily as correction ("no, it works
#: fine now"), and a false misunderstanding does not just sit there — it feeds
#: the connection dynamic, so enough of them render "we keep missing each
#: other" out of him saying things are fine.
_CORRECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bthat'?s not what I meant\b", re.I),
    re.compile(r"\byou misunderstood\b", re.I),
    re.compile(r"\bthat'?s (?:not right|wrong|incorrect)\b", re.I),
    re.compile(r"\bno,?\s+(?:I meant|I didn'?t|that'?s not|it'?s not)\b", re.I),
)

#: Her own undertakings. Detected on her side of the exchange, not his.
_COMMITMENT_PATTERN = re.compile(r"\bI(?:'ll| will| am going to| shall)\s+(.{4,90})", re.I)

_WARM = re.compile(
    r"\b(?:thank you|thanks|perfect|exactly|love (?:it|this)|great work|nicely done|"
    r"this is good|beautiful)\b", re.I,
)
_DIFFICULT = re.compile(
    r"\b(?:frustrating|annoying|you keep|again\?|this is wrong|not again|"
    r"i already (?:said|told you))\b", re.I,
)

#: Facets a single exchange may never produce. A disposition needs time.
_FORBIDDEN_FROM_EXCHANGE = frozenset({Facet.TRAIT})


def _clean(fragment: str) -> str:
    text = fragment.strip().strip(".,;:!?—-").strip()
    return " ".join(text.split())[:120]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _sentence_around(text: str, span: tuple[int, int]) -> str:
    """The sentence a match came from, as the note on the evidence.

    Wider than the match on purpose. ``audit()`` exists so a human can check
    why she believes something, and a claim shown next to the fragment that
    produced it is checkable only against itself.
    """
    start = max(
        (text.rfind(mark, 0, span[0]) for mark in (".", "!", "?", "\n")), default=-1
    )
    end = min(
        (pos for pos in (text.find(mark, span[1]) for mark in (".", "!", "?", "\n")) if pos != -1),
        default=len(text),
    )
    return _clean(text[start + 1 : end])


class InterpersonalObserver:
    """Turns exchanges into proposals, and admitted proposals into records."""

    def __init__(
        self,
        model: PersonModel,
        *,
        propose: Callable[[Exchange], Sequence[Proposal]] | None = None,
    ) -> None:
        self.model = model
        #: Optional model-backed extractor. Its suggestions are forced to
        #: INFERRED and pass the same gate as everything else.
        self._propose = propose

    # -- noticing ----------------------------------------------------------

    def notice(self, exchange: Exchange) -> list[Proposal]:
        """Everything this exchange suggests. Writes nothing."""
        proposals: list[Proposal] = []
        proposals.extend(self._stated(exchange))
        proposals.extend(self._corrections(exchange))
        proposals.extend(self._affect(exchange))
        proposals.extend(self._commitments(exchange))

        if self._propose is not None:
            try:
                for proposal in self._propose(exchange):
                    # A proposer cannot promote its own confidence. Whatever it
                    # claims to know, it inferred.
                    proposals.append(
                        Proposal(
                            claim=proposal.claim,
                            facet=proposal.facet,
                            provenance=Provenance.INFERRED,
                            episode_id=exchange.episode_id,
                            subject=proposal.subject,
                            valence=proposal.valence,
                            conditions=proposal.conditions,
                            rationale=proposal.rationale or "proposed by model",
                            at=exchange.at,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - a bad proposer must not break a turn
                logger.warning("interpersonal proposer failed: %s", exc)

        return [p for p in proposals if self._admissible(p)]

    def _admissible(self, proposal: Proposal) -> bool:
        if proposal.facet in _FORBIDDEN_FROM_EXCHANGE:
            logger.debug(
                "refusing %s proposal from a single exchange: %s",
                proposal.facet, proposal.claim,
            )
            return False
        if not proposal.claim.strip():
            return False
        # Only the person's own words about themselves are STATED.
        if proposal.provenance is Provenance.STATED and proposal.subject is not Subject.THEM:
            return False
        return True

    # -- detectors ---------------------------------------------------------

    def _stated(self, exchange: Exchange) -> list[Proposal]:
        """What he said about himself, kept as he said it.

        The claim is the matched words verbatim rather than a paraphrase. An
        earlier version rebuilt one — prefixing "does not want" onto whatever
        followed the verb — which turned "I don't like long meetings" into
        "does not want like long meetings" and filed it as something he told
        her. A module whose entire thesis is that rewording person-knowledge
        loses it should not begin by rewording it, and there is no phrasing
        more faithful to what someone meant than what they actually typed.
        """
        out: list[Proposal] = []
        claimed: list[tuple[int, int]] = []
        for pattern, facet in _STATED_PATTERNS:
            for match in pattern.finditer(exchange.user_text):
                span = match.span()
                if any(_overlaps(span, taken) for taken in claimed):
                    # A more specific pattern already read these words. Letting
                    # a general one read them again records one statement twice.
                    continue
                claim = _clean(match.group(0))
                if not claim:
                    continue
                claimed.append(span)
                out.append(
                    Proposal(
                        claim=claim,
                        facet=facet,
                        provenance=Provenance.STATED,
                        episode_id=exchange.episode_id,
                        rationale=_sentence_around(exchange.user_text, span),
                        at=exchange.at,
                    )
                )
        return out

    def _corrections(self, exchange: Exchange) -> list[Proposal]:
        for pattern in _CORRECTION_PATTERNS:
            match = pattern.search(exchange.user_text)
            if match is None:
                continue
            return [
                Proposal(
                    claim="I misread what he meant",
                    facet=Facet.MISUNDERSTANDING,
                    provenance=Provenance.OBSERVED,
                    episode_id=exchange.episode_id,
                    subject=Subject.BETWEEN,
                    valence=Valence.DIFFICULT,
                    rationale=_sentence_around(exchange.user_text, match.span()),
                    at=exchange.at,
                )
            ]
        return []

    def _affect(self, exchange: Exchange) -> list[Proposal]:
        out: list[Proposal] = []
        warm = _WARM.search(exchange.user_text)
        if warm:
            out.append(
                Proposal(
                    claim="that landed well",
                    facet=Facet.AFFECT,
                    provenance=Provenance.OBSERVED,
                    episode_id=exchange.episode_id,
                    subject=Subject.THEM,
                    valence=Valence.WARM,
                    rationale=_sentence_around(exchange.user_text, warm.span()),
                    at=exchange.at,
                )
            )
        difficult = _DIFFICULT.search(exchange.user_text)
        if difficult:
            out.append(
                Proposal(
                    claim="that did not land well",
                    facet=Facet.AFFECT,
                    provenance=Provenance.OBSERVED,
                    episode_id=exchange.episode_id,
                    subject=Subject.THEM,
                    valence=Valence.DIFFICULT,
                    rationale=_sentence_around(exchange.user_text, difficult.span()),
                    at=exchange.at,
                )
            )
        return out

    def _commitments(self, exchange: Exchange) -> list[Proposal]:
        match = _COMMITMENT_PATTERN.search(exchange.assistant_text)
        if match is None:
            return []
        claim = _clean(match.group(1))
        if not claim:
            return []
        return [
            Proposal(
                claim=claim,
                facet=Facet.COMMITMENT,
                provenance=Provenance.OBSERVED,
                episode_id=exchange.episode_id,
                subject=Subject.BETWEEN,
                rationale=_sentence_around(exchange.assistant_text, match.span()),
                at=exchange.at,
            )
        ]

    # -- recording ---------------------------------------------------------

    def admit(self, proposals: Iterable[Proposal]) -> list[Observation]:
        """Write the proposals into the model.

        Separate from ``notice`` so a caller can filter, review, or route
        low-confidence proposals somewhere a human sees them before they
        become things she believes.
        """
        written: list[Observation] = []
        for proposal in proposals:
            if not self._admissible(proposal):
                continue
            observation = self.model.observe(
                proposal.claim,
                episode_id=proposal.episode_id,
                facet=proposal.facet,
                subject=proposal.subject,
                valence=proposal.valence,
                provenance=proposal.provenance,
                conditions=proposal.conditions,
                note=proposal.rationale,
                at=proposal.at,
            )
            written.append(observation)
        return written

    def observe_exchange(self, exchange: Exchange) -> list[Observation]:
        """Notice and admit in one step, for callers that want the default."""
        return self.admit(self.notice(exchange))
