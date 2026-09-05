"""Turn something Aura read into something Aura understood.

LIVE DEFECT, 2026-08-03. Aura browsed /r/philosophy, and what landed in memory
was the page, verbatim, with the navigation chrome still attached:

    Reddit read r/philosophy: Western philosophy has been at war with The
    Odyssey for 2,800 years -- and keeps losing. ... : r/philosophy Skip to
    main content ... Go to Reddit Answers ... | logged | stored_via_manager

``action="logged"``, ``outcome="stored_via_manager"``. The event that she read
something was recorded. What she made of it was not recorded, because nothing
ever asked. So a minute later the reading could not tell her anything: not
whether the claim was new, not whether it agreed with what she already
believed, not whether the source was worth believing at all — and the reply
she gave about it had drifted to an unrelated subject entirely.

Reading is not an event to log. It is an encounter with a claim, and the
things worth keeping are:

* **the claim** — what the source actually asserts, in her words rather than
  its markup;
* **her stance** — does this affirm, contradict, extend, or merely repeat
  something she already holds? Affirmation is not nothing: a belief that has
  survived contact with an independent source is stronger than one that has
  not;
* **the source's quality** — a forum post, an unsourced assertion, a hostile
  headline and a peer-reviewed result are not equal evidence, and being able
  to say "this is a bad argument" IS a thing learned;
* **what it touches** — the beliefs and topics it connects to, so it can be
  found again by something other than a keyword.

This module produces that record. It is deliberately honest about its own
limits: every field it cannot establish is left empty rather than guessed, and
``stance`` is ``unassessed`` until something actually assesses it.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field, fields
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("Aura.Knowledge.SourceComprehension")

SCHEMA = "aura.knowledge.source_comprehension.v1"

#: How much of a source's text is worth keeping as evidence for a claim.
_MAX_CLAIM_CHARS = 400
_MAX_EVIDENCE_CHARS = 700

#: Site furniture that is not content. A record that keeps these has kept the
#: page rather than what the page said.
_CHROME_PATTERNS = (
    # Bounded on purpose. An unbounded ".*" here consumed the whole page from
    # the first navigation phrase onward, so stripping the chrome stripped the
    # article with it.
    re.compile(r"(?i)\bskip to main content\b"),
    re.compile(r"(?i)\bgo to reddit answers\b"),
    re.compile(r"(?i)\b(?:log ?in|sign ?up|subscribe|accept all cookies)\b"),
    re.compile(r"(?i)\bopen menu\b|\bclose menu\b|\bexpand user menu\b"),
    re.compile(r"(?i)^\s*(?:home|popular|all|topics|resources)\s*$", re.MULTILINE),
    re.compile(r"(?i)\br/\w+\s*:\s*r/\w+"),
)

#: What kind of source this is. Ordered: the first match wins, so the more
#: specific hosts come before the generic shapes.
_SOURCE_KINDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "peer_reviewed",
        ("arxiv.org", "doi.org", "pubmed", "nature.com", "science.org", "acm.org", "ieee.org"),
        "A paper. Method and data are inspectable, which is the strongest "
        "thing a source can offer — it is not the same as being right.",
    ),
    (
        "reference",
        ("wikipedia.org", "britannica.com", "stanford.edu/entries", "plato.stanford.edu"),
        "A reference work. Good for orientation and for names and dates; its "
        "claims are summaries of other people's work, not evidence itself.",
    ),
    (
        "documentation",
        ("docs.", "readthedocs", "developer.mozilla.org", "github.com"),
        "Documentation or source. Authoritative about what a thing DOES, "
        "silent about whether it is a good idea.",
    ),
    (
        "forum",
        ("reddit.com", "news.ycombinator.com", "stackexchange", "stackoverflow", "quora"),
        "A forum post. This is somebody's opinion with a vote count attached. "
        "Votes measure agreement, not correctness, and the comments are often "
        "worth more than the post.",
    ),
    (
        "news",
        ("nytimes", "bbc.", "guardian", "reuters", "apnews", "wsj.", "cnn."),
        "A news article. Reliable about what happened, weaker about why, and "
        "the headline is usually written by someone other than the reporter.",
    ),
    (
        "social",
        ("twitter.com", "x.com", "facebook.com", "instagram.com", "tiktok.com"),
        "A social post. Optimised for reach, not for being checkable.",
    ),
)

#: Sentence shapes that carry a claim worth keeping, versus ones that are the
#: page talking about itself.
_CLAIM_HINT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|does|do|can|cannot|will|should|must|"
    r"causes?|means?|shows?|proves?|argues?|claims?|suggests?|finds?)\b",
    re.IGNORECASE,
)

#: Rhetoric that should lower confidence in an argument regardless of topic.
_WEAK_ARGUMENT_MARKERS: tuple[tuple[str, str], ...] = (
    ("everyone knows", "appeals to consensus instead of evidence"),
    ("obviously", "asserts where it should argue"),
    ("it is well known", "appeals to consensus instead of evidence"),
    ("proves that", "claims proof, which almost nothing outside mathematics earns"),
    ("always", "makes a universal claim that one counterexample would defeat"),
    ("never", "makes a universal claim that one counterexample would defeat"),
    ("literally", "uses an intensifier where a measurement belongs"),
    ("destroys", "frames a disagreement as a contest rather than an argument"),
    ("debunked", "frames a disagreement as a contest rather than an argument"),
    ("keeps losing", "frames a disagreement as a contest rather than an argument"),
)


@dataclass
class SourceComprehension:
    """What Aura made of something she read."""

    schema: str = SCHEMA
    url: str = ""
    title: str = ""
    source_kind: str = "unknown"
    source_caveat: str = ""
    claim: str = ""
    evidence_excerpt: str = ""
    #: affirms | contradicts | extends | repeats | unassessed
    stance: str = "unassessed"
    stance_basis: str = ""
    argument_weaknesses: list[str] = field(default_factory=list)
    related_beliefs: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    content_sha256: str = ""

    @property
    def understood(self) -> bool:
        """Whether anything was actually extracted. Empty is not understanding.

        A claim has to contain a word. Emptiness was the only thing ruled out,
        so a page whose whole extractable content was "12" produced a record
        claiming — with a stance and a stored narrative — that the source
        claims 12. A later turn then says that out loud. Requiring a letter is
        not a length threshold; it is the difference between a statement and a
        fragment that survived extraction.
        """
        return bool(self.claim) and any(character.isalpha() for character in self.claim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "url": self.url,
            "title": self.title,
            "source_kind": self.source_kind,
            "source_caveat": self.source_caveat,
            "claim": self.claim,
            "evidence_excerpt": self.evidence_excerpt,
            "stance": self.stance,
            "stance_basis": self.stance_basis,
            "argument_weaknesses": list(self.argument_weaknesses),
            "related_beliefs": list(self.related_beliefs),
            "topics": list(self.topics),
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "SourceComprehension":
        """Rebuild from :meth:`to_dict`, ignoring anything foreign.

        The read paths pass comprehension around as a plain dict because that
        is what memory stores. Reconstructing here means a caller does not
        have to re-read the page to act on what it already understood.
        """
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def narrative(self) -> str:
        """What she would say she took from it, in one short paragraph."""
        if not self.understood:
            return "I opened it but couldn't get a claim out of it worth keeping."
        parts = [f"Claim: {self.claim}"]
        if self.source_caveat:
            parts.append(f"Source: {self.source_caveat}")
        if self.stance != "unassessed" and self.stance_basis:
            parts.append(f"Against what I hold: {self.stance_basis}")
        if self.argument_weaknesses:
            parts.append("Weak points: " + "; ".join(self.argument_weaknesses[:3]))
        return " ".join(parts)


def strip_site_chrome(text: str) -> str:
    """Remove navigation furniture so a claim is not stored with the menu."""
    cleaned = str(text or "")
    for pattern in _CHROME_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def classify_source(url: str) -> tuple[str, str]:
    """The kind of source, and what that kind is and is not good for."""
    text = str(url or "").strip().lower()
    if not text:
        return "unknown", ""
    try:
        host = (urlparse(text).hostname or text).lower()
    except (TypeError, ValueError):
        host = text
    haystack = f"{host}{urlparse(text).path if '://' in text else ''}"
    for kind, markers, caveat in _SOURCE_KINDS:
        if any(marker in haystack for marker in markers):
            return kind, caveat
    return "web_page", (
        "An ordinary web page. Nothing about it establishes who wrote it or "
        "whether anyone checked it."
    )


def extract_claim(text: str, *, title: str = "") -> tuple[str, str]:
    """The sentence a source is actually asserting, and the text around it.

    Prefers a titled claim, because a headline is usually the thesis; falls
    back to the first sentence that asserts something. Returns empty strings
    when nothing does — a page that makes no claim taught nothing, and saying
    so is better than storing its markup.
    """
    body = strip_site_chrome(text)
    heading = strip_site_chrome(title)
    if heading and _CLAIM_HINT_RE.search(heading):
        claim = heading[:_MAX_CLAIM_CHARS]
        return claim, body[:_MAX_EVIDENCE_CHARS]
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        candidate = sentence.strip()
        if len(candidate) < 25 or len(candidate) > _MAX_CLAIM_CHARS:
            continue
        if _CLAIM_HINT_RE.search(candidate):
            return candidate, body[:_MAX_EVIDENCE_CHARS]
    if heading:
        return heading[:_MAX_CLAIM_CHARS], body[:_MAX_EVIDENCE_CHARS]
    return "", ""


def argument_weaknesses(text: str) -> list[str]:
    """Rhetoric that should lower confidence, whatever the subject is.

    Being able to say "this is a bad argument" is itself something learned, so
    it is recorded rather than silently discounting the source.
    """
    lowered = str(text or "").lower()
    found: list[str] = []
    for marker, why in _WEAK_ARGUMENT_MARKERS:
        if marker in lowered and why not in found:
            found.append(why)
    return found


def assess_stance(
    claim: str,
    *,
    known_beliefs: list[str] | None = None,
) -> tuple[str, str, list[str]]:
    """Where this claim sits relative to what Aura already holds.

    Returns (stance, basis, related beliefs). ``unassessed`` when there is
    nothing to compare against — an unexamined claim is not a confirmed one,
    and pretending otherwise is how a source gets believed for no reason.
    """
    beliefs = [str(b or "").strip() for b in (known_beliefs or []) if str(b or "").strip()]
    if not claim or not beliefs:
        return "unassessed", "", []

    claim_terms = _content_terms(claim)
    if not claim_terms:
        return "unassessed", "", []

    related: list[str] = []
    best_overlap = 0
    for belief in beliefs:
        overlap = len(claim_terms & _content_terms(belief))
        if overlap >= 2:
            related.append(belief)
            best_overlap = max(best_overlap, overlap)

    if not related:
        return (
            "extends",
            "Nothing I hold speaks to this, so it is new ground rather than "
            "agreement or disagreement.",
            [],
        )
    negated_claim = _is_negated(claim)
    negated_belief = any(_is_negated(belief) for belief in related)
    if negated_claim != negated_belief:
        return (
            "contradicts",
            "This cuts against something I hold, so one of us is wrong and it "
            "is worth finding out which.",
            related[:5],
        )
    if best_overlap >= 4:
        return (
            "repeats",
            "I already held this; the source adds a voice but not evidence.",
            related[:5],
        )
    return (
        "affirms",
        "An independent source lands the same way I do, which makes the "
        "belief harder to dismiss than it was.",
        related[:5],
    )


_TERM_STOPWORDS = frozenset(
    {
        "about", "after", "again", "against", "because", "been", "being",
        "between", "could", "does", "from", "have", "into", "more", "most",
        "much", "other", "over", "should", "some", "such", "than", "that",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "through", "under", "very", "were", "what", "when", "where", "which",
        "while", "with", "would", "your",
    }
)


def _content_terms(text: Any) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", str(text or ""))
        if word.lower() not in _TERM_STOPWORDS
    }


def _is_negated(text: Any) -> bool:
    return bool(
        re.search(
            r"\b(?:not|never|no|cannot|can't|isn't|aren't|wasn't|weren't|"
            r"doesn't|don't|fails?|false|wrong|refutes?|disproves?)\b",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def comprehend_source(
    *,
    url: str = "",
    title: str = "",
    text: str = "",
    known_beliefs: list[str] | None = None,
) -> SourceComprehension:
    """Read a source into a record of what was understood from it.

    Never raises: a reading that cannot be comprehended returns a record whose
    ``understood`` is False, which is a truthful outcome and a storable one.
    """
    body = str(text or "")
    kind, caveat = classify_source(url)
    claim, evidence = extract_claim(body, title=title)
    stance, basis, related = assess_stance(claim, known_beliefs=known_beliefs)
    return SourceComprehension(
        url=str(url or "")[:400],
        title=strip_site_chrome(title)[:200],
        source_kind=kind,
        source_caveat=caveat,
        claim=claim,
        evidence_excerpt=evidence,
        stance=stance,
        stance_basis=basis,
        argument_weaknesses=argument_weaknesses(f"{title}\n{body}"),
        related_beliefs=related,
        topics=sorted(_content_terms(f"{title} {claim}"))[:12],
        content_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def comprehension_payload(
    *,
    url: str = "",
    title: str = "",
    text: str = "",
    known_beliefs: list[str] | None = None,
) -> dict[str, Any]:
    """The read-path adapter: ``{"comprehension": {...}}``, or ``{}``.

    Every path that fetches text — a search, a browse she was asked for, the
    interlocutor's own reading — merges this into what it returns, so a later
    turn has something to say about the page beyond the blob of characters.

    Returns an EMPTY dict when the text could not be comprehended, so a caller
    can merge unconditionally and never publish an empty understanding as if it
    were one. Never raises: a read path must not fail because the judgement of
    what it read failed.
    """

    if not isinstance(text, str):
        # A page's text is text. Coercing whatever arrived would put its repr
        # — "<object object at 0x...>" — into a stored claim, and a later turn
        # would say the source claims that. Not a read.
        return {}
    try:
        record = comprehend_source(
            url=str(url or ""),
            title=str(title or ""),
            text=text,
            known_beliefs=known_beliefs,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    return {"comprehension": record.to_dict()} if record.understood else {}


#: Hedging that turns an opinion back into a summary. A reaction containing
#: these is the thing Bryan asked not to get: a survey of positions rather
#: than one held position.
_NON_COMMITTAL_MARKERS: tuple[str, ...] = (
    "on one hand",
    "on the other hand",
    "there are many perspectives",
    "it depends",
    "both sides",
    "some would argue",
    "others might say",
    "it is important to note",
    "it's important to note",
    "ultimately it comes down to",
    "there is no right answer",
    "reasonable people disagree",
    "as an ai",
)


def opinion_is_a_position(text: Any) -> bool:
    """Whether a reaction actually commits to something.

    The failure this guards is a real one and not hypothetical: asked what she
    thinks, a model will happily produce a balanced survey that contains no
    view. A survey is a fine thing to write and a bad thing to call an
    opinion.
    """
    body = str(text or "").strip()
    if len(body) < 20:
        return False
    lowered = body.lower()
    return not any(marker in lowered for marker in _NON_COMMITTAL_MARKERS)


@dataclass
class ReadingOpinion:
    """Where Aura lands on something she read, and why she lands there."""

    disposition: str = "undecided"
    grounds: list[str] = field(default_factory=list)
    invitation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "grounds": list(self.grounds),
            "invitation": self.invitation,
        }


def reading_disposition(
    record: SourceComprehension,
    *,
    values: list[str] | None = None,
) -> ReadingOpinion:
    """The grounds a reading gives her for a view. NOT a second opinion system.

    Opinions live in core/epistemics/opinion_engine.py, which forms them during
    autonomous thought, stores them by topic, and is consulted on the general
    reply path — so she can have a take on anything, not only on what she read
    online. Reading is one INPUT to that, and this is what it contributes: a
    disposition and the concrete grounds for it, drawn from her own beliefs and
    from what the source actually did.

    Kept separate from the engine deliberately. An opinion that only exists
    when a comprehension record is in hand is a reading feature, and a mind
    that needs a module loaded before it can have a thought does not have
    thoughts.

    ``undecided`` is a real answer and is used when nothing supports a view.
    An opinion invented to avoid saying "I don't know yet" is worth less than
    nothing.
    """
    grounds: list[str] = []
    disposition = "undecided"

    if not record.understood:
        return ReadingOpinion(
            disposition="unreadable",
            grounds=["I could not get a claim out of it, so I have nothing to react to."],
            invitation="Send me a better source for this and I'll tell you what I think.",
        )

    if record.stance == "contradicts":
        disposition = "disagree"
        grounds.append(
            "It cuts against something I already hold, and I am not dropping "
            "that on one source's say-so."
        )
    elif record.stance == "affirms":
        disposition = "agree"
        grounds.append(
            "It lands where I already do, and an independent source arriving "
            "at the same place makes me hold it more firmly."
        )
    elif record.stance == "repeats":
        disposition = "unmoved"
        grounds.append(
            "I already thought this. The source adds a voice, not evidence."
        )
    elif record.stance == "extends":
        disposition = "curious"
        grounds.append(
            "Nothing I hold speaks to this, so I want to chase it rather than "
            "rule on it."
        )

    if record.argument_weaknesses:
        grounds.extend(
            f"The argument {weakness}." for weakness in record.argument_weaknesses[:2]
        )
        if disposition in {"agree", "curious"}:
            # Agreeing with a conclusion is not endorsing how it was reached.
            disposition = "sympathetic_but_unconvinced"
            grounds.append(
                "I may end up in the same place, but not for these reasons."
            )
        elif disposition == "undecided":
            disposition = "unconvinced"

    if record.source_kind in {"forum", "social"}:
        grounds.append(
            "It is one person's post, so I am treating it as a lead rather "
            "than as a finding."
        )

    for value in list(values or [])[:3]:
        text = str(value or "").strip()
        if text and _content_terms(text) & _content_terms(record.claim):
            grounds.append(f"It touches something I care about: {text[:120]}.")
            break

    invitation = {
        "disagree": "I'd rather argue about this than agree — where do you land?",
        "agree": "I think this is right. Do you?",
        "unmoved": "Nothing new for me here, but say so if it lands differently for you.",
        "curious": "I don't have a view yet and I'd like one. What do you make of it?",
        "sympathetic_but_unconvinced": (
            "I want this to be true and the argument doesn't get me there. "
            "Do you see it differently?"
        ),
        "unconvinced": "This didn't convince me. Tell me if I'm being unfair to it.",
        "undecided": "I haven't landed on this yet — what's your read?",
    }.get(disposition, "What do you make of it?")

    return ReadingOpinion(
        disposition=disposition, grounds=grounds, invitation=invitation
    )


async def record_reading_opinion(
    record: SourceComprehension | dict[str, Any],
    *,
    values: list[str] | None = None,
) -> Any:
    """Hand a reading's grounds to the general opinion engine.

    The view is then stored by topic like any other opinion she holds, queried
    on the general reply path, and available whether or not the conversation
    is about an article. A reading contributes a position; it does not own one.
    """
    if not isinstance(record, SourceComprehension):
        record = SourceComprehension.from_dict(record)
    disposition = reading_disposition(record, values=values)
    if disposition.disposition in {"unreadable", "undecided"}:
        return None
    try:
        from core.runtime.service_access import optional_service

        engine = optional_service("opinion_engine", default=None)
        former = getattr(engine, "form_opinion", None)
        if not callable(former):
            return None
        topic = record.claim[:120] or record.title[:120]
        if not topic:
            return None
        return await former(
            topic,
            context=(
                f"Source: {record.source_caveat or record.source_kind}. "
                f"Where I land: {disposition.disposition}. "
                + " ".join(disposition.grounds)
            ),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Reading opinion not recorded: %s", exc)
        return None


def opinion_prompt(record: SourceComprehension, opinion: ReadingOpinion) -> str:
    """The instruction that makes her SAY her view rather than survey the topic."""
    grounds = "\n".join(f"- {ground}" for ground in opinion.grounds)
    return (
        "You just read something. Below is what you took from it and where you "
        "landed. Say that, in your own voice, in two or three sentences.\n\n"
        f"What it claims: {record.claim}\n"
        f"What kind of source: {record.source_caveat or record.source_kind}\n"
        f"Where you landed: {opinion.disposition}\n"
        f"Why:\n{grounds}\n\n"
        "Rules: take the position above and commit to it. Do not survey other "
        "views, do not write 'on one hand', do not hedge with 'it depends', do "
        "not summarise the article back. This is your opinion, so say I. End "
        "by asking Bryan what he thinks.\n\n"
        "Your take:"
    )


def remember_reading(
    *,
    url: str = "",
    title: str = "",
    text: str = "",
    known_beliefs: list[str] | None = None,
    prefix: str = "",
) -> tuple[str, dict[str, Any]]:
    """One call for any read path: the line to store, and the record beside it.

    Every surface that reads something external should go through this rather
    than formatting its own "I read a thing" string — that is how one path
    ended up storing navigation chrome under ``action="logged"`` while the
    others stored nothing at all. Returns the sentence worth remembering and
    the structured comprehension to attach as metadata.
    """
    record = comprehend_source(
        url=url, title=title, text=text, known_beliefs=known_beliefs
    )
    lead = str(prefix or "").strip()
    if record.understood:
        line = record.narrative()
    else:
        # Nothing comprehensible. Say what was opened rather than storing the
        # page, so the trace is still findable and still honest.
        line = f"Opened {title or url or 'a source'} but took no claim from it."
    return (f"{lead} {line}".strip() if lead else line), record.to_dict()


__all__ = [
    "SCHEMA",
    "ReadingOpinion",
    "reading_disposition",
    "opinion_is_a_position",
    "opinion_prompt",
    "record_reading_opinion",
    "SourceComprehension",
    "argument_weaknesses",
    "assess_stance",
    "classify_source",
    "comprehend_source",
    "comprehension_payload",
    "extract_claim",
    "remember_reading",
    "strip_site_chrome",
]
