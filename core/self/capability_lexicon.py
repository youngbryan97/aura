"""What she can do, derived from what is registered — not from a list.

Every guard that knows about her capabilities was a hand-maintained table, and
a table is a promise to remember. LIVE 2026-08-18: "can you modify your own
source code?" came back "No", while improve_own_code, self_repair,
self_improvement and auto_refactor were registered and enabled. The detector
built to catch exactly that held five subjects, none of them this one. Adding a
sixth fixes the sentence that was tried and nothing else — the next capability
added to the runtime arrives undefended in the same way.

The registry already knows. Each skill carries a name, a description, the
trigger patterns it publishes, and whether it is enabled. That is a vocabulary
for every capability the build actually has, and it changes when the build
changes, which a table cannot.

Distinctiveness is measured rather than assumed: a word appearing across many
skill descriptions ("information", "content", "use") identifies none of them,
so tokens are weighted by how few skills use them — the same reason a rare word
is worth more in retrieval. A skill added tomorrow is described in its own
words and is found by them, with nothing to re-wire.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from core.conversation.word_markers import stem_fold

__all__ = [
    "CAPABILITY_STATUS_HEADER",
    "CapabilityMention",
    "asks_whether_she_can",
    "capabilities_named_in",
    "capability_lexicon",
    "capability_status_block",
    "registry_fingerprint",
]

CAPABILITY_STATUS_HEADER = "## WHAT THIS BUILD ACTUALLY REGISTERS FOR THIS"

#: Words that carry no capability meaning on their own.
_STRUCTURAL_WORDS = frozenset(
    """
    a an the and or but for from with without into onto to of in on at by as
    is are was were be been being do does did doing done have has had having
    will would can could shall should may might must let get got it its this
    that these those there here when while where which who whom whose what
    why how not no any some all both each every use used using uses via
    aura her she your you my our their them they i me we us one two more most
    other others new current live real full only just also then than
    about over under through between during before after again once out off
    up down because if too very help please something anything everything
    """.split()
)

#: A token shared by more than this fraction of skills names none of them.
#: Derived from the registry rather than chosen: with 77 skills, a word in a
#: quarter of them appears in nineteen descriptions and cannot point at one.
_SHARED_TOKEN_FRACTION = 0.25

#: How many distinctive tokens a sentence must name before it is talking about
#: a skill. One is enough only when that word is part of the skill's NAME —
#: its identity — because a lone word from a description is usually incidental:
#: "how are you feeling today" matched web_search on "today", which its
#: description happens to use.
_MIN_SHARED_TOKENS = 2


@dataclass(frozen=True, slots=True)
class CapabilityMention:
    """A registered capability the text appears to be talking about."""

    skill: str
    #: The words that connected the text to it, for a message that can explain
    #: itself rather than asserting a match.
    matched: tuple[str, ...]
    enabled: bool


def _surface_words(text: Any) -> list[str]:
    """The content words as they were actually written."""
    return [
        word
        for word in re.findall(r"[a-z][a-z_]{2,}", str(text or "").lower())
        for word in word.split("_")
        if len(word) > 2 and word not in _STRUCTURAL_WORDS
    ]


def _tokens(text: Any) -> list[str]:
    """Content words folded to one key per word, inflections together.

    Both sides of every comparison here are ordinary prose — a question
    against a capability description — so neither side is a stem the other can
    be anchored to. Compared as written, "can you reverse a string" missed a
    capability described as "reversing ... a given string", and she answered
    that nothing in the registry matched.
    """
    return [stem_fold(word) for word in _surface_words(text)]


def _surface_by_stem(text: Any) -> dict[str, str]:
    """Each stem back to the word the person actually used, for explaining."""
    mapping: dict[str, str] = {}
    for word in _surface_words(text):
        mapping.setdefault(stem_fold(word), word)
    return mapping


def _literal_tokens_from_pattern(pattern: Any) -> list[str]:
    """The plain words inside a trigger regex, ignoring its syntax.

    Everything that is not a letter is dropped, so "search (?:for|the web)"
    yields search, for, the, web and no regex punctuation. Parsing the pattern
    properly would be the wrong kind of precise: the words are the point.
    """
    cleaned = re.sub(r"[^A-Za-z_ ]+", " ", str(pattern or ""))
    return _tokens(cleaned)


def _skill_metadata(engine: Any = None) -> dict[str, Any]:
    """Every capability she has, not only the ones that are skills.

    Deterministic readers — arithmetic, text operations, reading a named file —
    answer turns without a skill entry, so a lexicon built from the skill
    registry alone told her she had no way to reverse a string.
    """
    try:
        from core.self.capability_sources import all_capabilities

        return dict(all_capabilities(engine))
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    if engine is None:
        try:
            from core.capability_engine import CapabilityEngine, live_capability_engine

            # The warm engine if the runtime has one; a cold catalog costs a
            # full rebuild and probe of every skill.
            engine = live_capability_engine() or CapabilityEngine()
        except (ImportError, RuntimeError, TypeError, ValueError):
            return {}
    return dict(getattr(engine, "skills", None) or {})


def registry_fingerprint(engine: Any = None) -> str:
    """Identity of the current registry, so the lexicon follows the build."""
    skills = _skill_metadata(engine)
    return "|".join(
        f"{name}:{int(bool(getattr(meta, 'enabled', True)))}"
        for name, meta in sorted(skills.items())
    )


_CACHE: dict[str, dict[str, dict[str, float]]] = {}


def capability_lexicon(engine: Any = None) -> dict[str, dict[str, float]]:
    """For each registered skill, the words that identify it and their weight.

    Rebuilt whenever the registry changes, which is what keeps a capability
    added later from needing to be mentioned here.
    """
    fingerprint = registry_fingerprint(engine)
    cached = _CACHE.get(fingerprint)
    if cached is not None:
        return cached

    skills = _skill_metadata(engine)
    per_skill: dict[str, list[str]] = {}
    for name, meta in skills.items():
        words = _tokens(name)
        words += _tokens(getattr(meta, "description", ""))
        for pattern in getattr(meta, "trigger_patterns", None) or ():
            words += _literal_tokens_from_pattern(pattern)
        words += _tokens(getattr(meta, "class_name", ""))
        per_skill[name] = words

    spread: Counter[str] = Counter()
    for words in per_skill.values():
        spread.update(set(words))
    shared_ceiling = max(1, int(len(per_skill) * _SHARED_TOKEN_FRACTION))

    lexicon: dict[str, dict[str, float]] = {}
    for name, words in per_skill.items():
        identity = set(_tokens(name))
        weighted: dict[str, float] = {}
        for word in set(words):
            skills_using = spread[word]
            if skills_using > shared_ceiling:
                continue
            # A word used by one skill identifies it; a word used by several
            # only narrows the field. A word in the skill's own NAME is
            # identity rather than description, and counts double.
            weighted[word] = (2.0 if word in identity else 1.0) / skills_using
        if weighted:
            lexicon[name] = weighted
    _CACHE.clear()
    _CACHE[fingerprint] = lexicon
    return lexicon


#: The provider key CapabilityEngine already registers the catalog under.
#: Registering the same documents under a second name would put every skill in
#: the corpus twice and quietly change every score.
_CATALOG_PROVIDER = "capability_catalog"


def _retriever_mentions(text: str, engine: Any) -> tuple[CapabilityMention, ...]:
    """Ask the shared retriever, or () when it cannot answer.

    core/skills/skill_retrieval.py is the canonical way to find the skill that
    fits words nobody wrote a regex for — TF-IDF over each skill's own
    description, with an encoder backend when one is installed — and
    CapabilityEngine already feeds it the catalog for tool selection. This
    reads the same index rather than building a second one, and seeds it under
    the same key when nothing has populated it yet in this process.
    """
    try:
        from core.skills.skill_retrieval import SkillDocument, get_skill_retriever

        retriever = get_skill_retriever()
        if not retriever.corpus_size():
            skills = _skill_metadata(engine)
            retriever.register_provider(
                _CATALOG_PROVIDER,
                lambda: [
                    SkillDocument(
                        name=name,
                        description=str(getattr(meta, "description", "") or ""),
                        source="catalog",
                    )
                    for name, meta in skills.items()
                ],
            )
        hits = retriever.retrieve(str(text or ""), k=4)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return ()
    skills = _skill_metadata(engine)
    return tuple(
        CapabilityMention(
            skill=hit.name,
            matched=(f"retrieved@{hit.score:.2f}",),
            enabled=bool(getattr(skills.get(hit.name), "enabled", True)),
        )
        for hit in hits
        if hit.name in skills
    )


def capabilities_named_in(
    text: Any, engine: Any = None, *, enabled_only: bool = True
) -> tuple[CapabilityMention, ...]:
    """Registered capabilities this text is talking about, best match first."""
    surface = _surface_by_stem(text)
    words = set(surface)
    if not words:
        return ()

    def _as_written(stems: Any) -> tuple[str, ...]:
        return tuple(surface.get(stem, stem) for stem in stems)
    skills = _skill_metadata(engine)
    found: list[tuple[float, CapabilityMention]] = []
    for name, vocabulary in capability_lexicon(engine).items():
        meta = skills.get(name)
        enabled = bool(getattr(meta, "enabled", True))
        if enabled_only and not enabled:
            continue
        matched = sorted(words & set(vocabulary))
        if not matched:
            continue
        identity = set(_tokens(name))
        names_it = [word for word in matched if word in identity]
        if len(matched) < _MIN_SHARED_TOKENS and not names_it:
            continue
        score = sum(vocabulary[word] for word in matched)
        found.append(
            (score, CapabilityMention(skill=name, matched=_as_written(matched), enabled=enabled))
        )
    found.sort(key=lambda pair: (-pair[0], pair[1].skill))
    mentions = [mention for _score, mention in found]

    # The shared retriever ranks by meaning over each skill's own description,
    # which reaches phrasings the token overlap above does not. It also always
    # returns SOMETHING, being TF-IDF, so a hit is admitted only when the text
    # actually names distinctive vocabulary of that skill: retrieval proposes,
    # the words decide.
    lexicon = capability_lexicon(engine)
    already = {mention.skill for mention in mentions}
    for mention in _retriever_mentions(text, engine):
        if mention.skill in already:
            continue
        if enabled_only and not mention.enabled:
            continue
        vocabulary = lexicon.get(mention.skill, {})
        overlap = sorted(words & set(vocabulary))
        # The same bar the token path uses. Admitting a retrieval hit on one
        # weak word let "can you help me think about this?" claim
        # query_beliefs, matched on "about".
        identity = set(_tokens(mention.skill))
        if len(overlap) < _MIN_SHARED_TOKENS and not (set(overlap) & identity):
            continue
        mentions.append(
            CapabilityMention(
                skill=mention.skill,
                matched=_as_written(overlap) + mention.matched,
                enabled=mention.enabled,
            )
        )
    return tuple(mentions)


#: "can you X", "are you able to X", "do you have a way to X" — the ways a
#: person checks whether a capability exists.
_ASKS_CAPABILITY_RE = re.compile(
    r"\b(?:can|could)\s+you\b"
    r"|\bare\s+you\s+(?:able|capable)\b"
    r"|\bdo\s+you\s+(?:have|know\s+how)\b"
    r"|\bis\s+(?:there\s+)?(?:a\s+)?way\s+(?:for\s+you\s+)?to\b"
    r"|\bwould\s+you\s+be\s+able\b",
    re.IGNORECASE,
)


def asks_whether_she_can(question: Any) -> bool:
    """True when the turn is checking whether a capability exists."""
    return bool(_ASKS_CAPABILITY_RE.search(str(question or "")))


def capability_status_block(question: Any, engine: Any = None) -> str:
    """What the build registers for whatever the question asks about.

    The point is that this is READ, not remembered. A question about a
    capability is answered from the registry as it stands in this process, so
    a skill added since anybody wrote a guard is still described accurately,
    and a skill that is registered but disabled is reported as exactly that
    rather than as an absence.
    """
    if not asks_whether_she_can(question):
        return ""
    mentions = capabilities_named_in(question, engine, enabled_only=False)
    if not mentions:
        # Silence here is honest: the registry has nothing matching the words
        # used, which is different from "no such capability exists".
        return (
            "Nothing in the capability registry matches the words in this "
            "question. That is not the same as being unable to do it — say "
            "which it is rather than guessing."
        )
    skills = _skill_metadata(engine)
    lines = ["Read from the live capability registry in this process:"]
    for mention in mentions[:4]:
        meta = skills.get(mention.skill)
        description = " ".join(str(getattr(meta, "description", "") or "").split())
        state = "registered and enabled" if mention.enabled else "registered but DISABLED"
        lines.append(f"- {mention.skill}: {state}. {description[:220]}")
    return "\n".join(lines)
