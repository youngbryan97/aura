"""What she felt when a memory was made, kept with it, and what a percept finds there.

A memory stored only its valence: one number for a whole feeling. Recall
weighed that number against her mood and against nothing that had just
arrived, so the percept in front of her joined the search as one appended word
among the question's twenty. In the content runs a threat, an error and a
disconnection brought back exactly the same memories from the same fork, because
nothing she had stored used any of those words and nothing else about the
percept reached the search.

A percept arrives already appraised. The affect table says which emotions each
kind moves, and that appraisal is part of perceiving it, not something decided
afterwards. Memory is cued by feeling as well as by content: what was learned in
a state comes back more readily in that state (Bower, "Mood and memory",
American Psychologist 36, 1981), and in the context models of free recall the
emotional state at encoding is part of the context retrieval is cued by (Talmi,
Lohnas and Daw, "A retrieved context model of the emotional modulation of
memory", Psychological Review 126, 2019).

So two things are kept:

    felt       the emotions she held above her own baseline when the memory
               was made. Above baseline, because her resting level of
               anticipation is not something any moment made her feel.

    carries    how much of what she felt then lies on the emotions a percept
               moves now: the share of the stamped feeling on those names.

A feeling every candidate shares cannot choose between them. A cue is worth
less the more items it is bound to (Watkins and Watkins, "Buildup of proactive
inhibition as a cue-overload effect", Journal of Experimental Psychology:
Human Learning and Memory 1, 1975), and in the harness her feelings ran high
across twenty emotions at once, so every memory carried nearly the same
profile and the share on any percept's emotions was nearly the same number.
`distinctive` takes each candidate's feeling against the mean of the pool it
competes in, so what a percept finds is what sets a memory apart.

Recall uses `carries` the way it already uses word overlap with the percept.
A recollection closes the gap to full relevance in proportion to how much of
the percept it carries and how salient the percept was, so there is no weight
of its own here.

The stamp is text rather than a mapping because some stores accept only flat
metadata, and a nested value is dropped by them without saying so.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "FELT_KEY",
    "carries",
    "decode",
    "distinctive",
    "encode",
    "felt_now",
    "percept_carried",
    "stamp",
    "stamp_from_repository",
]

#: Where the stamp lives in a memory's metadata.
FELT_KEY: str = "felt"

#: Places kept per emotion. Three is below the resolution the affect phase
#: moves an emotion by in one turn, so rounding cannot change which emotions
#: a memory carries.
_PLACES: int = 3


def felt_now(affect: Any) -> dict[str, float]:
    """The emotions she holds above her own baseline, and by how much.

    Reads `emotions` and `mood_baselines` off whatever affect object it is
    given. An emotion at or below its baseline is not something this moment
    made her feel and is left out, so a memory made at rest carries nothing.
    """
    emotions = getattr(affect, "emotions", None)
    if not isinstance(emotions, Mapping):
        return {}
    baselines = getattr(affect, "mood_baselines", None)
    if not isinstance(baselines, Mapping):
        baselines = {}
    out: dict[str, float] = {}
    for name, value in emotions.items():
        try:
            above = float(value) - float(baselines.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        above = round(above, _PLACES)
        if above > 0.0:
            out[str(name)] = above
    return out


def encode(felt: Mapping[str, float]) -> str:
    """The stamp as flat text, strongest first: ``"dread:0.120 fear:0.080"``."""
    ordered = sorted(
        ((str(name), float(value)) for name, value in felt.items() if float(value) > 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    return " ".join(f"{name}:{value:.{_PLACES}f}" for name, value in ordered)


def decode(text: Any) -> dict[str, float]:
    """The mapping back out of a stamp. Anything unreadable reads as nothing felt."""
    if isinstance(text, Mapping):
        out: dict[str, float] = {}
        for name, value in text.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0.0:
                out[str(name)] = number
        return out
    out = {}
    for part in str(text or "").split():
        name, sep, value = part.partition(":")
        if not sep or not name:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if number > 0.0:
            out[name] = number
    return out


def stamp(metadata: dict[str, Any], affect: Any) -> dict[str, Any]:
    """Keep what she feels now with the memory being written.

    A stamp already present is left alone: the writer that had the state in
    hand when the memory was made knows better than a later reading of it.
    """
    if not isinstance(metadata, dict) or metadata.get(FELT_KEY):
        return metadata
    felt = felt_now(affect)
    if felt:
        metadata[FELT_KEY] = encode(felt)
    return metadata


def stamp_from_repository(metadata: Any) -> dict[str, Any]:
    """Stamp a memory from the state the repository holds now.

    The memory facade calls this on every write, so a memory made by any
    caller carries the feeling it was made in. A caller that stamped from the
    state it had in hand is left as it wrote it.
    """
    payload = metadata if isinstance(metadata, dict) else {}
    try:
        from core.container import ServiceContainer

        repo = ServiceContainer.get("state_repository", default=None)
        current = getattr(repo, "_current", None) if repo is not None else None
        affect = getattr(current, "affect", None)
        if affect is not None:
            stamp(payload, affect)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("memory_facade", exc, action="wrote the memory without the feeling it was made in")
    return payload


def distinctive(felts: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Each candidate's feeling less what the pool it competes in shares.

    Keyed as given. The mean is taken over the candidates that carry a feeling
    at all; one without a stamp has nothing to say about what is common. A
    pool of one has nothing to contrast with and keeps its feeling whole.
    """
    decoded = {key: decode(value) for key, value in felts.items()}
    carrying = [feeling for feeling in decoded.values() if feeling]
    if len(carrying) < 2:
        return decoded
    names = {name for feeling in carrying for name in feeling}
    mean = {name: sum(feeling.get(name, 0.0) for feeling in carrying) / len(carrying) for name in names}
    out: dict[str, dict[str, float]] = {}
    for key, feeling in decoded.items():
        out[key] = {
            name: above
            for name, value in feeling.items()
            if (above := value - mean.get(name, 0.0)) > 0.0
        }
    return out


def percept_carried(in_words: float, appraisal: Iterable[str], felt: Any) -> float:
    """How much of a percept a recollection carries, in its words or in its feeling.

    Taken together the way two independent chances are, so a recollection that
    carries the percept either way carries it, and one that carries it both
    ways carries it more. Without the feeling, a threat, an error and a
    disconnection brought back the same memories from the same moment whenever
    no stored text used their names.
    """
    words = max(0.0, min(1.0, float(in_words)))
    named = tuple(appraisal or ())
    feeling = carries(named, felt) if named else 0.0
    return 1.0 - (1.0 - words) * (1.0 - feeling)


def carries(named: Iterable[str], felt: Any) -> float:
    """The share of what she felt then that lies on the emotions named now.

    One when everything she felt is among them, zero when none of it is or
    when she felt nothing above her baseline.
    """
    feeling = decode(felt)
    total = sum(feeling.values())
    if total <= 0.0:
        return 0.0
    names = {str(name) for name in named}
    return max(0.0, min(1.0, sum(value for name, value in feeling.items() if name in names) / total))
