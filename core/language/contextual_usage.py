"""Bounded observations of how a concept is used in a particular setting.

Co-occurrence, delivery, and an explicitly identified referent are different
relations. None of them alone establishes a definition or a cultural meaning.
The language substrate keeps the observation so a later interpretation can
be tested against use that was not available when it was proposed.
"""

from __future__ import annotations

import math
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"\w+(?:['\u2019-]\w+)*", re.UNICODE)
_STRETCH = re.compile(r"(.)\1{2,}", re.UNICODE)
_MAX_TERMS = 128
_MAX_CUES = 16
_MAX_TERM_CHARS = 64
_MAX_CONTEXT_CHARS = 128


def lexical_terms(text: str) -> tuple[str, ...]:
    """Keep surface words without equating nearby concepts or spelling variants."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return tuple(token.casefold()[:_MAX_TERM_CHARS]
                 for token in _sample_tokens(_TOKEN.findall(normalized)))


def _sample_tokens(tokens: list[str]) -> list[str]:
    """Bound retention while preserving the opening, middle, and conclusion."""
    if len(tokens) <= _MAX_TERMS:
        return tokens
    edge = _MAX_TERMS // 4
    selected = set(range(edge)) | set(range(len(tokens) - edge, len(tokens)))
    frequencies = Counter(token.casefold() for token in tokens)
    interior = range(edge, len(tokens) - edge)
    rare = sorted(interior, key=lambda index: (frequencies[tokens[index].casefold()], index))
    selected.update(rare[:_MAX_TERMS // 4])
    stride_slots = _MAX_TERMS - len(selected)
    selected.update(edge + (index * (len(tokens) - 2 * edge)) // stride_slots
                    for index in range(stride_slots))
    if len(selected) < _MAX_TERMS:
        selected.update(index for index in interior if len(selected) < _MAX_TERMS)
    return [tokens[index] for index in sorted(selected)[:_MAX_TERMS]]


@dataclass(frozen=True, slots=True)
class UsageCue:
    """One measured cue; its interpretation is deliberately absent."""

    name: str
    value: bool | int | float | str
    channel: str
    source_id: str
    observed_at: float

    def __post_init__(self) -> None:
        if (not self.name or not self.channel or not self.source_id
                or len(self.name) > _MAX_TERM_CHARS
                or len(self.channel) > _MAX_TERM_CHARS
                or len(self.source_id) > _MAX_CONTEXT_CHARS
                or type(self.value) not in (bool, int, float, str)
                or (type(self.value) is str and len(self.value) > _MAX_CONTEXT_CHARS)
                or (type(self.value) in (int, float) and not math.isfinite(self.value))
                or not math.isfinite(self.observed_at) or self.observed_at < 0):
            raise ValueError("a usage cue needs a bounded measured source")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "channel": self.channel,
                "source_id": self.source_id, "observed_at": self.observed_at}


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One source's lexical exposure, with no automatic sense assignment."""

    source_id: str
    context_id: str
    terms: tuple[str, ...]
    observed_at: float = field(default_factory=time.time)
    setting: str = ""
    community: str = ""
    speaker: str = ""
    cues: tuple[UsageCue, ...] = ()
    referents: tuple[str, ...] = ()
    stretched_terms: tuple[str, ...] = ()
    original_token_count: int = 0

    def __post_init__(self) -> None:
        if (not self.source_id or not self.context_id or not self.terms
                or len(self.terms) > _MAX_TERMS or len(self.cues) > _MAX_CUES
                or not math.isfinite(self.observed_at) or self.observed_at < 0
                or any(len(value) > _MAX_CONTEXT_CHARS for value in (
                    self.source_id, self.context_id, self.setting, self.community,
                    self.speaker))
                or any(not term or len(term) > _MAX_TERM_CHARS for term in (
                    *self.terms, *self.referents, *self.stretched_terms))
                or len(self.referents) > 16 or len(self.stretched_terms) > 16):
            raise ValueError("usage needs bounded source, context, and lexical evidence")
        if self.original_token_count < len(self.terms) or self.original_token_count > 100_000:
            raise ValueError("usage token count is inconsistent with retained evidence")
        if any(abs(cue.observed_at - self.observed_at) > 300 for cue in self.cues):
            raise ValueError("usage cue is not from this observation window")

    @classmethod
    def from_text(
        cls, source_id: str, context_id: str, text: str, *,
        observed_at: float | None = None, setting: str = "", community: str = "",
        speaker: str = "", cues: tuple[UsageCue, ...] = (),
        referents: tuple[str, ...] = (),
    ) -> UsageEvent:
        """Read text form without guessing the speaker's motive or emotion."""
        timestamp = time.time() if observed_at is None else float(observed_at)
        full_surface = _TOKEN.findall(unicodedata.normalize("NFKC", text))
        surface = _sample_tokens(full_surface)
        stretched = tuple(dict.fromkeys(token.casefold()[:_MAX_TERM_CHARS]
                                            for token in surface if _STRETCH.search(token)))[:16]
        return cls(source_id, context_id, lexical_terms(text), timestamp,
                   setting, community, speaker, cues,
                   tuple(term.casefold() for term in referents), stretched,
                   len(full_surface))

    def features_for(self, term: str) -> dict[str, Any]:
        """One bounded context view for the existing semantic experiment lane."""
        key = term.casefold()
        if key not in self.terms and key not in self.referents:
            raise ValueError("term was neither spoken nor independently identified")
        counts = Counter(self.terms)
        features: dict[str, Any] = {
            "usage:spoken": key in counts,
            "usage:frequency": min(counts[key], 16),
            "usage:stretched": key in self.stretched_terms,
            "usage:indirect_referent": key in self.referents and key not in counts,
            "usage:sampled": self.original_token_count > len(self.terms),
        }
        if self.setting:
            features["context:setting"] = self.setting
        if self.community:
            features["context:community"] = self.community
        if self.speaker:
            features["context:speaker"] = self.speaker
        for neighbor in dict.fromkeys(token for token in self.terms if token != key):
            if len(features) >= 48:
                break
            features[f"co:{neighbor}"] = True
        for cue in self.cues:
            if len(features) >= 64:
                break
            features[f"cue:{cue.channel}:{cue.name}"] = cue.value
        return features

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "context_id": self.context_id,
                "terms": list(self.terms), "observed_at": self.observed_at,
                "setting": self.setting, "community": self.community,
                "speaker": self.speaker, "cues": [cue.to_dict() for cue in self.cues],
                "referents": list(self.referents),
                "stretched_terms": list(self.stretched_terms),
                "original_token_count": self.original_token_count}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UsageEvent:
        return cls(
            str(payload["source_id"]), str(payload["context_id"]),
            tuple(payload["terms"]), float(payload["observed_at"]),
            str(payload.get("setting", "")), str(payload.get("community", "")),
            str(payload.get("speaker", "")),
            tuple(UsageCue(**raw) for raw in payload.get("cues", ())),
            tuple(payload.get("referents", ())),
            tuple(payload.get("stretched_terms", ())),
            int(payload.get("original_token_count", len(payload["terms"]))),
        )


@dataclass(frozen=True, slots=True)
class MeaningFeedback:
    """A later, attributed interpretation of one earlier usage."""

    source_id: str
    usage_source_id: str
    term: str
    sense: str
    origin: str
    observed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if (not all((self.source_id, self.usage_source_id, self.term,
                     self.sense, self.origin)) or self.source_id == self.usage_source_id
                or self.origin not in {"user_correction", "observed_referent",
                                       "verified_source"}
                or any(len(value) > _MAX_CONTEXT_CHARS for value in (
                    self.source_id, self.usage_source_id, self.term, self.sense))
                or not math.isfinite(self.observed_at)):
            raise ValueError("meaning feedback needs independent, attributed evidence")

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "usage_source_id": self.usage_source_id,
                "term": self.term, "sense": self.sense, "origin": self.origin,
                "observed_at": self.observed_at}


__all__ = ["MeaningFeedback", "UsageCue", "UsageEvent", "lexical_terms"]
