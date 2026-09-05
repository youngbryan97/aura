"""core/interiority/text_features.py — statistics of a message, not a word list.

Text is the channel Aura mostly has, and the question is what can be
read from it honestly. Not an affect lexicon: scoring a message against
forty words picked for their feeling is a lookup table, it cannot be
wrong in an interesting way, and it is what this package replaces.

What is defensible is *distributional*. First-person singular rate,
negation rate, absolutist terms, lexical diversity, sentence length,
and burst structure are the features the literature on language and
affective state actually uses, and their common property is that the
writer is not managing them. They also mean nothing in isolation: a
first-person rate of 8% is high for one person and low for another,
which is why every value here is returned raw and read against that
person's own baseline by
:class:`~core.interiority.other_minds.OtherMindsModel`.

The function-word lists below are closed-class grammar, not sentiment.
"not" and "never" are negations in every context; "always" and
"everything" are quantifiers. None of them carries a valence, and
swapping the vocabulary of the message for synonyms with the opposite
feeling leaves every number here unchanged — which is the test that
separates this from the thing it replaces.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

from core.interiority.evidence import Reading, absent, measured

#: Closed-class grammar. Negation is a syntactic operation, not a mood.
_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither",
     "nor", "cannot", "cant", "wont", "dont", "didnt", "isnt", "arent",
     "wasnt", "werent", "hasnt", "havent", "couldnt", "wouldnt", "shouldnt"}
)

#: Universal quantifiers and absolutes. Their rate rises with rigidity of
#: framing regardless of whether the framing is positive or negative,
#: which is why they are a channel and not a polarity.
_ABSOLUTES = frozenset(
    {"always", "never", "everything", "nothing", "everyone", "nobody", "all",
     "none", "every", "completely", "totally", "absolutely", "entirely",
     "constantly", "forever", "must", "impossible", "definitely"}
)

_FIRST_SINGULAR = frozenset({"i", "me", "my", "mine", "myself", "im", "ive", "id", "ill"})
_FIRST_PLURAL = frozenset({"we", "us", "our", "ours", "ourselves", "were", "weve"})
_SECOND = frozenset({"you", "your", "yours", "yourself", "youre", "youve"})

_WORD = re.compile(r"[a-z']+")
_SENTENCE = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class TextStatistics:
    """Raw counts and rates. No value here has a sign."""

    words: int
    sentences: int
    first_singular_rate: float
    first_plural_rate: float
    second_person_rate: float
    negation_rate: float
    absolute_rate: float
    type_token_ratio: float
    mean_sentence_length: float
    mean_word_length: float
    terminal_burst: float
    ellipsis_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "words": float(self.words),
            "sentences": float(self.sentences),
            "first_singular_rate": self.first_singular_rate,
            "first_plural_rate": self.first_plural_rate,
            "second_person_rate": self.second_person_rate,
            "negation_rate": self.negation_rate,
            "absolute_rate": self.absolute_rate,
            "type_token_ratio": self.type_token_ratio,
            "mean_sentence_length": self.mean_sentence_length,
            "mean_word_length": self.mean_word_length,
            "terminal_burst": self.terminal_burst,
            "ellipsis_rate": self.ellipsis_rate,
        }


def statistics(message: str) -> TextStatistics:
    """Distributional features of a message. Deterministic and lexicon-free."""
    raw = str(message or "")
    lowered = raw.lower()
    tokens = _WORD.findall(lowered)
    total = len(tokens)
    if total == 0:
        return TextStatistics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sentences = max(1, len(_SENTENCE.findall(raw)))
    unique = len(set(tokens))

    def rate(vocab: frozenset[str]) -> float:
        return sum(1 for t in tokens if t in vocab) / total

    terminal = sum(raw.count(c) for c in "!?")
    return TextStatistics(
        words=total,
        sentences=sentences,
        first_singular_rate=rate(_FIRST_SINGULAR),
        first_plural_rate=rate(_FIRST_PLURAL),
        second_person_rate=rate(_SECOND),
        negation_rate=rate(_NEGATIONS),
        absolute_rate=rate(_ABSOLUTES),
        type_token_ratio=unique / total,
        mean_sentence_length=total / sentences,
        mean_word_length=sum(len(t) for t in tokens) / total,
        terminal_burst=min(1.0, terminal / max(1.0, float(sentences))),
        ellipsis_rate=min(1.0, raw.count("...") / max(1.0, float(sentences))),
    )


def channels(message: str) -> dict[str, Reading]:
    """Statistics folded into the two channels text actually carries.

    ``lexical`` is the composite of the rates a writer does not manage.
    ``text`` is the message's own shape — how long, how varied, how
    broken up. Both are returned as measurements because they are
    measurements; what they *mean* is settled against the person's own
    baseline downstream, not here.
    """
    stats = statistics(message)
    if stats.words == 0:
        return {"lexical": absent(source="text:empty"), "text": absent(source="text:empty")}

    # Composite of the unmanaged rates. Squashed rather than clipped so a
    # very high rate stays distinguishable from a merely high one.
    lexical_raw = (
        stats.first_singular_rate * 2.0
        + stats.negation_rate * 2.5
        + stats.absolute_rate * 2.0
    )
    lexical = 1.0 - math.exp(-lexical_raw)

    # Shape: short, fragmented, punctuation-heavy messages sit high; long
    # even ones sit low. This is form, not content.
    shape_raw = (
        stats.terminal_burst
        + stats.ellipsis_rate
        + max(0.0, 1.0 - stats.mean_sentence_length / 20.0)
    ) / 3.0

    confidence = min(1.0, stats.words / 25.0)
    return {
        "lexical": measured(
            max(0.0, min(1.0, lexical)),
            source="text_features:unmanaged_rates",
            confidence=confidence,
        ),
        "text": measured(
            max(0.0, min(1.0, shape_raw)),
            source="text_features:shape",
            confidence=confidence,
        ),
    }


__all__ = ["TextStatistics", "channels", "statistics"]
