"""What a thing costs in the substrate, which she may not change.

She may change the language she thinks in. She may not change the ruler she is
measured with, and both of those are description length.

A word she made has a short NAME. Counting names, a way of building that hands
every long thing a one-word label collapses every description length in the
system and every promotion looks like a triumph. Nothing improved: the ruler
moved. That failure is available to any system that gets to define both its
representations and the measure of how complicated they are.

So there are two lengths and they answer different questions.

    what it costs to SAY        in the language as it stands now. This is the
                                search prior: shorter things are tried first,
                                and she is allowed to make things shorter.

    what it costs to BE         fully written out in the substrate she was
                                given. Nothing she admits changes it, because
                                a word made by a maker costs the maker plus
                                what went into it, all the way down.

The first may be learned. The second is the certificate. A promotion that
cannot show a saving on the second has not compressed anything — it has
renamed something.
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["what_it_costs_to_be", "what_the_language_costs_to_be"]

logger = logging.getLogger("Aura.TheRulerSheCannotMove")


def what_it_costs_to_be(word: Any, name: str = "") -> int:
    """A word's size written out in the substrate, whatever it is called now.

    A word she was given costs one. A word some maker produced costs that
    maker's own size plus the cost of everything it was given, all the way
    down — which is what it would take to write it out without the maker.
    """
    term = getattr(word, "term", None)
    if term is not None and hasattr(term, "how_long"):
        made_from = getattr(word, "words", ()) or ()
        return int(term.how_long()) + sum(
            what_it_costs_to_be(one) for one in made_from
        )
    # A word built by one of the ways of building that predate the algebra
    # names its parts, and its parts are what it costs.
    steps = getattr(word, "steps", None)
    if steps:
        return sum(what_it_costs_to_be(one) for one in steps)
    inner = getattr(word, "word", None)
    if inner is not None:
        times = int(getattr(word, "times", 1) or 1)
        return max(1, times) * what_it_costs_to_be(inner)
    # Anything else is a word she was given, or one read off examples, and
    # both are one thing in the substrate.
    return 1


def what_the_language_costs_to_be(words: dict[str, Any] | None = None) -> int:
    """What the whole vocabulary would cost written out. Never falls by renaming."""
    if words is None:
        from core.cognition.an_invented_kind import addressings

        words = addressings()
    return sum(what_it_costs_to_be(word, name) for name, word in words.items())
