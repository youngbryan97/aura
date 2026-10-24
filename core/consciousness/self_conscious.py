"""A feeling about herself: her arousal, with her self as its object.

Her self never reached her attention. Across the 2,960 turns of seed 7 at
`82549c823` the workspace was won by deliberation, memory, the world model, the
body and the lifetime state, and not once by the self. The one bid that put her
self forward was priced on identity instability, and her stability read 1.0 on
every one of those turns, so the bid never entered. Feeling and self met at
attention only through gains with arbitrary directions, so when both were set
the competition went somewhere different every time and nowhere in particular
(the paired test of 21 October: an interaction of 17 whitened units, direction
p 0.99).

People attend to themselves most when two things happen together. Self-focused
attention rises with how far the self has just fallen short of itself, and more
so when feeling runs high (Duval and Wicklund 1972; Ingram 1990; Mor and
Winquist 2002). The self-conscious emotions need both: arousal, and a self that
is in question (Tracy and Robins 2004). The two-factor account says which part
each supplies (Schachter and Singer 1962): arousal is how strong the feeling
is, and what it is about comes from elsewhere. Arousal with nothing about the
self in question is not a feeling about herself; a self in question while she
is calm is a thought she can set aside.

So this feeling bids at her arousal, as far as her self is more in question
than she is used to. How unsettled she usually is is a running level and spread
(`associative_novelty.Spread`) kept by the workspace, so it forks with the
competition it prices.

The bid seldom wins. Every bid's own priority is capped at one, and the sources
she attends to most score a third above that on what she has come to feel
about them and what winning them has earned, so a feeling about herself that
has never won only shows where the competition is close: on her organism A,S
-> G read direction p 0.0077, 0.0086, 0.019 and 0.124 across four probes on
23 and 24 October. What self-focus under arousal does to the rest of thinking
does not wait on winning. Self-consciousness under pressure turns a person to
monitoring themselves explicitly, which narrows attention and makes processing
deliberate where it was fluent (Baumeister, Journal of Personality and Social
Psychology 46, 1984; Beilock and Carr, Journal of Experimental Psychology:
General 130, 2001). `monitoring` is that: as far as her self is in question
and she is stirred above her usual, how focused and how deep the next thought
is allowed to be move toward the top of their ranges, and how creative and how
hot toward the bottom, each by that share of the way.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from core.world_model.associative_novelty import Spread

__all__ = ["claim", "in_question", "monitoring"]

#: Which end of its range explicit monitoring pulls each modifier toward.
_MONITORED: dict[str, bool] = {
    "focus_mod": True,
    "depth_mod": True,
    "creativity_mod": False,
    "temperature_mod": False,
}


def in_question(stability: float, usual: Spread | None) -> float:
    """How far her self is in question beyond what she is used to, in [0, 1).

    A self that has never wavered has no spread, and then any departure is as
    far beyond it as a departure can be.
    """
    unsettled = 1.0 - max(0.0, min(1.0, float(stability)))
    return (usual or Spread()).beyond(unsettled)


def claim(arousal: float, question: float) -> float:
    """The feeling's claim on attention: how strongly it is felt, times whether it is about her."""
    return max(0.0, min(1.0, float(arousal))) * max(0.0, min(1.0, float(question)))


def monitoring(modifiers: MutableMapping[str, Any], share: float) -> float:
    """Move each modifier ``share`` of the way to the end of its range that self-monitoring pulls it to.

    Within the range homeostasis holds it to
    (`homeostatic_coupling.MODIFIER_BOUNDS`). Returns the share applied; a
    modifier that is not there is left out.
    """
    from core.consciousness.homeostatic_coupling import MODIFIER_BOUNDS

    share = max(0.0, min(1.0, float(share)))
    if share <= 0.0 or not isinstance(modifiers, MutableMapping):
        return 0.0
    for name, upward in _MONITORED.items():
        if name not in modifiers:
            continue
        try:
            value = float(modifiers[name])
        # not a failure: a modifier that is not a number is left as it is.
        except (TypeError, ValueError):
            continue
        low, high = MODIFIER_BOUNDS[name]
        end = high if upward else low
        modifiers[name] = value + share * (end - value)
    return share
