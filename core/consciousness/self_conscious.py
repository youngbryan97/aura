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
"""

from __future__ import annotations

from core.world_model.associative_novelty import Spread

__all__ = ["claim", "in_question"]


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
