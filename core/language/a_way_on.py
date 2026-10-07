"""Whether a label drawn on a screen is a way on: what to press to get from a menu into the thing itself.

A game's title, its instructions and its menus stand still and wait to be
told to go on. Every word on them can be clicked, and most do nothing:
"SCORE", "Shield", "750 pts" are things to read. A person reads the screen
and presses the one that goes on: Play, Start, Next, Continue, Easy, OK.
LIVE-like 2026-10-05 she clicked a game's every label in turn, three seconds
each, before reaching "Next".

The words below are a floor for the ways on somebody thought of; the learned
surface (core/language/learned_matcher.py) settles the ones nobody did, and
every label the floor is sure about teaches it.
"""
from __future__ import annotations

import re

from core.language.learned_matcher import LearnedMatcher
from core.language.model_features import model_hidden_features

__all__ = ["how_much_it_leads_on"]

#: Labels that go on, whole: a label that merely contains "play" ("Display")
#: is not one.
_GOES_ON = re.compile(
    r"^(?:play(?:\s+(?:now|game|again))?|start(?:\s+game)?|begin|next|continue|skip(?:\s+intro)?|ok(?:ay)?|go!?|"
    r"enter|let'?s\s+go|ready|i'?m\s+ready|yes|accept|done|new\s+game|easy|normal|1\s*player|one\s+player|"
    r"(?:click|press|tap)\s+(?:here\s+)?to\s+(?:play|start|begin|continue)|(?:press|hit)\s+(?:any\s+key|space|enter|start))$",
    re.IGNORECASE,
)

#: Labels that are read, not pressed: numbers, scores, measures.
_IS_READ = re.compile(r"^[\d\s.,:/%+-]+(?:pts?|points?)?$|^(?:score|time(?:\s+left)?|lives?|level|speed|health|energy|hi-?score)\b", re.IGNORECASE)

_A_WAY_ON = LearnedMatcher(
    name="a_way_on",
    positives=("Play", "Start Game", "Next", "Continue", "Click to play", "Skip", "OK", "Go!", "Easy", "Play Now", "Let's go", "Begin"),
    negatives=("SCORE", "Time Left", "Shield", "750 pts", "Speed", "Accelerate", "Brake", "Bombs", "Lives: 3", "Credits", "More Games", "Sound"),
    features=model_hidden_features,
)


#: The single words of the floor, for labels the screen reading got one letter wrong ("Nex", "P1ay").
_ONE_WORD = ("play", "start", "begin", "next", "continue", "skip", "enter", "ready", "easy", "normal", "done", "accept")


def _one_letter_off(word: str, other: str) -> bool:
    """Whether ``word`` is ``other`` with one letter wrong, missing or added."""
    if abs(len(word) - len(other)) > 1 or len(other) < 4:
        return False
    if len(word) == len(other):
        return sum(a != b for a, b in zip(word, other, strict=True)) <= 1
    short, long = sorted((word, other), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


def _plain(label: str) -> str:
    return " ".join(re.sub(r"[^\w' !]+", " ", str(label or "")).split()).strip()


def how_much_it_leads_on(label: str) -> float:
    """How much more worth trying first a click on ``label`` is than a key: above 1 a way on, below 1 a thing to read.

    A label that ends in a colon names the value beside it ("Meter:", "Lives:",
    "Time left:"): it is read, whatever its word.
    """
    if str(label or "").rstrip().endswith(":"):
        return 0.5
    plain = _plain(label)
    if not plain:
        return 1.0
    if _GOES_ON.match(plain):
        _A_WAY_ON.observe(plain, holds=True)
        return 1.6
    if " " not in plain and any(_one_letter_off(plain.lower(), word) for word in _ONE_WORD):
        return 1.4
    if _IS_READ.match(plain) or len(plain) > 40:
        if len(plain) <= 40:
            _A_WAY_ON.observe(plain, holds=False)
        return 0.5
    learned = _A_WAY_ON.decide_without_waiting(plain)
    if learned is True:
        return 1.3
    if learned is False:
        return 0.6
    return 0.8
