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

__all__ = ["asks_to_choose", "confirms", "how_much_it_leads_on", "offers_a_way_on", "says_it_is_being_made_ready",
           "says_it_is_paused"]

#: Labels that go on, whole: a label that merely contains "play" ("Display")
#: is not one. "Try again" is one: LIVE 2026-10-10 a lander's end screen said OUCH! and TRY AGAIN over twinkling stars,
#: and she played the stars for three minutes.
_GOES_ON = re.compile(
    r"^(?:play(?:\s+(?:now|game|again))?|start(?:\s+game)?|begin|next|continue|resume|unpause|skip(?:\s+intro)?|ok(?:ay)?|go!?|"
    r"try\s+again\??|retry|replay|restart|again\??|"
    r"enter|let'?s\s+go|ready|i'?m\s+ready|yes|accept|done|new\s+game|easy|normal|1\s*player|one\s+player|"
    r"(?:click|press|tap)\s+(?:here\s+)?to\s+(?:play|start|begin|continue)|(?:press|hit)\s+(?:any\s+key|space|enter|start))$",
    re.IGNORECASE,
)

#: Labels that are read, not pressed: numbers, scores, measures.
_IS_READ = re.compile(r"^[\d\s.,:/%+-]+(?:pts?|points?)?$|^(?:score|time(?:\s+left)?|lives?|level|speed|health|energy|hi-?score)\b", re.IGNORECASE)

#: A measure as a game draws it: a number and its unit ("= 0 ft.", "120 m", "3 sec"), its noughts read as letters
#: as often as not; and the caption of a gauge ("Launch Meter", "Glide Meter").
_A_MEASURE = re.compile(
    r"^[=x×~]?\s*[\dOoDQ]+(?:[.,]\d+)?\s*(?:ft|feet|m|km|mi|miles?|mph|kph|sec|secs|s|min|pts?|%|x)\.?$|\b(?:meter|gauge)\b",
    re.IGNORECASE)

_A_WAY_ON = LearnedMatcher(
    name="a_way_on",
    positives=("Play", "Start Game", "Next", "Continue", "Click to play", "Skip", "OK", "Go!", "Easy", "Play Now", "Let's go", "Begin"),
    negatives=("SCORE", "Time Left", "Shield", "750 pts", "Speed", "Accelerate", "Brake", "Bombs", "Lives: 3", "Credits", "More Games", "Sound"),
    features=model_hidden_features,
)


#: The single words of the floor, for labels the screen reading got one letter wrong ("Nex", "P1ay").
_ONE_WORD = ("play", "start", "begin", "next", "continue", "resume", "skip", "enter", "ready", "easy", "normal", "done", "accept")


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


def offers_a_way_on(said: str) -> bool:
    """Whether writing on a screen includes a way on by its words alone ("START", "How to play", "Next"): a phrase of
    up to three of its words that the floor is sure goes on. No learned guess is asked: this is read every look."""
    words = _plain(said).split()
    for size in (1, 2, 3):
        for at in range(len(words) - size + 1):
            phrase = " ".join(words[at:at + size])
            if _GOES_ON.match(phrase) or _GOES_ON.match(phrase.rstrip("! ")):
                return True
    return False


def how_much_it_leads_on(label: str) -> float:
    """How much more worth trying first a click on ``label`` is than a key: above 1 a way on, below 1 a thing to read.

    A label that ends in a colon names the value beside it ("Meter:", "Lives:",
    "Time left:"): it is read, whatever its word; and so is a measure with its
    unit, and a gauge's caption: LIVE 2026-10-07 she clicked "= 0 ft." and
    "LAUNCH METER" off a game's screen to see what they did.
    """
    if str(label or "").rstrip().endswith(":") or _A_MEASURE.search(str(label or "").strip()):
        return 0.5
    plain = _plain(label)
    if not plain:
        return 1.0
    # The one of several alike controls that stands apart from the rest is the open one, the lit one, the chosen
    # one (core/perception/shapes_that_look_pressable.py): likelier the way on than any other unnamed shape.
    if plain.startswith("the one that stands out at"):
        return 1.4
    # "Play Now!" goes on as "Play Now" does: a button's exclamation mark is its tone, not its word.
    if _GOES_ON.match(plain) or _GOES_ON.match(plain.rstrip("! ")):
        _A_WAY_ON.observe(plain, holds=True)
        return 1.6
    if " " not in plain and any(_one_letter_off(plain.lower(), word) for word in _ONE_WORD):
        return 1.4
    # A short number alone is one to pick, as a level or a page is (core/skills/screen_pursuit_bearings.py
    # `_a_readout`): LIVE 2026-10-08 the one open level of a game was "1", rated a thing to read, and never pressed.
    if re.fullmatch(r"[1-9]\d?", plain):
        return 0.9
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


#: Words that ask a person to choose: a player, a ball, a level, a character.
_CHOOSE = re.compile(r"\b(?:please )?(?:select|choose|pick) (?:a |an |the |your |number )", re.I)

#: Labels that go on once a choice is made, as against the choices themselves.
_CONFIRMS = re.compile(r"^\W*(?:next|continue|ok|okay|done|go|confirm|start|play|begin|enter|let'?s go|ready)\W*$", re.I)


def asks_to_choose(said: str) -> bool:
    """Whether a screen's words ask a person to choose something: a player, a ball, a level, a character."""
    return bool(_CHOOSE.search(said or ""))


#: A screen that says the thing is stopped and waits to be let go on: "PAUSED", "Game Paused", "Pause Menu".
_PAUSED = re.compile(r"\b(?:paused|pause\s+menu)\b", re.I)
#: Words that say a screen is still being made ready: what is on it is not yet the thing, and is waited out.
_BEING_MADE_READY = re.compile(r"\b(?:loading|downloading|launching (?:the )?emulator|initiali[sz]ing|please wait|"
                               r"buffering|starting up|one moment)\b", re.I)


def says_it_is_being_made_ready(said: str) -> bool:
    """Whether a screen's words say it is still being made ready ("Downloading game metadata...", "Loading 40%"), so
    that nothing on it is pressed until it says otherwise."""
    return bool(_BEING_MADE_READY.search(said or ""))


def says_it_is_paused(said: str) -> bool:
    """Whether a screen's words say what was going on is stopped until she lets it go on: what stopped it lets it go
    on (core/skills/screen_pursuit_decision.py `_go_on_from_a_pause`)."""
    return bool(_PAUSED.search(said or ""))


def confirms(label: str) -> bool:
    """Whether a label is what goes on after a choice ("next", "OK", "start"), not a choice."""
    return bool(_CONFIRMS.match(str(label or "")))
