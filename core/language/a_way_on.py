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

__all__ = ["asks_to_choose", "confirms", "how_much_it_leads_on", "label_is_a_way_on", "nearly_says", "offers_a_way_on",
           "says_it_is_being_made_ready", "says_it_is_paused"]

#: Labels that go on, whole: a label that merely contains "play" ("Display")
#: is not one. "Try again" is one: LIVE 2026-10-10 a lander's end screen said OUCH! and TRY AGAIN over twinkling stars,
#: and she played the stars for three minutes.
_GOES_ON = re.compile(
    r"^(?:play(?:\s+(?:now|game|again))?|start(?:\s+game)?|begin|next|continue|resume|unpause|skip(?:\s+intro)?|ok(?:ay)?|go!?|"
    r"try\s+again\??|retry|replay|restart|again\??|"
    # The same with what is gone on to or again ("Next Level", "Try this level again", "Replay Round"): offline
    # 2026-10-11 a painting game's "TRY THIS LEVEL AGAIN" was read as no way on, and its end screen was never left.
    r"(?:try|play|retry|replay|restart|next)(?:\s+(?:this|the|that|your))?\s+(?:level|round|stage|game|hole|race|course|"
    r"mission|track|puzzle|page|room|day|night|wave)(?:\s+again)?\s*[>»!.]*|"
    r"enter|let'?s\s+go|ready|i'?m\s+ready|yes|accept|done|new\s+game|easy|normal|1\s*player|one\s+player|"
    # A thing made is finished by saying it is: offline 2026-10-12 an alien builder's "I'M DONE!" rated no way on, and
    # its "START OVER" was pressed instead, twenty times.
    r"i'?m\s+done|all\s+done|i'?m\s+finished|finish(?:ed)?|that'?s\s+it|"
    r"(?:click|press|tap)\s+(?:here\s+)?to\s+(?:play|start|begin|continue)|(?:press|hit)\s+(?:any\s+key|space|enter|start))$",
    re.IGNORECASE,
)

#: Labels that are read, not pressed: numbers, scores, measures.
_IS_READ = re.compile(r"^[\d\s.,:/%+-]+(?:pts?|points?)?$|^(?:score|time(?:\s+left)?|lives?|level|speed|health|energy|hi-?score)\b"
                      r"|^[a-z]+s\s+left\b"
                      # What an end announces, in its own words or a line of them: read, never pressed (LIVE 2026-10-10
                      # she clicked "GAME" and then "OVER", the two lines of a GAME OVER title).
                      r"|^(?:game\s*over|game|over|you\s+(?:win|won|lose|lost)|your\s+score|final\s+score|well\s+done|"
                      r"congratulations|level\s+(?:complete|cleared)|time'?s\s+up|the\s+end)[!.]*$", re.IGNORECASE)

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


#: Ways on of two words, for labels the screen reading got a letter or two wrong, or put a stray mark before or after;
#: and those of them that go on, not back to the start.
_TWO_WORDS = ("try again", "play again", "play game", "play now", "start game", "new game", "start over")
_ON = tuple(phrase for phrase in _TWO_WORDS if phrase != "start over")


def _letters_apart(one: str, other: str, at_most: int) -> bool:
    """Whether two short strings are within ``at_most`` letters of each other, a letter wrong, missing or added."""
    if abs(len(one) - len(other)) > at_most:
        return False
    row = list(range(len(other) + 1))
    for i, a in enumerate(one, 1):
        last, row[0] = row[0], i
        for j, b in enumerate(other, 1):
            last, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, last + (a != b))
    return row[-1] <= at_most


def nearly_says(label: str, phrases: tuple[str, ...] = _TWO_WORDS) -> str:
    """The phrase a short label nearly says, where the screen reading has it a letter or two out or with a stray mark
    or two at either end ("TRY ABAIN", "UUTRY-AGAIN", "try againfu m"); "" where it says none. LIVE 2026-10-11 a crash
    screen's TRY AGAIN was read three such ways, none was known for a way back, and she pressed keys at the screen for a
    minute and a half each time."""
    said = " ".join(re.sub(r"[^a-z ]+", " ", str(label or "").lower()).split())
    if not 4 <= len(said) <= 20:
        return ""
    squeezed = said.replace(" ", "")
    for phrase in phrases:
        whole = phrase.replace(" ", "")
        # As it stands; or with up to three stray letters before it or after it.
        tries = [squeezed] + [squeezed[cut:] for cut in (1, 2, 3)] + [squeezed[:-cut] for cut in (1, 2, 3)]
        if any(len(tried) >= len(whole) - 1 and _letters_apart(tried, whole, 1) for tried in tries):
            return phrase
    return ""


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


def label_is_a_way_on(label: str, *, from_ocr: bool = False) -> bool:
    """A whole navigation label, with optional bounded OCR tolerance.

    Exact semantic labels retain the strict grammar. A measured OCR label
    may differ by one character from one unambiguous known label. This is
    a control candidate, never evidence that it was pressed or led onward.
    """
    plain = _plain(label).rstrip('! ')
    if _GOES_ON.fullmatch(plain):
        return True
    if (not from_ocr or not 3 <= len(plain) <= 24 or str(label).rstrip().endswith(':')
            or _IS_READ.match(plain) or _A_MEASURE.search(plain)):
        return False
    said = plain.casefold()
    candidates = {word for word in _ONE_WORD if ' ' not in said and _one_letter_off(said, word)}
    candidates.update(phrase for phrase in _ON if _letters_apart(said, phrase, 1))
    return len(candidates) == 1


#: An unnamed shape, by where it is ("the shape at 95% across, 90% down").
_AT_A_PLACE = re.compile(r"^the shape at (\d+)\s*%?\s*across,?\s*(\d+)\s*%?\s*down$", re.I)


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
    # An unnamed shape in the bottom right corner is where a page of a story puts the way to the next one: LIVE
    # 2026-10-11 a comic before play had its arrow there, and she clicked a shape in the middle of the drawing.
    corner = _AT_A_PLACE.match(plain)
    if corner and int(corner.group(1)) >= 85 and int(corner.group(2)) >= 80:
        return 1.3
    # "Play Now!" goes on as "Play Now" does: a button's exclamation mark is its tone, not its word.
    if _GOES_ON.match(plain) or _GOES_ON.match(plain.rstrip("! ")):
        _A_WAY_ON.observe(plain, holds=True)
        return 1.6
    if " " not in plain and any(_one_letter_off(plain.lower(), word) for word in _ONE_WORD):
        return 1.4
    # Starting over is a way back that throws away what was made, not a way on: it is known for one when a run's end
    # is judged, and is no more to click than any other label (a trap half built kept by "Edit the trap").
    if nearly_says(plain, _ON):
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
_CHOOSE = re.compile(r"\b(?:please )?(?:select|choose|pick) (?:a |an |the |your |number )"
                     # Or the same with its space misread: offline 2026-10-12 a heading with a sparkle drawn on it read
                     # "CHOOSETYOUR CHARACTER", and the two characters under it were left unchosen.
                     r"|\b(?:select|choose|pick)\w?your\b"
                     # Or with what is chosen named bare: offline 2026-10-12 "choose difficulty" over BEGINNER, INTERMEDIATE
                     # and ADVANCED was not taken to ask anything, and its heading was clicked thirty times.
                     r"|\b(?:select|choose|pick)\s+(?:level|stage|world|character|player|mode|map|track|song|difficulty|"
                     r"team|car|ball|side|hero|skill|speed)s?\b"
                     # Or with the choices named in a row: offline 2026-10-12 "Choose Ed, Edd, or Eddy" was taken for
                     # play, and the three of them were pulled and let go at for three and a half minutes.
                     r"|\b(?:select|choose|pick)\s+[\w'.]+(?:\s*,\s*[\w'.]+)*\s*,?\s*or\s+[\w'.]+"
                     # Or a screen headed by what is chosen on it: LIVE 2026-10-11 a "LEVEL SELECT" screen was taken for
                     # play, and she pulled and let go at its levels for three minutes.
                     r"|\b(?:level|stage|world|character|player|mode|map|track|song|difficulty|team)\s+"
                     r"(?:select(?:ion)?|choice|chooser)\b", re.I)

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
