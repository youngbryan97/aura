"""How a game ended, read from the words on its last screen, and whether a request asks for a win.

An end screen says who won in a sentence: "You win!", "The computer wins.",
"Game over", "Congratulations, you beat level 3", "Try again". Who won is the
subject of the winning verb. When it is the player ("you", "player") she won;
when it is anyone else she lost. A screen that names no winner but says the
game is over, or asks to try again, is a loss, and so is a player with nothing left to lose ("Lives: 0"). Two
sides' scores written out ("You 5 Computer 3") settle it by the numbers.

It abstains when the words do not say, so an unread ending is never reported
as a win.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Final

__all__ = ["asks_to_win", "how_it_ended", "how_it_ended_in", "what_it_asks_of_a_player"]

_WINNING: Final = re.compile(r"\b(win|wins|won|winner|victory|victorious|beat|beats|defeated|champion)\b")
_THE_PLAYER: Final = frozenset({"you", "player", "player 1", "p1", "your", "yours"})
_OVER: Final = re.compile(r"\b(game over|you lose|you lost|try again|out of lives|no lives|time'?s up|you died|defeat)\b")
_PRAISED: Final = re.compile(
    r"\b(congratulations|congrats|well done|you did it|you made it|"
    r"(?:level|stage|round|wave|mission|world)\s*\d*\s*(?:complete|completed|clear|cleared|passed|beaten))\b"
)
_SIDES: Final = re.compile(r"\b(you|player)\s*:?\s*(\d+)\D{1,24}?\b([a-z]+)\s*:?\s*(\d+)\b")
#: What a player has of their own to lose, at nothing: "Lives: 0", "health 0", "0 hearts".
_NOTHING_LEFT: Final = re.compile(r"\b(lives|life|health|hp|hearts|energy|chances|balls left|tries left)\s*:?\s*0\b|\b0\s+(lives|hearts|chances)\b")


def _plain(text: str) -> str:
    # A zero drawn in a game's font is read as a letter as often as a digit
    # ("Computer o", "You Ø"); standing alone where a score stands, it is one.
    value = re.sub(r"(?<=\s)[oOØø](?=\s|$)", "0", str(text or ""))
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(value.casefold().split())


def _who_won(clause: str) -> str:
    """The subject of the first winning verb in a clause, or "" where there is none."""
    found = _WINNING.search(clause)
    if not found:
        return ""
    before = clause[: found.start()].strip(" ,!.:")
    words = re.findall(r"[a-z0-9]+", before)
    subject = " ".join(words[-2:]) if len(words) >= 2 and words[-2] in ("player",) else (words[-1] if words else "")
    if found.group(1) in ("winner", "champion", "victory", "victorious"):
        after = re.findall(r"[a-z0-9]+", clause[found.end():])
        subject = subject or (after[0] if after and after[0] in ("is",) and len(after) > 1 else "")
    return subject


def how_it_ended_in(parts: list[str]) -> str:
    """How it ended, read from a screen's separate pieces of writing first.

    Text read off a screen in one run interleaves its pieces: LIVE 2026-10-04
    "You win!" in large letters over "You 5 Computer 0" came out as "You You 5
    win! Computer o Press SPACE to play again", in which nobody wins, and
    "play again" said only that it was over, so a win was taken for a loss.
    A piece that says who won, or what each side scored, is taken first; a
    screen that only says it is over is read whole.
    """
    for part in parts:
        said = _who_it_says_won(_plain(part))
        if said:
            return said
    return how_it_ended(" ".join(parts))


def _who_it_says_won(text: str) -> str:
    """"won" or "lost" where the words say who won or what each side scored; "" otherwise."""
    if not text:
        return ""
    sides = _SIDES.search(text)
    if sides and sides.group(3) not in ("to", "and", "points"):
        mine, theirs = int(sides.group(2)), int(sides.group(4))
        if mine != theirs:
            return "won" if mine > theirs else "lost"
    for clause in re.split(r"[.!?;\n]+", text):
        who = _who_won(clause)
        if not who:
            continue
        if who in _THE_PLAYER or who.endswith("you"):
            return "won"
        if not re.search(r"\b(not|didn't|did not|never)\b", clause):
            return "lost"
    return ""


def how_it_ended(words: str) -> str:
    """"won", "lost", or "" where the words do not say."""
    text = _plain(words)
    if not text:
        return ""
    sides = _SIDES.search(text)
    if sides and sides.group(3) not in ("to", "and", "points"):
        mine, theirs = int(sides.group(2)), int(sides.group(4))
        if mine != theirs:
            return "won" if mine > theirs else "lost"
    for clause in re.split(r"[.!?;\n]+", text):
        who = _who_won(clause)
        if not who:
            continue
        if who in _THE_PLAYER or who.endswith("you"):
            return "won"
        if not re.search(r"\b(not|didn't|did not|never)\b", clause):
            return "lost"
    if _PRAISED.search(text):
        return "won"
    if _OVER.search(text) or _NOTHING_LEFT.search(text):
        return "lost"
    return ""


_ASKS: Final = re.compile(
    r"\b(until (?:you|she|i) (?:win|beat)|and win\b|win (?:it|the game|against|a game|at least)|"
    r"beat (?:it|the game|them|the computer|the level|each|every|all)|to win\b|try to win|"
    r"until (?:you've|you have) won)"
)


def asks_to_win(request: str) -> bool:
    """Whether a request asks for the game to be won, not only played."""
    return bool(_ASKS.search(_plain(request)))


_FOR_POINTS: Final = re.compile(
    r"\b(as many (?:points|\w+) as (?:possible|you can)|highest score|high score|beat your (?:best|score|record)|"
    r"get the most|score as (?:many|much|high)|before (?:the )?time runs out|in (?:one|two|three|\d+) minutes?)\b"
)
_TO_WIN: Final = re.compile(r"\b(first to|wins?\b|beat (?:the|him|her|them|your opponent)|defeat|win the|to win)\b")


def what_it_asks_of_a_player(words: str) -> str:
    """What a game's own words set a player to do: "win" it, run up a "score", or "" where they do not say.

    A game that counts points against a clock has no winner to be: Toonami:
    Tunnel Rush says "you have two minutes to collect as many points as
    possible". Asked to win one, the honest end is a finished run and its
    score, not a run of losses until a clock that never declares a winner.
    """
    text = _plain(words)
    if _TO_WIN.search(text):
        return "win"
    if _FOR_POINTS.search(text):
        return "score"
    return ""
