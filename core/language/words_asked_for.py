"""What a screen asks to have typed into it, where, and how it is sent: read off the screen's own words.

Screens ask for words the same way everywhere: "Enter your name", "Type your
question above", "Answer?", "Search", "Enter code". The kind of words asked for
decides what goes in: her name where a name is asked, a question where a
question is, an answer where something is asked of her. Where the field is, the
screen often says ("above", "below", "here"); how it is sent, it says too
("click Ask", "press Enter") or shows as a control beside it.

This reads exactly that, and abstains where the words ask for nothing typed. It
knows no screen, game or program.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ANSWER", "CODE", "NAME", "QUESTION", "SEARCH", "WORDS", "WordsAsked", "words_asked_for"]

NAME = "a name"
QUESTION = "a question"
ANSWER = "an answer"
SEARCH = "something to look for"
CODE = "a code"
WORDS = "words"

#: How each kind is asked for, most particular first.
_ASKED: tuple[tuple[str, re.Pattern[str]], ...] = (
    (NAME, re.compile(r"\b(?:enter|type|write|put|fill in|give)\s+(?:in\s+)?(?:your|a|the)\s+(?:first\s+|player\s+|user\s*)?name\b"
                      r"|\byour name\b|\bname\s*:", re.I)),
    (CODE, re.compile(r"\b(?:enter|type)\s+(?:the|a|your|this|in)?\s*(?:secret\s+)?(?:code|password|pin)\b"
                      r"|\b(?:code|password)\s*:", re.I)),
    (QUESTION, re.compile(r"\b(?:type|ask|enter|write)\s+(?:in\s+)?(?:your|a|any)\s+(?:yes(?:\s+or\s+|\s*/\s*)no\s+)?question"
                          r"|\byes(?:\s+or\s+|\s*/\s*)no\s+questions?\b", re.I)),
    (SEARCH, re.compile(r"\b(?:search|look up|find)\s*(?:for)?\s*:|\btype (?:to|here to) search\b", re.I)),
    (ANSWER, re.compile(r"\b(?:type|enter|write|give)\s+(?:in\s+)?(?:the|your|an?)\s+(?:answer|guess|word|solution)\b"
                        r"|\banswer\s*[?:]|\bguess the\b|\briddle me\b", re.I)),
    (WORDS, re.compile(r"\b(?:type|enter|write)\s+(?:in\s+)?(?:some|your|a|the)?\s*(?:text|words?|message|line)\b", re.I)),
)

#: Where the words say the field is, beside the words saying so.
_WHERE = re.compile(r"\b(above|below|here|to the right|to the left|beneath|underneath|over this)\b", re.I)

#: How the words say it is sent: a control to click, or a key.
_SENT_BY_CLICKING = re.compile(r"\b(?:click|press|tap|hit|push)\s+(?:on\s+)?(?:the\s+)?[\"'“]?([A-Za-z][A-Za-z ]{0,14}?)[\"'”]?(?:\s+button)?(?=[.!,;]|\s+(?:to|when|and|then)\b|$)", re.I)
_KEY_NAMES = {"enter", "return", "space", "spacebar", "space bar", "tab"}

#: Controls that send what was typed, when the screen shows one beside the field without saying.
SENDERS = ("ask", "ok", "okay", "submit", "enter", "go", "done", "send", "answer", "search", "next", "continue", "save")


@dataclass(frozen=True)
class WordsAsked:
    """What a screen asks to have typed: the kind, the words that asked, where the field is said to be (or ""), and
    the control or key that sends it (or "")."""

    kind: str
    asked_by: str
    where: str = ""
    sent_by: str = ""
    sent_by_key: str = ""


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+|\s{2,}|\d\.\s", text) if part.strip()]


def words_asked_for(text: str) -> WordsAsked | None:
    """What the screen's ``text`` asks to have typed, or None where it asks for nothing typed."""
    said = " ".join(str(text or "").split())
    if not said:
        return None
    found: tuple[str, str] | None = None
    for sentence in _sentences(str(text or "")) or [said]:
        for kind, pattern in _ASKED:
            if pattern.search(sentence):
                found = (kind, sentence)
                break
        if found:
            break
    if found is None:
        return None
    kind, sentence = found
    where = _WHERE.search(sentence)
    sent_by = ""
    sent_by_key = ""
    for clause in _sentences(said):
        clicked = _SENT_BY_CLICKING.search(clause)
        if clicked:
            control = clicked.group(1).strip().rstrip(".").lower()
            if control in _KEY_NAMES:
                sent_by_key = "return" if control in ("enter", "return") else control.replace(" bar", "").replace("bar", "")
            elif control.split()[0] in SENDERS or len(control.split()) == 1:
                sent_by = clicked.group(1).strip()
            if sent_by or sent_by_key:
                break
    return WordsAsked(kind, sentence, where.group(1).lower() if where else "", sent_by, sent_by_key)
