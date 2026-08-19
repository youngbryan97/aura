"""Reading a goal that has to be watched out of what somebody asked for.

Most requests name an action: open this, write that, move the other. Some
name a condition instead — keep going until, wait for it to finish, step
through until it says done — and those cannot be satisfied by any single
action. The desktop planner had no way to express one, so it planned the
first action it recognised and reported the objective complete.

What separates the two is structural, not a matter of phrasing. A watched
goal carries something to keep doing and a condition that ends it. Both have
to be present: "open Chrome until it opens" is one action, and "keep going"
with no end is a request nobody can accept. When either is missing this
returns nothing and the ordinary planner handles the request.

The condition is what the screen will say when the goal is reached, which is
the only kind of finishing test a watching loop can check. Pulling it out of
the request means a person never has to phrase it as one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: Cues that something is to be kept up rather than done once.
#:
#: Patterns rather than phrases, because the same continuation arrives in any
#: number of shapes: keep going, keep playing, keep refreshing, keep waiting.
#: A literal list of those missed "keep waiting until" and "play 2048 in
#: Chrome until" on the first try, which is the failure mode of every fixed
#: phrase list — it recognises the example it was written from.
CONTINUING: tuple[str, ...] = (
    r"\bkeep\s+(?:\w+ing|at\s+it|going)\b",
    r"\bkeeps?\s+on\b",
    r"\bplay(?:ing|s)?\b",
    r"\bstep(?:ping)?\s+through\b",
    r"\b(?:watch|wait)(?:ing|es)?\s+(?:for|until|till)\b",
    r"\bmonitor(?:ing|s)?\b",
    r"\bstick\s+with\b",
    r"\bcarry\s+on\b",
    r"\bas\s+long\s+as\b",
    r"\b(?:over\s+and\s+over|repeatedly|again\s+and\s+again)\b",
    r"\buntil\s+you\s+(?:get|reach|hit|see)\b",
)
_CONTINUING_RE = tuple(re.compile(pattern, re.IGNORECASE) for pattern in CONTINUING)

#: Words that introduce the condition which ends the watching.
UNTIL = ("until", "till", "til", "up to", "as soon as", "once you", "when you", "when it", "once it")
#: A browser's own furniture repeats page words in the tab strip and the
#: address bar, so a condition matched anywhere on screen can be satisfied by
#: the title before anything happens. Content starts below the chrome.
CHROME_BAND_TOP = 0.12
BROWSERS = ("chrome", "safari", "firefox", "edge", "arc", "brave", "opera")
#: Keys a task is about when it is played rather than filled in.
BOARD_KEYS = ("up", "down", "left", "right")
FORM_KEYS = ("tab", "return")


@dataclass(frozen=True)
class WatchedGoal:
    """A goal to keep at, and the thing on screen that means it is finished."""

    goal: str
    success_when: str
    target_app: str = ""
    move_keys: tuple[str, ...] = BOARD_KEYS
    region_top: float = 0.0
    region_bottom: float = 1.0
    unblock_with: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_target(self) -> dict[str, Any]:
        """The payload the pursuit action takes."""
        payload = {
            "goal": self.goal,
            "success_when": self.success_when,
            "move_keys": list(self.move_keys),
            "region_top": self.region_top,
            "region_bottom": self.region_bottom,
        }
        if self.target_app:
            payload["target_app"] = self.target_app
        if self.unblock_with:
            payload["unblock_with"] = self.unblock_with
        return payload


def _continuation(text: str) -> str:
    for pattern in _CONTINUING_RE:
        found = pattern.search(text)
        if found:
            return found.group(0).strip().lower()
    return ""


def _condition_clause(text: str) -> str:
    """The part of the request that says when to stop."""
    lowered = text.lower()
    best = -1
    clause = ""
    for word in UNTIL:
        found = lowered.find(word)
        if found < 0:
            continue
        if found > best:
            best = found
            clause = text[found + len(word) :]
    return clause.strip(" ,.—-")


def _finishing_test(clause: str) -> str:
    """What the screen will say when the goal is reached.

    A number is the strongest signal a screen can be matched on and needs no
    interpretation. Quoted text is the person naming the words themselves.
    Otherwise the last plain word of the clause carries it, which is where the
    thing being waited for sits in English.
    """
    if not clause:
        return ""
    quoted = re.search(r"[\"'“”‘’]([^\"'“”‘’]{1,60})[\"'“”‘’]", clause)
    if quoted:
        return quoted.group(1).strip()
    number = re.search(r"\b(\d[\d,]{0,9})\b", clause)
    if number:
        return number.group(1).replace(",", "")
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", clause)
    skip = {
        "the", "a", "an", "you", "it", "its", "get", "gets", "got", "reach", "reaches",
        "reached", "make", "makes", "made", "see", "sees", "seen", "say", "says", "said",
        "shows", "show", "shown", "appear", "appears", "tile", "and", "then", "that",
        "this", "there", "your", "have", "has", "hit", "hits", "one", "for", "with",
        "something", "anything", "finished", "done",
    }
    plain = [word for word in words if word.lower() not in skip]
    if plain:
        return plain[-1]
    # "until it is done" or "until finished" is a real condition even though
    # every word in it is ordinary.
    for word in reversed(words):
        if word.lower() in {"done", "finished", "complete", "passed", "failed", "ready"}:
            return word
    return ""


def _named_app(text: str) -> str:
    lowered = text.lower()
    for browser in BROWSERS:
        if browser in lowered:
            return {
                "chrome": "Google Chrome",
                "safari": "Safari",
                "firefox": "Firefox",
                "edge": "Microsoft Edge",
                "arc": "Arc",
                "brave": "Brave Browser",
                "opera": "Opera",
            }[browser]
    named = re.search(r"\bin\s+([A-Z][\w.]*(?:\s+[A-Z][\w.]*)?)", text)
    return named.group(1).strip() if named else ""


def _keys_for(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    if any(word in lowered for word in ("wizard", "form", "field", "installer", "setup", "next screen")):
        return FORM_KEYS
    return BOARD_KEYS


def read_watched_goal(objective: str) -> WatchedGoal | None:
    """A watched goal, when the request is one. Otherwise nothing."""
    text = str(objective or "").strip()
    if not text:
        return None
    cue = _continuation(text)
    if not cue:
        return None
    condition = _finishing_test(_condition_clause(text))
    if not condition:
        return None

    app = _named_app(text)
    in_browser = any(browser in app.lower() for browser in BROWSERS) or "://" in text
    return WatchedGoal(
        goal=text[:400],
        success_when=condition,
        target_app=app,
        move_keys=_keys_for(text),
        # Below the tab strip and address bar, so a page title cannot satisfy
        # the condition before a single move is made.
        region_top=CHROME_BAND_TOP if in_browser else 0.0,
        region_bottom=1.0,
        detail={"continuation": cue},
    )
