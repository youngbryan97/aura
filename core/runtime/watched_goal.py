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

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation

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

#: Words that introduce the condition which ends the watching, by saying WHEN.
UNTIL = ("until", "till", "til", "up to", "as soon as", "once you", "when you", "when it", "once it")

#: Words that introduce it by naming what is being AIMED AT.
#:
#: A finish can be said either way and they mean the same thing: "play until
#: you get a 256 tile" and "play it and get to a 256 tile" are one request.
#: LIVE 2026-08-26: the second was not recognised as a goal to keep at, so it
#: never reached the lane that plays anything — it was answered with a web
#: search while the board sat untouched. The capability was there; the
#: phrasing could not reach it.
AIMING_AT = (
    "get to", "gets to", "getting to", "get me to", "reach", "reaches", "reaching",
    "get a", "get an", "get the", "make a", "make an", "make the", "hit a", "hit an",
    "hit the", "score a", "score an", "score the", "up to a", "to a", "to an",
)
#: A browser's own furniture repeats page words in the tab strip and the
#: address bar, so a condition matched anywhere on screen can be satisfied by
#: the title before anything happens. Content starts below the chrome.
CHROME_BAND_TOP = 0.12
BROWSERS = ("chrome", "safari", "firefox", "edge", "arc", "brave", "opera")
#: Keys a task is about when it is played rather than filled in.
BOARD_KEYS = ("up", "down", "left", "right")
FORM_KEYS = ("tab", "return")
#: How long a watched goal runs before it stops itself.
#:
#: One number, read by everything that has to agree about it: the pursuit
#: that bounds itself, the action that wraps it, and the task that waits for
#: the action. When they disagreed the outermost one won, and a run that was
#: playing correctly was cancelled and reported as "Completed 0/0 steps".
PURSUIT_SECONDS = 600.0

#: The most anyone is asked to wait for one watched goal.
PURSUIT_CEILING_S = 3600.0

#: How many cycles a watched goal is allowed. This is the real bound on the
#: work: seconds are only how long that many cycles take.
PURSUIT_CYCLES = 200

#: What one cycle of a pursuit has been measured taking on this machine.
_A_CYCLE: dict[str, float] = {"seconds": 0.0}


#: Where what a cycle costs is kept between runs.
#:
#: Held in a process only, this is forgotten at every restart, so the first
#: watched goal after one always asks for the budget written down rather than
#: the one measured — and the first is usually the one somebody is watching.
_MEASURED_AT = Path.home() / ".aura" / "state" / "pursuit_cycle_seconds.json"


def a_cycle_took(seconds: float) -> None:
    """Record what one cycle of a pursuit actually cost."""
    spent = float(seconds or 0.0)
    if spent <= 0.0:
        return
    before = _remembered()
    _A_CYCLE["seconds"] = spent if before <= 0.0 else before * 0.7 + spent * 0.3
    _write_down(_A_CYCLE["seconds"])


def _remembered() -> float:
    """What a cycle cost, from this process or from the last one."""
    if _A_CYCLE["seconds"] > 0.0:
        return _A_CYCLE["seconds"]
    try:
        held = json.loads(_MEASURED_AT.read_text())
        kept = float(held.get("seconds") or 0.0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0
    if kept > 0.0:
        _A_CYCLE["seconds"] = kept
    return _A_CYCLE["seconds"]


def _write_down(seconds: float) -> None:
    """Keep it where the next run can read it."""
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "watched_goal.cycle_cost",
            domain="state_mutation",
            constraints={"path": str(_MEASURED_AT)},
        ):
            get_file_write_gateway().ensure_directory(
                _MEASURED_AT.parent, source="watched_goal"
            )
            get_file_write_gateway().write_text(
                _MEASURED_AT,
                json.dumps({"seconds": round(float(seconds), 3)}),
                source="watched_goal.cycle_cost",
            )
    except Exception as exc:  # noqa: BLE001 - measuring is not the task
        record_degradation(
            "watched_goal", exc, severity="info", action="kept a cycle cost in memory only"
        )


def seconds_a_cycle() -> float:
    """How long one cycle takes, as measured, or nothing if never measured."""
    return _remembered()


def time_for(cycles: int = PURSUIT_CYCLES) -> float:
    """Long enough to make the moves she is allowed to make.

    The flat number was chosen when a cycle was a keystroke and a glance. A
    cycle now reads the screen, grades the last prediction, and often thinks
    in words, and measured live 2026-08-26 it takes about fourteen seconds —
    so the same budget bought a sixth of the play it was written for, and a
    run that was building a 128 into the corner was stopped at move 43 of a
    game that needs a hundred and fifty.

    What bounds the work is the cycle count. Seconds are only how long that
    many cycles take, and she now knows how long that is.
    """
    measured = seconds_a_cycle()
    if measured <= 0.0:
        return PURSUIT_SECONDS
    return max(PURSUIT_SECONDS, min(PURSUIT_CEILING_S, float(cycles) * measured))


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
    #: Where the task happens: a URL if the request named one, otherwise the
    #: thing itself, which she has to find. Empty when the task is about
    #: whatever is already in front of her.
    where: str = ""
    max_seconds: float = field(default_factory=time_for)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_target(self) -> dict[str, Any]:
        """The payload the pursuit action takes."""
        payload = {
            "goal": self.goal,
            "success_when": self.success_when,
            "move_keys": list(self.move_keys),
            "region_top": self.region_top,
            "region_bottom": self.region_bottom,
            "max_seconds": self.max_seconds,
        }
        if self.target_app:
            payload["target_app"] = self.target_app
        if self.unblock_with:
            payload["unblock_with"] = self.unblock_with
        if self.where:
            payload["open_page"] = self.where
        return payload


def _continuation(text: str) -> str:
    for pattern in _CONTINUING_RE:
        found = pattern.search(text)
        if found:
            return found.group(0).strip().lower()
    return ""


def _condition_clauses(text: str) -> list[str]:
    """Every part of the request that could be saying when to stop, in order.

    More than one, because a request usually carries more than one candidate
    and the last is not always the right one. LIVE 2026-08-26: "Find 2048
    online, play it, and get to a 256 tile. Say what you are about to do
    before each move, and tell me here when you have it." — the rider on the
    end reads as a condition ("when you ... have it") and, taken as the
    latest, it stole the finish from the tile that was actually being played
    for.
    """
    found: list[tuple[int, str]] = []
    for word in (*UNTIL, *AIMING_AT):
        # As WORDS, not as letters that happen to be there.
        #
        # Matched anywhere at all, "til" sits inside "tiles" and "to a" inside
        # "onto a", so a request that merely mentions tiles gets its finishing
        # condition sliced out of the middle of a word. LIVE 2026-08-27: "the
        # classic one with numbered tiles" cut at "numbered ti|les", and what
        # she was left waiting for was the word "are".
        for at in re.finditer(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
            clause = text[at.end() :].strip(" ,.—-")
            if clause:
                found.append((at.start(), clause))
    found.sort(key=lambda pair: pair[0])
    return [clause for _at, clause in found]


def _condition_clause(text: str) -> str:
    """The single part of the request that says when to stop, best first."""
    clauses = _condition_clauses(text)
    return clauses[-1] if clauses else ""


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


def _best_finishing_test(text: str) -> str:
    """The strongest finish the request names.

    A number or a quoted phrase is a thing the screen can be matched against
    and needs no interpretation, so one of those beats a clause that names
    nothing in particular however late it appears. Without this, a closing
    "tell me when you have it" outranked the tile being played for.
    """
    clauses = _condition_clauses(text)
    named: list[str] = []
    for clause in clauses:
        test = _finishing_test(clause)
        if not test:
            continue
        if re.fullmatch(r"\d[\d,]*", test) or f'"{test}"' in text or f"'{test}'" in text:
            return test
        named.append(test)
    if named:
        return named[-1]
    # No clause named it, so the request names it some other way.
    #
    # People say a target in more ways than any list of verbs will hold: "I
    # want to see a 1024 tile", "show me a 512", "I'd like a 2048 out of
    # this". The continuation cue has already established that this is a
    # thing to keep at, and in a thing to keep at, a value named in it is
    # what finishing looks like. Listing the verbs instead would leave the
    # next phrasing outside the capability again.
    #
    # Two values are not targets: a duration says how long rather than what,
    # and the name of the thing being worked in is not a state of it.
    return _value_named_in(text, skipping=_where_it_happens(text))


#: Units that make a number a duration or a count of tries rather than a
#: state the screen will show.
NOT_A_TARGET = (
    "second", "seconds", "sec", "secs", "minute", "minutes", "min", "mins",
    "hour", "hours", "day", "days", "time", "times", "move", "moves", "round",
    "rounds", "turn", "turns", "try", "tries", "attempt", "attempts", "game", "games",
)


def _value_named_in(text: str, *, skipping: str = "") -> str:
    """The value this request is aiming at, when no clause introduced one.

    One value in a request that names a thing is that thing's name: "play
    2048" is not a request to reach 2048, and "keep playing 2048 for 30
    moves" says how long rather than what. It takes a second value for one of
    them to be a target — "keep playing 2048, I want to see a 1024 tile" —
    and the target is the later one, because the thing is named before the
    state it should reach.
    """
    body = str(text or "")
    excluded = {word.replace(",", "") for word in re.findall(r"\d[\d,]*", str(skipping or ""))}
    candidates: list[str] = []
    for found in re.finditer(r"\b(\d[\d,]{0,9})\b", body):
        value = found.group(1).replace(",", "")
        after = body[found.end() : found.end() + 24].strip().lower()
        if any(re.match(rf"{unit}\b", after) for unit in NOT_A_TARGET):
            # How long, or how many tries. Neither is a state of the screen.
            continue
        candidates.append(value)
    distinct = list(dict.fromkeys(candidates))
    remaining = [value for value in distinct if value not in excluded]
    if excluded and remaining:
        return remaining[-1]
    if len(distinct) < 2:
        return ""
    return distinct[-1]


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


def apps_named_in(text: str) -> tuple[str, ...]:
    """Applications a phrase could be naming, likeliest first.

    A phrase is not an application name. "open it in the browser" is a
    sentence, and the application in it is the browser — but handed over whole
    it is looked up as though somebody had installed something called "it in
    the browser". LIVE 2026-08-27: exactly that, and the run stopped one step
    from the thing it was asked to do.

    "The browser" names no product, so every browser is a candidate and
    whichever is really installed answers. Nothing here decides that: this
    reads the words, and the caller finds out what is there.
    """
    said = str(text or "").strip()
    if not said:
        return ()
    named = _named_app(said)
    if named:
        return (named,)
    lowered = said.lower()
    if re.search(r"\bbrowsers?\b", lowered):
        # Whichever of these is actually here. Ordered by how common they are
        # on a Mac, not by preference.
        return ("Safari", "Google Chrome", "Firefox", "Microsoft Edge", "Arc", "Brave Browser")
    # A phrase with a preposition in it is a sentence about an app rather than
    # the name of one, and the name is what comes before the preposition.
    head = re.split(r"\b(?:in|on|at|with|using|from|into)\b", said, maxsplit=1)[0].strip(" .,'\"")
    if head and head.lower() not in _NOT_AN_APP and head.lower() != lowered:
        return (head,)
    return ()


#: Words that name no application, however they are dressed up.
_NOT_AN_APP = frozenset({
    "it", "this", "that", "them", "one", "the app", "the application", "the window",
    "the page", "the site", "the thing",
})


def _keys_for(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    # Word boundaries: "form" sits inside "conformance" and "performance".
    if names_any(lowered, ("wizard", "form", "field", "installer", "setup", "next screen")):
        return FORM_KEYS
    return BOARD_KEYS



#: A URL written into a request. Lives here rather than beside the browser
#: work because reading one out of text is text analysis, and this module is
#: the one both the planner and the runtime's intent check can reach.
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def named_url(text: str) -> str:
    """A URL written into the request, if there is one."""
    found = URL_RE.search(str(text or ""))
    return found.group(0).rstrip(".,);") if found else ""


def _where_it_happens(text: str) -> str:
    """The place a task needs her to be, when the request implies one.

    A URL written into the request is the place. Otherwise it is the thing
    itself — she has to find it, which is a search and then a decision about
    which result is really it.

    Two things are deliberately not done. A domain is never guessed from a
    name, because that is how a request for one site opens another. And a
    request that says the thing is already open names no place at all: "2048
    is open in Chrome" is a statement about the world, not an instruction to
    go somewhere, and reading it as one sends her off to find what is already
    in front of her.
    """
    url = named_url(text)
    if url:
        return url
    for match in _SUBJECT_RE.finditer(text):
        before = str(match.group("before") or "").strip().lower()
        if before in _ALREADY_THERE:
            continue
        # Trimmed before the tail is looked for, because "online" at the end
        # of a name is only at the end once the space after it is gone.
        said = str(match.group("what") or "").strip()
        named = _LOCATION_TAIL.sub("", said).strip(" .,'\"")
        if not named or named.lower() in _NOT_A_PLACE:
            continue
        return named
    return ""


#: Words before the verb that mean it is describing a state rather than
#: asking for an action: "is open", "was already running".
_ALREADY_THERE = frozenset({"is", "are", "was", "were", "already", "still"})
#: Subjects that name nothing findable.
_NOT_A_PLACE = frozenset({"it", "this", "that", "the game", "game", "them", "one", "some", "any"})
#: Trailing words that describe where a thing lives rather than naming it.
_LOCATION_TAIL = re.compile(r"\s+(?:online|on\s+the\s+web|on\s+the\s+internet|in\s+my\s+browser)$", re.IGNORECASE)
#: Verbs that mean she is not there yet and has to get there.
#:
#: Deliberately wider than "open": a person says find, look up, pull up, go
#: to, or just names the thing they want played. The narrow list read "play
#: 2048" and missed "go find a 2048 game online", which is the same request
#: with more of the work spelled out.
_SUBJECT_RE = re.compile(
    r"(?P<before>\b\w+\s+)?"
    r"\b(?:go\s+to|go\s+find|navigate\s+to|head\s+to|find|search\s+for|look\s+up|"
    r"pull\s+up|bring\s+up|open|visit|load|play|use)\s+"
    r"(?:a\s+|an\s+|the\s+|some\s+)?"
    r"(?P<what>[A-Za-z0-9][^.\n,;:()\u2013\u2014]{0,40}?)"
    # Whatever sets an aside off ends the name, and a comma is not the only
    # thing that does. A dash, a semicolon, a colon and a bracket all close a
    # noun phrase in English. LIVE 2026-08-27: "Find a sliding puzzle online —
    # the classic one with numbered tiles" named no place at all, because the
    # aside ran past the forty characters the name is allowed and nothing in
    # it was a comma.
    r"(?=\s+(?:in|on|at|until|till|and|then)\b|[.,;:()\u2013\u2014\n]|\s+-\s|$)",
    re.IGNORECASE,
)


def read_watched_goal(objective: str) -> WatchedGoal | None:
    """A watched goal, when the request is one. Otherwise nothing."""
    text = str(objective or "").strip()
    if not text:
        return None
    cue = _continuation(text)
    if not cue:
        return None
    # A process that never says when to stop is still a process.
    #
    # Requiring a named finish made "find a sliding puzzle and work out how it
    # moves by playing it" structurally unplannable: no number, no quoted
    # phrase, nothing for a screen to be matched against — so it fell through
    # to the one-shot verbs and was answered "Done — opened Safari, and opened
    # Safari." LIVE 2026-08-27, which is the same shape of failure recorded
    # against this module on 2026-08-19 with a different phrasing.
    #
    # Every cue that gets this far is unambiguously about carrying on: keep
    # going, playing, stepping through, monitoring, over and over. None of
    # them can be true after one action. Not naming an end does not make a
    # request one-shot — it means the end is the budget, which is a thing the
    # pursuit already knows how to run to.
    condition = _best_finishing_test(text)

    app = _named_app(text)
    where = _where_it_happens(text)
    # Somewhere to go is itself a browser: naming Chrome is how a person
    # mentions it, not a condition on needing one.
    # Browser names are short words that live inside longer ones: "edge" in
    # "acknowledge", "arc" in "search", "opera" in "cooperative". Matched as
    # substrings, an acknowledgement or a search became evidence that the goal
    # needs a browser.
    in_browser = bool(where) or names_any(app, BROWSERS) or "://" in text
    return WatchedGoal(
        where=where,
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
