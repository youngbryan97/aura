"""Watching a video with somebody as something the two of them are doing, beside the conversation, not as a task.

Bryan, 13 October 2026: watching a video with her is "a way of *being*", not a job; and on 20 October, asked whether
talking to her while it plays would stop it: it must not. A task is finished by producing what was asked for and its
turn holds the conversation until then; two people watching a film talk while it plays and stop when one of them
leaves (core/together/a_with.py). So here:

- **She answers first.** The watching waits for the turn that asked for it to settle, so her reply (what she found,
  what she expects of it) comes before the video starts, and the conversation is free from then on.
- **It is a with.** The one the request opened (core/together/taking_it_up.py), or one begun for it. Each look is the
  with's shared field; each remark of hers a turn of hers in it; the with ends when the video does.
- **She does not talk over anybody.** A remark waits while either of them has the floor, and while a turn of the
  person's is being answered: her ordinary turn answers what was said to her, and a remark does not cut across it.
- **What plays reaches every turn of the conversation while it plays**, as a reading beside the others
  (core/brain/what_she_is_given.py ``OF_THIS_TURN``): what it is, how far in, what was seen and heard lately and what
  she said of it, so "did you see that?" has something to refer to.
- **And the conversation reaches every remark.** Each remark is her turn with what was said since the watching
  began, theirs and hers in the order it was said, as the exchanges any turn of hers reads; a remark she was thinking
  when they started talking is not said over them, and what she made of it at the end waits for the floor.

Her remarks are her own turns (core/skills/watching_along.py ``her_turn``); nothing here writes a word of hers.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.wall_clock import monotonic_time

logger = logging.getLogger("Aura.Together")

__all__ = ["her_answer_before_it_plays", "her_answer_when_nothing_plays", "stop_watching_if_asked", "owed_an_answer", "the_talk_since", "watch_in_a_with", "watching_with", "what_we_are_watching"]

#: How long the watching waits for the turn that asked for it to settle before it starts anyway.
ANSWER_WAITED_S = 180.0
#: How many of the latest looks, and of her latest remarks, the reading of a turn carries.
LATELY_LOOKS, LATELY_SAID = 6, 3

#: What is playing for whom, while it plays.
_PLAYING: dict[str, dict[str, Any]] = {}
_WATCHINGS: set[asyncio.Task] = set()
#: When she last finished watching something with each person (wall clock).
_ENDED_AT: dict[str, float] = {}
#: Who has asked to stop what they are watching with her, and their words.
_STOP_ASKED: dict[str, str] = {}
#: What asks to stop or close what is being watched together: said to her while it plays.
_ASKS_TO_STOP = r"(?i)\b(?:stop (?:it|this|that|the video|watching)|let'?s stop|close (?:it|this|that|the (?:tab|video|window))|turn (?:it|this|that) off|shut (?:it|this|that) off|that'?s enough|enough of (?:it|this)|end (?:it|this) here)\b"


def stop_watching_if_asked(said: str) -> str:
    """When the person watching with her asks to stop it, the watching is stopped: its reason to stop is theirs, and it
    ends as one she stops does, with her word on it and its tab closed. The title stopped, or "". LIVE 7 November 2026,
    asked "Let's stop this one here ... Close it and we'll move on", she answered "Done. I've closed it" while it
    played on: nothing in a watching heard a request to stop."""
    import re

    if not _PLAYING or not re.search(_ASKS_TO_STOP, str(said or "")) or re.search(r"(?i)\bdon'?t stop\b", str(said or "")):
        return ""
    titles = []
    for who, playing in list(_PLAYING.items()):
        _STOP_ASKED[who] = "they asked to stop it"
        titles.append(str(playing.get("title") or "it"))
    _JUST_STOPPED.update({"said": " ".join(str(said or "").split()), "at": time.time()})
    return ", ".join(titles)


_JUST_STOPPED: dict[str, Any] = {}


def stopped_by(said: str, *, within_s: float = 120.0) -> bool:
    """Whether ``said`` is what just stopped a watching: done, so no other lane acts on it as a task of its own. LIVE
    7 November 2026, "Stop this one ... Close it." stopped the watching and closed its tab, and the desktop lane also
    took "close it" as a task, failed it, and answered "This did not complete: unparsable decision"."""
    return (" ".join(str(said or "").split()) == _JUST_STOPPED.get("said")
            and time.time() - float(_JUST_STOPPED.get("at") or 0.0) <= within_s)


#: What she has watched with someone, newest last: title, who made it, with whom, when it ended, and her last word on it.
_WATCHED_TOGETHER: list[dict[str, Any]] = []
#: How many watchings are kept, and for how long after one ends it is part of what every turn carries, in seconds.
WATCHED_TOGETHER_KEPT, WATCHED_TOGETHER_FOR_S = 12, 3 * 3600.0


#: How many exchanges since the request a turn of hers reads while watching, besides the request and her answer.
RECENT_EXCHANGES = 8
#: How long before a watching began its request may have been made: her answer can take minutes.
REQUEST_BEFORE_S = 900.0

#: How many remarks asking her to think out loud counts for, as remarks of hers taken up: a welcome of 0.8 to begin.
INVITED_REMARKS = 3
#: Who has asked her to think out loud while something plays.
_ASKED_TO_THINK_OUT_LOUD: set[str] = set()

#: What she has of a video as it plays beside her, as the watching gives it (core/skills/watching_the_screen.py).
AS_IT_PLAYS = ("As it plays on their screen beside her window, she sees it every few seconds, hears its sound, and "
               "speaks when something in it is worth saying.")


async def her_answer_before_it_plays(request: str, found: str) -> str:
    """Her reply to the request to watch, before the video starts: her own turn, with what her browsing found of it
    (its title, its length, what its page says of itself) as fresh evidence, as each remark of hers is made
    (core/skills/watching_along.py ``her_turn``). LIVE 20 October 2026 the browsing's receipt went out in its place
    ("what the page plays (Found A short film about colour (0:06)...) That much is verified."), and an empty reply
    sent the request back through the browsing again. What was found stands in only if her turn gives nothing."""
    from core.skills.watching_along import her_turn

    try:
        # What she will have of it as it plays is part of what is so before it plays: LIVE 20 October 2026, without it
        # her answer said "I can't reliably stream my thoughts second-by-second on my own".
        said = await her_turn(request, f"before it plays. What was found of it: {found}. {AS_IT_PLAYS}", [])
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError, TimeoutError) as exc:
        record_degradation("watching_together", exc, severity="info", action="what was found stands as the reply")
        said = ""
    return said.strip() or found


async def her_answer_when_nothing_plays(request: str, found: str) -> str:
    """Her reply to a request to find something and watch it, when her browsing ended with nothing playing with them:
    her own turn, with what it found as evidence. LIVE 7 November 2026 the results page's text went back as her reply
    ("Jeff Allen shares humorous insights into 37 years of marriage ... 802K 6y ago Dry Bar Comedy")."""
    from core.skills.watching_along import her_turn

    try:
        said = await her_turn(request, f"what her browsing found, with nothing opened or playing yet: {found[:1500]}", [])
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError, TimeoutError) as exc:
        record_degradation("watching_together", exc, severity="info", action="what was found stands as the reply")
        said = ""
    return said.strip() or found


def _made_out_words(heard: Any) -> bool:
    """Whether her ear made out words in it: three or more of two letters or more."""
    import re

    return len(re.findall(r"[A-Za-z]{2,}", str(heard or ""))) >= 3


#: How many looks' caption lines are kept to tell a new line from one still on screen.
CAPTION_LINES_KEPT = 12


def said_on_screen(seen: Any, shown_before: list[str]) -> str:
    """The caption lines new on screen since the last looks: what is said there, without what is not speech
    ("[Music]", "[Applause]") and without what stays on screen (a channel's mark, a lower third), which a new line
    is not. ``shown_before`` is kept up to date."""
    import re

    found = re.search(r'words on it: "([^"]*)"', str(seen or ""))
    lines = [" ".join(line.split()) for line in (found.group(1).split(" / ") if found else [])]
    lines = [re.sub(r"\[[^\]]*\]", "", line).strip() for line in lines]
    new = [line for line in lines if len(re.findall(r"[A-Za-z]{2,}", line)) >= 2 and line not in shown_before]
    shown_before.extend(line for line in lines if line)
    del shown_before[:-CAPTION_LINES_KEPT]
    return " ".join(new)


def watching_with(*, within_s: float = 0.0, now: float | None = None) -> str:
    """Who she is watching something with now, or finished watching with less than ``within_s`` ago; "" if nobody.
    Somebody watching with her is there though they type nothing, so time alone since the watching counts from its
    end, as it does from a person's last word."""
    if _PLAYING:
        return next(iter(_PLAYING))
    now = time.time() if now is None else now
    recent = [(at, who) for who, at in _ENDED_AT.items() if now - at < within_s]
    return max(recent)[1] if recent else ""


def owed_an_answer() -> bool:
    """Whether a turn of the person's is being answered now: her orchestrator serving a turn that is not its own."""
    try:
        from core.runtime.service_access import resolve_orchestrator

        orchestrator = resolve_orchestrator()
        status = getattr(orchestrator, "status", None)
        return bool(getattr(status, "is_processing", False)) and not bool(
            getattr(orchestrator, "_current_task_is_autonomous", False))
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("whether a turn is being answered could not be read: %s", exc)
        return False


#: How far back what she has taken in of a video counts for whether something in it is new to her, in seconds of it.
NEW_TO_HER_S = 120.0
#: What something new is worth to her when her record says nothing of it: what a chance is before anything has been
#: seen of it (core/together/when_to_say_something.py reads an unknown as one half).
NEW_AND_UNKNOWN = 0.5


class WorthSaying:
    """Whether what has played since her last remark is worth one now: her own rule for speaking unasked inside a with
    (core/together/when_to_say_something.py), how much it is worth to her against what taking his attention from what
    plays costs.

    Its worth is what her record says of what it names (her strongest like or dislike among them, her stake in what it
    is about: core/together/remarks.py interest_in), or, where it says nothing, that enough has come since she last
    spoke that is new to her: words (by stem) she has not seen or heard in the last ``NEW_TO_HER_S`` of it, as many as
    the middle of her remarks so far came after (her first, at the opening where every word is new, not counted), worth
    ``NEW_AND_UNKNOWN``.

    LIVE 22 October 2026 its worth was how unlike each look's words were from the last look's. Her eyes word the same
    garage differently every five seconds, so it read 0.88 at the middle, and every one of the 125 readings of October's
    watchings said to speak: she spoke whenever fifteen seconds had passed and her last remark had been heard, 18 times
    in 11 minutes, about every 35 seconds. Replayed over the same looks this speaks 10 times, 24 to 114 seconds apart,
    sooner where more is coming. Before that (LIVE 20 October 2026) new words were the whole of the rule, and a sponsor's
    read is all new words: so newness is worth only what an unknown is, and her record's word on a thing outweighs it.

    What she took in since she last spoke, once it has been read against her own expectations by sight, sound and
    words (core/together/what_moves_her.py), says more than how new it is: how far it moved her, all channels together
    and habituated, is what it is worth when it reaches ``MOVED_AT``, and below that it is worth nothing. Newness stands
    in only where nothing since has been read enough to tell (the first looks, or her senses not reached).

    The cost side reads how much of each look is new to her against how much has been (a.eventful), so a lull is a
    stretch where little is new, not one her eyes happened to word like the last. Worth is the most since her last
    remark, so a remark at a lull can be about what came before it. What plays will not be there to point at later, so
    all of a remark's worth is lost by waiting (the rule's m)."""

    def __init__(self, held: Any, moves: Any = None) -> None:
        self.struck = ""
        self.held, self.interest, self.thing = held, 0.0, ""
        self.moves = moves
        self.recent: list[tuple[float, set[str]]] = []
        self.fresh, self.at_remarks = 0, []
        #: Whether something new to her that her record says nothing of has come since she last spoke.
        self.unknown_and_new = False

    def took_in(self, moment: Any) -> float | None:
        """Take in a look; how much of it is new to her, in [0, 1], or None where it has no words."""
        from core.skills.watching_along import _stems
        from core.together.remarks import interest_in

        seen = " ".join(part for part in (moment.seen, moment.read) if part)
        at = float(getattr(moment, "at", 0.0) or 0.0)
        came = _stems(seen)
        self.recent = [(when, stems) for when, stems in self.recent if at - when <= NEW_TO_HER_S]
        known = set().union(*(stems for _when, stems in self.recent))
        new = came - known
        self.recent.append((at, came))
        self.fresh += len(new)
        interest, thing = interest_in(seen, None)  # what her record says of it, and nothing for how new it is
        if interest > 0.0 and interest >= self.interest:
            self.interest, self.thing = interest, thing or seen[:80]
        elif interest <= 0.0 and new:
            # Her record says nothing of it: its worth is unknown, and new to her. A thing her record has a word on,
            # however small (a sponsor's read), is worth that word, new or not.
            self.unknown_and_new = True
            self.thing = self.thing or seen[:80]
        return len(new) / len(came) if came else None

    def enough_is_new(self) -> bool:
        """Whether as much has come that is new to her since she last spoke as was new when she did."""
        middle = sorted(self.at_remarks[1:])
        return self.fresh >= max(1, middle[len(middle) // 2] if middle else 1)

    def remarked(self) -> None:
        self.at_remarks.append(self.fresh)
        self.fresh, self.interest, self.thing, self.unknown_and_new = 0, 0.0, "", False
        if self.moves is not None:
            self.moves.remarked()

    def __call__(self, _moment: Any, now: float) -> bool:
        from core.together.what_moves_her import MOVED_AT
        from core.together.when_to_say_something import worth_saying_now

        # What they said moved her too, and she answers them for it: it is not also a remark's worth.
        moved = self.moves.moved(without=("together",)) if self.moves is not None else None
        thing = self.thing
        if moved is not None:
            stirred, words = moved
            from_her = stirred if stirred >= MOVED_AT else 0.0
            thing = words[:80] if from_her > self.interest else thing
            how = f"moved {stirred:.2f}"
        else:
            from_her = NEW_AND_UNKNOWN if self.unknown_and_new and self.enough_is_new() else 0.0
            how = f"{self.fresh} new since she spoke"
        # And what she has taken in and not shared: she speaks because she has something of it to share, which what
        # she makes out of it fills, and what moves her fills sooner (core/together/what_moves_her.py ``unshared``).
        if self.moves is not None and callable(getattr(self.moves, "unshared", None)):
            from core.together.what_moves_her import SHARE_AT, SHARE_WORTH

            taken, latest = self.moves.unshared()
            if taken >= SHARE_AT:
                sharing = min(1.0, SHARE_WORTH * taken / SHARE_AT)
                if sharing > max(self.interest, from_her):
                    thing, how = (latest[:80] or thing), f"{how}; {taken:.2f} taken in and not shared"
                from_her = max(from_her, sharing)
        worth = max(self.interest, from_her)
        say = worth_saying_now(self.held, now, interest=worth, thing=thing, lost_by_waiting=1.0)
        # What struck her, hers to know when she speaks (core/together/what_moves_her.py ``what_struck_her``).
        self.struck = self.moves.what_struck_her() if say.speak and self.moves is not None and callable(
            getattr(self.moves, "what_struck_her", None)) else ""
        logger.info("🤝 %s: %s (worth %.2f: her record %.2f in %s, %s)",
                    "worth saying now" if say.speak else "stays quiet", say.why, worth, self.interest,
                    thing or "what is on", how)
        return say.speak


#: Her feelings that move away from what is in front of her (core/affect/feelings_about.py WITHDRAW).
WITHDRAWING = ("fear", "disgust", "dread")
#: How far into the room she had left above her withdrawing feelings when it began they must stay, and for how many
#: looks running, for her to stop watching. A first limit, measured from where she started rather than set at a level
#: of ours: her own sweet spot, learned from what she has watched (WHAT_MOVES_HER.md, block 3), is to replace it.
LEAVES_AT_SHARE, LEAVES_AFTER_LOOKS = 0.5, 3


async def how_she_feels() -> dict[str, float]:
    """How she feels now, read from her state and written by nothing here: her valence and arousal, and the strongest
    of her withdrawing feelings. {} when her state cannot be read."""
    try:
        from core.container import ServiceContainer

        repo = ServiceContainer.get("state_repository", default=None)
        state = await repo.get_state() if repo is not None else None
        affect = getattr(state, "affect", None)
        emotions = dict(getattr(affect, "emotions", {}) or {})
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("her feelings could not be read: %s", exc)
        return {}
    if affect is None:
        return {}
    felt = {"valence": float(getattr(affect, "valence", 0.0) or 0.0), "arousal": float(getattr(affect, "arousal", 0.0) or 0.0)}
    withdrawing = [float(emotions[name] or 0.0) for name in WITHDRAWING if name in emotions]
    if withdrawing:
        felt["withdraw"] = max(withdrawing)
    return felt


async def _withdrawing() -> float | None:
    """The strongest of her withdrawing feelings now."""
    return (await how_she_feels()).get("withdraw")


#: How many watchings she must have come out of, with how she felt kept, before her own limit and sweet spot are read
#: from them instead of the first limit.
FEWEST_WATCHINGS = 5
#: The highest a limit of hers is ever read as: past it nothing is something she can be said to take.
HIGHEST_LIMIT = 0.95


def her_sweet_spot(folder: Any = None) -> dict[str, float]:
    """What her own watchings say she can take and what she enjoys, from how she felt before and after each (the
    "felt" rows of data/watched/*.jsonl): her limit, the most withdrawn she was in a watching she came out of no worse
    (valence no lower than it began); and her sweet spot, how much a watching moved her where her valence rose most on
    average (thirds of how far it moved her). Enjoyment of fear rises with it up to a point and then falls (Andersen et
    al. 2020; Berlyne's inverted U), and where the point lies is hers. {} before ``FEWEST_WATCHINGS``."""
    import json
    from pathlib import Path

    if folder is None:
        from core.config import config

        folder = Path(config.paths.data_dir) / "watched"
    watched: list[tuple[float, float, float]] = []
    for path in sorted(Path(folder).glob("*.jsonl"))[-200:]:
        try:
            for line in path.read_text(errors="ignore").splitlines():
                if '"felt"' not in line:
                    continue
                row = json.loads(line)
                before, after = dict(row.get("before") or {}), dict(row.get("after") or {})
                if row.get("kind") == "felt" and "valence" in before and "valence" in after:
                    watched.append((float(row.get("moved most") or 0.0), float(after["valence"]) - float(before["valence"]),
                                    float(row.get("highest withdraw") or 0.0)))
        except (OSError, ValueError, TypeError) as exc:
            logger.debug("a watching's feelings could not be read: %s", exc)
    if len(watched) < FEWEST_WATCHINGS:
        return {}
    found: dict[str, float] = {"watchings": float(len(watched))}
    fine = [withdraw for _moved, change, withdraw in watched if change >= 0.0]
    if fine:
        found["limit"] = min(HIGHEST_LIMIT, max(fine))
    thirds: dict[int, list[float]] = {}
    for moved, change, _withdraw in watched:
        thirds.setdefault(min(2, int(max(0.0, min(1.0, moved)) * 3)), []).append(change)
    best = max(thirds, key=lambda third: sum(thirds[third]) / len(thirds[third]))
    found["sweet spot"] = (best + 0.5) / 3.0
    return found


class WhatSheCanTake:
    """Whether she has had enough of what plays: her withdrawing feelings (fear, disgust, dread) staying above her
    limit for ``LEAVES_AFTER_LOOKS`` looks running. Watching something that frightens or disgusts her is hers to stop,
    as it would be anyone's (WHAT_MOVES_HER.md, Welfare)."""

    def __init__(self, read: Callable[[], Awaitable[float | None]] | None = None, *, limit: float | None = None) -> None:
        self.read = read or _withdrawing
        self.began: float | None = None
        self.over = 0
        #: Her own limit, read from the watchings she came out of (``her_sweet_spot``); None until there are enough.
        self.limit = limit
        #: The most withdrawn she has been in this watching, to be kept with how she came out of it.
        self.highest = 0.0

    async def __call__(self) -> str:
        now = await self.read()
        if now is None:
            return ""
        self.highest = max(self.highest, now)
        if self.began is None:
            self.began = now
            return ""
        mine = self.limit is not None
        limit = max(self.began, self.limit) if mine else self.began + LEAVES_AT_SHARE * (1.0 - self.began)
        self.over = self.over + 1 if now >= limit else 0
        if self.over < LEAVES_AFTER_LOOKS:
            return ""
        past = ("the most she has sat through and come out of no worse" if mine
                else f"half the room she had above {self.began:.2f} when it began")
        return f"her fear, disgust or dread stayed at {now:.2f} for {self.over} looks running, past {limit:.2f}, {past}"


#: How long a check of her feeling directions stands before it is made again, in seconds: a week.
FEELINGS_CHECK_STANDS_S = 7 * 24 * 3600.0
#: How long a check that found no direction reading liking stands: an hour, so it is soon made again.
NO_FEELING_FOUND_STANDS_S = 3600.0


def _feelings_check_file() -> Any:
    from pathlib import Path

    from core.config import config

    return Path(config.paths.data_dir) / "watching" / "what_her_feelings_read.json"


async def her_feelings_checked(ask: Callable[..., Awaitable[dict[str, Any]]] | None) -> dict[str, Any]:
    """Her feeling directions' check (core/together/what_moves_her.py ``what_her_feelings_read``), made once a week and
    kept: which direction reads liking, and where liking and disliking lie on it. {} when it has not passed."""
    import json

    from core.together.what_moves_her import what_her_feelings_read

    path = _feelings_check_file()
    try:
        kept = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as exc:
        logger.debug("the check of her feelings could not be read: %s", exc)
        kept = {}
    stands = FEELINGS_CHECK_STANDS_S if kept.get("found") else NO_FEELING_FOUND_STANDS_S
    if kept and time.time() - float(kept.get("checked_at") or 0.0) < stands:
        return dict(kept.get("found") or {})
    if ask is None:
        return {}
    found = await what_her_feelings_read(ask)
    if found is None:
        return {}  # not checked: made again next time, not kept
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked_at": time.time(), "found": found}, indent=1))
    except OSError as exc:
        logger.debug("the check of her feelings could not be kept: %s", exc)
    logger.info("🎬 her feeling directions checked: %s", found or "none reads liking")
    return found


def _in_her_neural_stream(line: str) -> None:
    """What moved her while she watches, shown in her neural stream as it happens."""
    try:
        from core.thought_stream import get_emitter

        get_emitter().emit("Watching 🎬", line, level="info", category="Perception")
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("what moved her was not shown in her stream: %s", exc)


def the_floor_is_free(held: Any) -> Callable[[float], bool]:
    """Whether the with leaves her the floor at a moment: nobody speaking, and nothing said to her still owed."""
    from core.together.a_with import HER, THEM

    def free(now: float) -> bool:
        who, owed = held.speaking(now), owed_an_answer()
        if who in (HER, THEM) or owed or not held.open:
            # Said each time, so a watching that stays quiet says why (LIVE 20 October 2026: nothing for 76 seconds).
            logger.info("🤝 a remark waits: %s", "the with is over" if not held.open else
                        f"{who} has the floor" if who in (HER, THEM) else "a turn of theirs is being answered")
            return False
        return True

    return free


_LATELY_READ: dict[str, Any] = {}


def _what_we_watched_lately(now: float) -> list[dict[str, Any]]:
    """What she watched with someone lately, from her record of each watching (data/watched), so it outlasts a restart:
    its title, who made it, when it ended, and her last word on it; and what has ended in this run since it was read.
    Read again at most every half minute."""
    import json
    from pathlib import Path

    kept = _LATELY_READ.get("watched")
    if kept is None or now - float(_LATELY_READ.get("at", 0.0)) > 30.0:
        kept = []
        try:
            from core.config import config

            folder = Path(config.paths.data_dir) / "watched"
            paths = [p for p in sorted(folder.glob("*.jsonl")) if now - p.stat().st_mtime <= WATCHED_TOGETHER_FOR_S]
        except (ImportError, AttributeError, OSError) as exc:
            logger.debug("what was watched lately could not be listed: %s", exc)
            paths = []
        for path in paths[-WATCHED_TOGETHER_KEPT:]:
            try:
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                title = next((r.get("title") for r in rows if r.get("title")), "")
                said = [r.get("words") for r in rows if r.get("kind") == "said" and r.get("words")]
                as_it_played = [(float(r.get("at") or 0.0), str(r.get("words")))
                                for r in rows if r.get("kind") == "said" and r.get("when") == "during" and r.get("words")]
                walls = [float(r.get("wall") or 0.0) for r in rows if r.get("wall")]
                ended = max(walls, default=path.stat().st_mtime) or path.stat().st_mtime
                began = min(walls, default=ended)
                by = next((r.get("by") for r in rows if r.get("kind") == "remembered" and r.get("by")), "")
                if title:
                    kept.append({"title": title, "by": by, "began": began, "ended": ended,
                                 "her last word": said[-1] if said else "", "as it played": as_it_played})
            except (OSError, ValueError) as exc:
                logger.debug("a watching could not be read: %s", exc)
        _LATELY_READ.update({"at": now, "watched": kept})
    titles = {w["title"] for w in kept}
    every = [*kept, *(w for w in _WATCHED_TOGETHER if w["title"] not in titles)]
    # The same thing watched again is one thing watched, as of its latest watching.
    latest = {w["title"]: w for w in sorted(every, key=lambda w: float(w.get("ended") or 0.0))}
    return sorted(latest.values(), key=lambda w: float(w.get("ended") or 0.0))


#: The longest break between two watchings that are one sitting, in seconds: past it, what was watched before is
#: another sitting's, as an evening's films are not the morning's.
A_SITTING_BREAKS_AFTER_S = 45 * 60.0


def this_sitting(watched: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
    """What they watched in this sitting: counting back from now, each watching that ended within
    ``A_SITTING_BREAKS_AFTER_S`` of the next (or of now), and none before the first break. LIVE 7 November 2026, asked
    "what did you think of all of them" after a long day of rehearsals, she was handed a dozen watchings of three hours
    and said she did not have the list in front of her."""
    ordered = sorted(watched, key=lambda w: float(w.get("ended") or 0.0))
    kept: list[dict[str, Any]] = []
    later = now
    for watching in reversed(ordered):
        ended = float(watching.get("ended") or 0.0)
        if later - ended > A_SITTING_BREAKS_AFTER_S or now - ended > WATCHED_TOGETHER_FOR_S:
            break
        kept.append(watching)
        later = float(watching.get("began") or ended)
    return list(reversed(kept))


#: How much of what she said of each watching of the sitting she is handed: her remarks as it played, spread across it,
#: each cut to its head, and her last word on it. LIVE 7 November 2026, dress rehearsal: handed 240 characters of her
#: last word on each, asked "So what did you think of all three?", she said of Allen Pan's snake legs that "the sound
#: design hit before I understood why" and it was "physical flinching": what she said of another of his videos, two
#: hours before, from her recall.
REMARKS_KEPT, REMARK_CHARS, LAST_WORD_CHARS = 6, 200, 480


def _spread(items: list[Any], most: int) -> list[Any]:
    """At most ``most`` of ``items``, spread evenly across them, first and last kept."""
    if len(items) <= most:
        return items
    if most <= 1:
        return items[-1:]
    step = (len(items) - 1) / (most - 1)
    return [items[round(i * step)] for i in range(most)]


def what_we_are_watching(_said: str = "", *, now: float | None = None) -> str:
    """The reading every turn of the conversation carries while something plays: what, how far in, lately; and what
    she has watched together with someone in the last hours, in the order they watched it, with her last word on each:
    what a person has in mind of what they have just been watching with someone."""
    now = time.time() if now is None else now
    recent = this_sitting(_what_we_watched_lately(now), now)
    if not _PLAYING and not recent:
        return ""
    from core.skills.watching_along import clock, what_she_watched

    blocks = []
    if recent:
        lines = ["## WHAT WE HAVE WATCHED TOGETHER LATELY"]
        playing = {str(p.get("title") or "") for p in _PLAYING.values()}
        for number, watched in enumerate(recent, 1):
            ago = max(1, round((now - float(watched["ended"])) / 60))
            by = f" by {watched['by']}" if watched.get("by") else ""
            last = " ".join(str(watched.get("her last word") or "").split())[:LAST_WORD_CHARS]
            line = f"{number}. {watched['title']}{by}, ended {ago} min ago."
            # What she thought as it played, in her words then: what a person draws on when asked what they made of
            # it. Not for what plays now, whose remarks are in the talk.
            remarks = [] if watched["title"] in playing else _spread(list(watched.get("as it played") or []), REMARKS_KEPT)
            if remarks:
                line += " As it played she said: " + " / ".join(
                    f"({clock(at)}) {' '.join(words.split())[:REMARK_CHARS]}" for at, words in remarks)
            lines.append(line + (f" Her last word on it: {last}" if last else ""))
        blocks.append("\n".join(lines))
    for playing in _PLAYING.values():
        lines = [f"## WHAT IS PLAYING WHILE WE TALK\n{playing['title']} ({playing['url']}), "
                 f"at {clock(playing['at'])} of {clock(playing['duration'])}."]
        lately = what_she_watched(playing["moments"][-LATELY_LOOKS:])
        if lately:
            lines.append(f"Lately:\n{lately}")
        said = [words for _at, words in playing["said"][-LATELY_SAID:]]
        if said:
            lines.append("Said of it so far: " + " / ".join(said))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def the_talk_since(began: float, request: str, spoken: list[tuple[float, str, str]]) -> list[dict[str, Any]]:
    """What was said since the watching began, as the exchanges a turn of hers reads: the request and her answer, each
    remark of hers after what she was shown of the video before it, and the person's turns and her answers to them,
    in the order they came. ``spoken`` is her remarks: when, what she was shown, what she said.

    LIVE 21 October 2026, rehearsal 11: with every remark folded into one reply to the request, the request was the
    last thing anyone had said to her, and three minutes in a remark began "Let me think out loud before it starts".
    What plays comes as its own turn and each remark answers it, the shape a commentary on a stream has."""
    from core.together.remarks import said_since
    from core.utils.injected_blocks import stamp_runtime_payload

    remarks = {words for _at, _shown, words in spoken}
    pairs: list[tuple[str, list[str]]] = [(request, [])]
    # From the request on, not from when the watching began: her answer to it can be kept before the player has opened.
    # LIVE 7 November 2026, rehearsal 1: her answer was kept at 03:57:44 and the watching began after it, so no remark
    # saw that she had answered, and the first one answered the request again ("I'm going to go out on a limb and say
    # this is Rick Astley").
    talk = said_since(began - REQUEST_BEFORE_S)
    asked_at = max((at for at, role, words in talk if role == "user" and words == request), default=None)
    talk = [turn for turn in talk if turn[0] >= (asked_at if asked_at is not None else began)]
    events = [*((at, role, words) for at, role, words in talk if not (role == "assistant" and words in remarks)),
              *((at, "remark", (shown, words)) for at, shown, words in spoken)]
    for _at, role, words in sorted(events, key=lambda e: e[0]):
        if role == "user" and words == request and len(pairs) == 1:
            continue  # the request heads it already
        if role == "user":
            pairs.append((words, []))
        elif role == "remark" and len(pairs) > 1 and not pairs[-1][1]:
            pairs[-1][1].append(words[1])  # said after a turn of theirs nothing had answered: it answers that
        elif role == "remark":
            # Hers, not something sent to her: LIVE 21 October 2026, "(what was playing) ..." read as the person pasting
            # the page, and a remark began "I can't actually watch or listen to the video".
            pairs.append((f"[Aura's own eyes and ears on the video they are watching together] {words[0]}", [words[1]]))
        elif words not in pairs[-1][1]:
            pairs[-1][1].append(words)
    said = [(user, aura) for user, aura in pairs if aura]
    # The request and her answer, and the newest of the rest: a whole video's remarks made each turn's prompt long
    # enough to stall her (LIVE 21 October 2026, 89,827 characters).
    said = said[:1] + said[1:][-RECENT_EXCHANGES:] if said and said[0][0] == request else said[-RECENT_EXCHANGES:]
    return [stamp_runtime_payload({"user": user, "aura": " ".join(aura)}) for user, aura in said]


def _read_in_every_turn() -> None:
    from core.brain.what_she_is_given import OF_THIS_TURN

    if what_we_are_watching not in OF_THIS_TURN:
        OF_THIS_TURN.append(what_we_are_watching)


async def _her_answer_settled(turn: Any, *, at_most_s: float) -> None:
    """Until the turn that asked for the watching has settled what it serves, or ``at_most_s`` has passed."""
    deadline = monotonic_time() + at_most_s
    while turn is not None and not getattr(turn, "finalized", True) and monotonic_time() < deadline:  # noqa: ASYNC110 - a turn has no event to wait on; its flag is all it says
        await asyncio.sleep(0.5)


async def watch_in_a_with(open_player: Callable[[], Awaitable[Any]], request: str, *, title: str, url: str,
                          at_most_s: float | None = None, turn: Any = None, who: str = "", by: str = "",
                          **pace: float) -> None:
    """Watch in the with the request opened (or one begun for it), once her answer to it has gone out."""
    from core.skills.watching_along import (
        her_turn,
        into_her_memory,
        kept_at,
        watch_along,
        what_it_calls_up,
        what_she_remembers_of,
        what_she_remembers_of_them,
    )
    from core.together.a_with import IT, THEM
    from core.together.taking_it_up import install, who_she_is_with

    taking = install()
    who = who or who_she_is_with(taking._her_record())
    withs = taking._withs()
    held = withs.current(who) or withs.begin(who, f"watching {title}", proposed_by=THEM)
    # This watching's talk begins with it: the with may be older (LIVE 21 October 2026, one open all night carried
    # every earlier watching into each remark).
    watching_began = time.time() - 5.0
    from core.intent.asks_for_remarks import asks_for_remarks

    # Asked once to think out loud as it plays, asked for the next one too: "Now Salad Fingers. Same again." What
    # follows still counts, as remarks of hers taken up or let go by.
    if asks_for_remarks(request):
        _ASKED_TO_THINK_OUT_LOUD.add(who)
    held.invited = max(held.invited, INVITED_REMARKS if who in _ASKED_TO_THINK_OUT_LOUD else 0)
    await _her_answer_settled(turn, at_most_s=ANSWER_WAITED_S)
    # Before it opens, so nothing plays unwatched: whether her feeling directions read liking, checked once a week.
    from core.together.what_moves_her import her_expectations as _her_expectations

    feelings = await her_feelings_checked(_her_expectations())
    # Who is in it is read by a small model of its own: loaded beside the player opening, so no reading waits on it.
    from core.perception import who_is_in_it

    warming = asyncio.ensure_future(asyncio.to_thread(who_is_in_it._reader))
    player = await open_player()
    if player is None:
        logger.info("🤝 %s could not be opened where %s watches; nothing to watch together.", title, who)
        return
    playing = _PLAYING[who] = {"title": title, "url": url, "at": 0.0, "duration": 0.0, "moments": [], "said": []}
    spoken: list[tuple[float, str, str]] = []
    _read_in_every_turn()

    from core.perception.what_her_senses_expect import SoundExpectation
    from core.perception.who_is_in_it import people_in
    from core.together.what_moves_her import WhatMovesHer, her_expectations, her_self_text, what_videos_are_usually_like

    keep = kept_at(title, url)
    # What moves her in it, read off her own expectations of what she sees, hears and is told
    # (core/together/what_moves_her.py), and what her ear expects of each second of it.
    remembered = await what_she_remembers_of(title, url)
    # Who made it, and what she remembers of what they made before: part of what she expects of this one.
    of_them = await what_she_remembers_of_them(by, title) if by else ""
    if remembered or of_them:
        keep({"kind": "remembered", "events": len(remembered), "of them": bool(of_them), "by": by, "wall": time.time()})
    moves = WhatMovesHer(f"{title}, by {by}" if by else title, ask=her_expectations(), keep=keep,
                         self_text=her_self_text(), remembered=remembered, feelings=feelings, show=_in_her_neural_stream,
                         of_them=of_them, whom=people_in, recall=what_it_calls_up,
                         usual=await asyncio.to_thread(what_videos_are_usually_like))
    ears = SoundExpectation()
    worth = WorthSaying(held, moves=moves)

    shown_before: list[str] = []

    def took_in(moment: Any, plays: dict[str, Any]) -> None:
        playing.update(at=float(plays.get("at") or moment.at), duration=float(plays.get("duration") or 0.0))
        playing["moments"].append(moment)
        shown = " ".join(part for part in (moment.seen, moment.read) if part)[:400] or title
        # How much of it is new to her, not how unlike its words are from the last look's (WorthSaying).
        held.shown(shown, time.time(), change=worth.took_in(moment))
        at = float(moment.at or 0.0)
        moves.saw(at, getattr(moment, "sight", None), moment.seen)
        seconds = [row for row in (getattr(moment, "sounds", None) or []) if isinstance(row.get("loudness"), (int, float))]
        for back, row in enumerate(reversed(seconds)):
            _surprise, onset = ears.heard(float(row["loudness"]))
            moves.sounded(max(0.0, at - back), ears.above_usual, onset)
        # What is said, as she takes it in: what her ear wrote down, or where it made out next to nothing (sung words
        # over music, a film in another language), what the captions or subtitles say. LIVE 7 November 2026, a
        # concert: her ear heard "." while the captions read "just cuz you feel it / doesn't mean it's there", and her
        # remarks were of the camera and the light.
        captions = said_on_screen(moment.seen, shown_before)
        moves.heard(at, moment.read if _made_out_words(moment.read) else (captions or moment.read))
        # What they said since the last look, as the with heard it.
        for turn in list(held.turns):
            if turn.by == THEM and turn.at > their_last["at"]:
                their_last["at"] = turn.at
                moves.they_spoke(at)

    their_last = {"at": time.time()}

    def what_she_took_it_in_as() -> tuple[Any, ...]:
        return (moves.events_text(), max((felt.evidence for felt in moves.felt), default=0.0), moves.about_her(),
                moves.events_kept(), moves.people_kept(), list(moves.reminded_of))

    def into_her_memory_with(watched: Any) -> Any:
        watched.with_whom, watched.by = who, by
        return into_her_memory(watched, url=url)

    shown_last = {"observation": ""}

    def she_said(words: str) -> None:
        worth.remarked()
        playing["said"].append((playing["at"], words))
        spoken.append((time.time(), shown_last["observation"][-500:], words))  # its newest looks: a watching stays short to read
        taking.she_said(words, who=who, unprompted=True)

    async def her_turn_in_it(asked: str, observation: str, said: list[str]) -> str:
        shown_last["observation"] = observation
        talk = await asyncio.to_thread(the_talk_since, watching_began, asked, list(spoken))
        # Every turn of hers in a with after her answer is a remark: what was asked is already in her conversation.
        return await her_turn(asked, observation, said, exchanges=talk, remark=True)

    # What she can take is her own once she has come out of enough watchings to say (her_sweet_spot).
    sweet = await asyncio.to_thread(her_sweet_spot)
    can_take = WhatSheCanTake(limit=sweet.get("limit"))
    felt_before = await how_she_feels()
    if sweet:
        keep({"kind": "her sweet spot", **sweet, "wall": time.time()})
    # Under way only inside what ends it: her appraisal reads a watching while it is registered (moves.done ends it).
    moves.begin()
    try:
        async def stop_when() -> str:
            return _STOP_ASKED.pop(who, "") or await can_take()

        await watch_along(player, request, turn=her_turn_in_it, before=False, may_speak=the_floor_is_free(held),
                          worth=worth, on_moment=took_in,
                          on_said=she_said, keep=keep, at_most_s=at_most_s,
                          remember=into_her_memory_with,
                          look_in=moves.look_in, stop_when=stop_when, whole=what_she_took_it_in_as, **pace)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("together", exc, severity="warning", action="the watching together stopped")
    finally:
        if not warming.done():
            warming.cancel()
        moves.done()
        ended = _PLAYING.pop(who, None)
        if ended is not None:
            _ENDED_AT[who] = time.time()
            # What they watched together, kept for the turns after it: LIVE 7 November 2026, after three videos, asked
            # "So what did you think of all three?", she answered that she had opened only one.
            said = [words for _at, words in ended.get("said") or []]
            _WATCHED_TOGETHER.append({"title": ended.get("title") or title, "by": by, "with": who,
                                      "ended": time.time(), "her last word": said[-1] if said else "",
                                      "as it played": [(float(at), str(words)) for at, words in (ended.get("said") or [])[:-1]]})
            del _WATCHED_TOGETHER[:-WATCHED_TOGETHER_KEPT]
        taking.left(who, by=IT)
        # How she came out of it, beside how it moved her: what her sweet spot and her limit are learned from. After
        # the watching is over, so nothing of it is left under way if this cannot be kept.
        try:
            keep({"kind": "felt", "before": felt_before, "after": await how_she_feels(),
                  "highest withdraw": can_take.highest,
                  "moved most": max((felt.evidence for felt in moves.felt), default=0.0), "wall": time.time()})
        except OSError as exc:
            record_degradation("together", exc, severity="info", action="how she came out of a watching was not kept")


def start_watching_in_a_with(open_player: Callable[[], Awaitable[Any]], request: str, *, title: str, url: str,
                             at_most_s: float | None = None, by: str = "") -> asyncio.Task:
    """Begin the watching beside the turn that asked for it; the turn goes on to her answer and ends.

    The watching is the conversation's, not the turn's, so it starts with nothing the turn bound: a task is given a
    copy of the context it was started in. LIVE 21 October 2026 every remark ran inside the request's turn: its clock,
    its delivery, the person's question to check each remark against, and its evidence, where a camera look the
    request was misread as had failed, so each draft saying "I can see" was rejected as claiming a failed sight."""
    import contextvars

    from core.runtime.turn_outcome import current_turn

    task = asyncio.get_running_loop().create_task(
        watch_in_a_with(open_player, request, title=title, url=url, at_most_s=at_most_s, turn=current_turn(), by=by),
        name="watching_in_a_with", context=contextvars.Context())
    _WATCHINGS.add(task)
    task.add_done_callback(_WATCHINGS.discard)
    return task
