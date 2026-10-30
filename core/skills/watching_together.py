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

__all__ = ["her_answer_before_it_plays", "owed_an_answer", "the_talk_since", "watch_in_a_with", "what_we_are_watching"]

#: How long the watching waits for the turn that asked for it to settle before it starts anyway.
ANSWER_WAITED_S = 180.0
#: How many of the latest looks, and of her latest remarks, the reading of a turn carries.
LATELY_LOOKS, LATELY_SAID = 6, 3

#: What is playing for whom, while it plays.
_PLAYING: dict[str, dict[str, Any]] = {}
_WATCHINGS: set[asyncio.Task] = set()


#: How many exchanges since the request a turn of hers reads while watching, besides the request and her answer.
RECENT_EXCHANGES = 8

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


class WorthSaying:
    """Whether what has played since her last remark is worth one now: her own rule for speaking unasked inside a with
    (core/together/when_to_say_something.py), how much it interests her (core/together/remarks.py interest_in) against
    what taking his attention from what plays costs. Not that something new was said: LIVE 20 October 2026 new words
    were the whole of the rule, and a sponsor's read is all new words.

    Interest is the most any look since her last remark held, so a remark at a lull can be about what came before it.
    What plays will not be there to point at later, so all of a remark's worth is lost by waiting (the rule's m)."""

    def __init__(self, held: Any) -> None:
        self.held, self.interest, self.thing = held, 0.0, ""

    def took_in(self, moment: Any) -> None:
        from core.together.remarks import interest_in

        seen = " ".join(part for part in (moment.seen, moment.read) if part)
        change = self.held.field_of_view[-1].change if self.held.field_of_view else None
        interest, thing = interest_in(seen, change)
        if interest >= self.interest:
            self.interest, self.thing = interest, thing or seen[:80]

    def remarked(self) -> None:
        self.interest, self.thing = 0.0, ""

    def __call__(self, _moment: Any, now: float) -> bool:
        from core.together.when_to_say_something import worth_saying_now

        say = worth_saying_now(self.held, now, interest=self.interest, thing=self.thing, lost_by_waiting=1.0)
        logger.info("🤝 %s: %s (interest %.2f in %s)", "worth saying now" if say.speak else "stays quiet", say.why,
                    self.interest, self.thing or "what is on")
        return say.speak


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


def what_we_are_watching(_said: str = "") -> str:
    """The reading every turn of the conversation carries while something plays: what, how far in, lately."""
    if not _PLAYING:
        return ""
    from core.skills.watching_along import clock, what_she_watched

    blocks = []
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
    events = [*((at, role, words) for at, role, words in said_since(began) if not (role == "assistant" and words in remarks)),
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
                          at_most_s: float | None = None, turn: Any = None, who: str = "", **pace: float) -> None:
    """Watch in the with the request opened (or one begun for it), once her answer to it has gone out."""
    from core.skills.watching_along import her_turn, into_her_memory, kept_at, watch_along
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
    player = await open_player()
    if player is None:
        logger.info("🤝 %s could not be opened where %s watches; nothing to watch together.", title, who)
        return
    playing = _PLAYING[who] = {"title": title, "url": url, "at": 0.0, "duration": 0.0, "moments": [], "said": []}
    spoken: list[tuple[float, str, str]] = []
    _read_in_every_turn()

    worth = WorthSaying(held)

    def took_in(moment: Any, plays: dict[str, Any]) -> None:
        from core.together.taking_it_up import _how_much_changed

        playing.update(at=float(plays.get("at") or moment.at), duration=float(plays.get("duration") or 0.0))
        playing["moments"].append(moment)
        shown = " ".join(part for part in (moment.seen, moment.read) if part)[:400] or title
        held.shown(shown, time.time(), change=_how_much_changed(held, shown))
        worth.took_in(moment)

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

    try:
        await watch_along(player, request, turn=her_turn_in_it, before=False, may_speak=the_floor_is_free(held),
                          worth=worth, on_moment=took_in,
                          on_said=she_said, keep=kept_at(title, url), at_most_s=at_most_s,
                          remember=lambda watched: into_her_memory(watched, url=url), **pace)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("together", exc, severity="warning", action="the watching together stopped")
    finally:
        _PLAYING.pop(who, None)
        taking.left(who, by=IT)


def start_watching_in_a_with(open_player: Callable[[], Awaitable[Any]], request: str, *, title: str, url: str,
                             at_most_s: float | None = None) -> asyncio.Task:
    """Begin the watching beside the turn that asked for it; the turn goes on to her answer and ends.

    The watching is the conversation's, not the turn's, so it starts with nothing the turn bound: a task is given a
    copy of the context it was started in. LIVE 21 October 2026 every remark ran inside the request's turn: its clock,
    its delivery, the person's question to check each remark against, and its evidence, where a camera look the
    request was misread as had failed, so each draft saying "I can see" was rejected as claiming a failed sight."""
    import contextvars

    from core.runtime.turn_outcome import current_turn

    task = asyncio.get_running_loop().create_task(
        watch_in_a_with(open_player, request, title=title, url=url, at_most_s=at_most_s, turn=current_turn()),
        name="watching_in_a_with", context=contextvars.Context())
    _WATCHINGS.add(task)
    task.add_done_callback(_WATCHINGS.discard)
    return task
