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

logger = logging.getLogger("Aura.Together")

__all__ = ["her_answer_before_it_plays", "owed_an_answer", "the_talk_since", "watch_in_a_with", "what_we_are_watching"]

#: How long the watching waits for the turn that asked for it to settle before it starts anyway.
ANSWER_WAITED_S = 180.0
#: How many of the latest looks, and of her latest remarks, the reading of a turn carries.
LATELY_LOOKS, LATELY_SAID = 6, 3

#: What is playing for whom, while it plays.
_PLAYING: dict[str, dict[str, Any]] = {}
_WATCHINGS: set[asyncio.Task] = set()


async def her_answer_before_it_plays(request: str, found: str) -> str:
    """Her reply to the request to watch, before the video starts: her own turn, with what her browsing found of it
    (its title, its length, what its page says of itself) as fresh evidence, as each remark of hers is made
    (core/skills/watching_along.py ``her_turn``). LIVE 20 October 2026 the browsing's receipt went out in its place
    ("what the page plays (Found A short film about colour (0:06)...) That much is verified."), and an empty reply
    sent the request back through the browsing again. What was found stands in only if her turn gives nothing."""
    from core.skills.watching_along import her_turn

    try:
        said = await her_turn(request, f"before it plays. What was found of it: {found}", [])
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


def the_floor_is_free(held: Any) -> Callable[[float], bool]:
    """Whether the with leaves her the floor at a moment: nobody speaking, and nothing said to her still owed."""
    from core.together.a_with import HER, THEM

    def free(now: float) -> bool:
        return bool(held.open) and held.speaking(now) not in (HER, THEM) and not owed_an_answer()

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


def the_talk_since(began: float, request: str, spoken: list[tuple[float, str]]) -> list[dict[str, Any]]:
    """What was said since the watching began, as the exchanges a turn of hers reads: the person's turns and her
    answers from the conversation she keeps, her remarks where they came, the request heading it all."""
    from core.together.remarks import said_since
    from core.utils.injected_blocks import stamp_runtime_payload

    pairs: list[tuple[str, list[str]]] = [(request, [])]
    for _at, role, words in sorted([*said_since(began), *((at, "assistant", w) for at, w in spoken)], key=lambda e: e[0]):
        if role == "user":
            pairs.append((words, []))
        elif words not in pairs[-1][1]:
            pairs[-1][1].append(words)
    return [stamp_runtime_payload({"user": user, "aura": " ".join(aura)}) for user, aura in pairs if aura]


def _read_in_every_turn() -> None:
    from core.brain.what_she_is_given import OF_THIS_TURN

    if what_we_are_watching not in OF_THIS_TURN:
        OF_THIS_TURN.append(what_we_are_watching)


async def _her_answer_settled(turn: Any, *, at_most_s: float) -> None:
    """Until the turn that asked for the watching has settled what it serves, or ``at_most_s`` has passed."""
    deadline = time.monotonic() + at_most_s
    while turn is not None and not getattr(turn, "finalized", True) and time.monotonic() < deadline:  # noqa: ASYNC110 - a turn has no event to wait on; its flag is all it says
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
    await _her_answer_settled(turn, at_most_s=ANSWER_WAITED_S)
    player = await open_player()
    if player is None:
        logger.info("🤝 %s could not be opened where %s watches; nothing to watch together.", title, who)
        return
    playing = _PLAYING[who] = {"title": title, "url": url, "at": 0.0, "duration": 0.0, "moments": [], "said": []}
    spoken: list[tuple[float, str]] = []
    _read_in_every_turn()

    def took_in(moment: Any, plays: dict[str, Any]) -> None:
        playing.update(at=float(plays.get("at") or moment.at), duration=float(plays.get("duration") or 0.0))
        playing["moments"].append(moment)
        held.shown(" ".join(part for part in (moment.seen, moment.read) if part)[:400] or title, time.time())

    def she_said(words: str) -> None:
        playing["said"].append((playing["at"], words))
        spoken.append((time.time(), words))
        taking.she_said(words, who=who, unprompted=True)

    async def her_turn_in_it(asked: str, observation: str, said: list[str]) -> str:
        talk = await asyncio.to_thread(the_talk_since, held.began, asked, list(spoken))
        # Every turn of hers in a with after her answer is a remark: what was asked is already in her conversation.
        return await her_turn(asked, observation, said, exchanges=talk, remark=True)

    try:
        await watch_along(player, request, turn=her_turn_in_it, before=False, may_speak=the_floor_is_free(held), on_moment=took_in,
                          on_said=she_said, keep=kept_at(title, url), at_most_s=at_most_s,
                          remember=lambda watched: into_her_memory(watched, url=url), **pace)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("together", exc, severity="warning", action="the watching together stopped")
    finally:
        _PLAYING.pop(who, None)
        taking.left(who, by=IT)


def start_watching_in_a_with(open_player: Callable[[], Awaitable[Any]], request: str, *, title: str, url: str,
                             at_most_s: float | None = None) -> asyncio.Task:
    """Begin the watching beside the turn that asked for it; the turn goes on to her answer and ends."""
    from core.runtime.turn_outcome import current_turn

    task = asyncio.get_running_loop().create_task(
        watch_in_a_with(open_player, request, title=title, url=url, at_most_s=at_most_s, turn=current_turn()),
        name="watching_in_a_with")
    _WATCHINGS.add(task)
    task.add_done_callback(_WATCHINGS.discard)
    return task
