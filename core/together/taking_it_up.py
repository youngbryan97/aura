"""Taking up what somebody proposed: the reading, her answer, and the with that follows.

One utterance of somebody's goes through three steps here, in the order Clark
(1996) gives them. It is understood: what it asks of her is read from what
her model expects next (core/together/what_was_proposed.py). It is taken up or
not: when it proposes doing something together, her answer is read off her
record (core/together/her_answer.py). And if she joins, there is from then on
something the two of them are doing (core/together/a_with.py), which holds
until one of them leaves.

Her reply is not written here. What was proposed and what her record says
she wants travel to her model as a reading of the turn
(core/brain/what_she_is_given.py), beside the other readings and in the same
role, and she says it her way.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text_behind
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root
from core.together.a_with import HER, THEM, Together, get_together
from core.together.her_answer import Answer, her_answer
from core.together.what_was_proposed import Proposal, WhatItAsksOfHer

logger = logging.getLogger("Aura.Together")

__all__ = ["TakingItUp", "get_taking_it_up", "install"]

HEADER = "## WHAT WAS PROPOSED TO YOU AND WHAT YOUR RECORD SAYS YOU WANT"
SENSES_HEADER = "## WHAT YOU CAN SENSE AND REACH NOW"
SCREEN_HEADER = "## WHAT IS ON THE SCREEN"

#: What a with is of when what was proposed named nothing she can name.
BEING_TOGETHER = "being together"


class TakingItUp:
    def __init__(self, reader: WhatItAsksOfHer | None = None, *, record: Any = None, together: Together | None = None) -> None:
        self._reader = reader
        self._record = record
        self._together = together
        #: What was read of the newest thing somebody said, by what they said.
        self._this: dict[str, Any] = {}
        self._seeded = False

    # ── the parts, found when first needed ────────────────────────────

    def _reads(self) -> WhatItAsksOfHer:
        if self._reader is None:
            from core.brain.llm.what_she_expects_next import get_expects_next

            self._reader = WhatItAsksOfHer(get_expects_next().score, _stored())
            self._keeps = True
        return self._reader

    def _keep(self) -> None:
        """What the answers usually score is learned once and kept, not relearned at every start."""
        if not getattr(self, "_keeps", False) or self._reader is None:
            return
        try:
            path = _file()
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text_behind(path, json.dumps(self._reader.held()), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            record_degradation(
                "together", exc, severity="warning",
                action="what the answers usually score is held in memory only",
            )

    def _her_record(self) -> Any:
        if self._record is None:
            from core.conation.what_she_wants import get_what_she_wants

            self._record = get_what_she_wants()
        return self._record

    def _withs(self) -> Together:
        if self._together is None:
            self._together = get_together()
        return self._together

    def _seed(self) -> None:
        """Set the reader against what people have already said to her, once."""
        if self._seeded:
            return
        reader = self._reads()
        if reader.heard_before() > 0:
            self._seeded = True
            return
        try:
            from core.conversation.persistence import get_persistence

            stored = get_persistence().get_recent_history(limit=1024)
        # A store that will not read raises the database's own error as often as the OS's.
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError, sqlite3.Error) as exc:
            record_degradation(
                "together", exc, severity="warning",
                action="what an utterance asks of her is read against later ones only",
            )
            return
        said = [str(turn.get("content", "") or "") for turn in stored if str(turn.get("role", "")).lower() == "user"]
        folded = sum(1 for text in dict.fromkeys(said) if reader.remember(text))
        self._seeded = True
        self._keep()
        logger.info("👂 What an utterance asks of her is set against %d things people have said to her.", folded)

    # ── an utterance of theirs ────────────────────────────────────────

    def read(
        self, said: str, *, who: str, at: float | None = None, scored: Any = None
    ) -> dict[str, Any]:
        """Read what `said` asks of her, answer it if it is hers to answer, and keep the with in step.

        With `scored` the scores were taken by the worker that owns the model.
        Without, her fast model takes them here: call it off the event loop.
        A turn's own retries hand over the same words again, and what was
        already read of them is what comes back.
        """
        at = time.time() if at is None else at
        said = " ".join(str(said or "").split())
        if not said:
            return {}
        if self._this.get("said") == said and self._this.get("who") == who:
            return self._this
        if scored is None:
            self._seed()
        proposal: Proposal = self._reads().heard(said, scored=scored)
        self._keep()
        record = self._her_record()
        about = [f"topic:{term}" for term in record.talk.subjects_of(said)]
        withs = self._withs()
        held = withs.current(who)
        answer: Answer | None = None
        if held is not None:
            held.said(THEM, at, about=about, until=at)
        if proposal.kind == "join":
            answer = her_answer(record, who, about)
            if answer.joins and held is None:
                what = about[0] if about else BEING_TOGETHER
                held = withs.begin(who, what, proposed_by=THEM, at=at)
                held.said(THEM, at, about=about, until=at)
                record.started(who, f"with:{what}", label=who.split(":", 1)[-1])
        read = {
            "said": said,
            "who": who,
            "proposal": proposal.as_dict(),
            "answer": answer.as_dict() if answer is not None else None,
            "with": held.status(at) if held is not None and held.open else None,
        }
        self._this = read
        logger.info(
            "🤝 Read as %s (%.2f)%s%s.",
            proposal.kind or "nothing",
            proposal.sure,
            "" if answer is None else f"; her answer: {'joins' if answer.joins else 'asks what it is' if answer.joins is None else 'not now'}",
            "" if not read["with"] else f"; with {who} in {read['with']['what']}",
        )
        return read

    async def heard(self, said: str, *, who: str, at: float | None = None, client: Any = None) -> dict[str, Any]:
        """`read`, off the event loop. Never raises: an utterance not read is answered as it would have been.

        With the client a turn holds, the scores are asked of its worker.
        """
        try:
            scored = None
            if client is not None:
                from core.brain.llm.what_she_expects_next import asked_of_the_worker
                from core.together.what_was_proposed import CONTENT_FREE

                reader = self._reads()
                if not reader.has_a_reference():
                    # Her cortex's own reference, once: what the answers
                    # score after nothing in particular.
                    for empty in CONTENT_FREE:
                        reader.remember(empty, scored=await asked_of_the_worker(client, empty, reader.seconds()) or None)
                scored = await asked_of_the_worker(client, said, reader.seconds())
                if not scored:
                    return {}
            return await asyncio.to_thread(self.read, said, who=who, at=at, scored=scored)
        except (ImportError, ArithmeticError, LookupError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "together", exc, severity="warning",
                action="one utterance was not read for what it asks of her",
            )
            return {}

    def left(self, who: str, *, by: str = THEM, at: float | None = None) -> bool:
        """One of them left. Said by whatever surface the with is on: a window closed, a film over, her going.

        Not read from what was said. On her fast model a parting is not told
        from a greeting well enough to end something on
        (core/together/what_was_proposed.py).
        """
        held = self._withs().end(who, by=by, at=at)
        if held is None:
            return False
        self._her_record().stopped(f"with:{held.what}", who)
        logger.info("🤝 The with with %s in %s is over; %s left.", who, held.what, by)
        return True

    # ── her turn ──────────────────────────────────────────────────────

    def she_said(self, said: str, *, who: str, at: float | None = None, unprompted: bool = False) -> None:
        """Her turn in the with she is in with `who`, if she is in one."""
        held = self._withs().current(who)
        if held is None:
            return
        at = time.time() if at is None else at
        about = [f"topic:{term}" for term in self._her_record().talk.subjects_of(str(said or ""))]
        held.said(HER, at, about=about, until=at, unprompted=unprompted)

    def needs_her_senses(self) -> bool:
        """Whether the turn just read asks her to do or join something, or is part of a with."""
        read = self._this
        return bool(read) and (
            (read.get("proposal") or {}).get("kind") in {"do", "join"} or bool(read.get("with"))
        )

    async def take_what_it_needs(self) -> None:
        """For a turn that asks her to do or join something: what she can sense and reach, measured.

        Live, 13 October, invited to watch a video in his browser, she said
        "I don't have access to video streams or the internet", with her
        screen sense granted and the web reachable. Her capability ledger
        measures each sense by running the precondition its executor runs
        (core/self/capability_ledger.py); those lines are what she is handed,
        and when she is joining something, or is in a with, what is on the
        screen as well. In a with, what the screen shows is also the with's
        shared field (core/together/a_with.py).
        """
        if not self.needs_her_senses():
            return
        read = self._this
        try:
            from core.self.capability_ledger import get_capability_ledger

            measured = await asyncio.to_thread(get_capability_ledger().measure_all)
            read["senses"] = [
                f"- {availability.as_evidence_line()}" for availability in measured.values() if availability.known
            ]
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("together", exc, severity="warning", action="her senses were not measured for this turn")
        if (read.get("proposal") or {}).get("kind") == "join" or read.get("with"):
            read["screen"] = await _the_screen_now()
            held = self._withs().current(read.get("who", ""))
            if held is not None and read["screen"]:
                held.shown(_what_is_in_front(read["screen"]), time.time(), change=_how_much_changed(held, read["screen"]))

    def of_this_turn(self, said: str) -> str:
        """The readings her model is handed with what was said: her answer, her senses, the screen."""
        read = self._this
        if not read or " ".join(str(said or "").split()) != read.get("said"):
            return ""
        blocks = []
        if read.get("senses"):
            blocks.append("\n".join([SENSES_HEADER, *read["senses"]]))
        if read.get("screen"):
            blocks.append(f"{SCREEN_HEADER}\n{read['screen']}")
        answer, proposal = read.get("answer"), read.get("proposal") or {}
        if not answer or proposal.get("kind") != "join":
            return "\n\n".join(blocks)
        wants = {
            True: "joining is worth more to you than what you would leave for it",
            False: "what you would leave for it is worth more to you than joining",
            None: "nothing you have met tells joining from staying apart",
        }[answer["joins"]]
        # What she wants and what she would leave, and not the arithmetic
        # under it: handed the numbers, she recited them (live, 13 October,
        # "higher expected value (-0.209 vs +0.266)"). They are in the log and
        # in her record's status for anybody who asks.
        lines = [
            HEADER,
            "- What was said proposes doing something together.",
            f"- Your record: {wants}.",
        ]
        if answer["leaving"]:
            lines.append(f"- You are in the middle of: {answer['leaving']}.")
        if answer["unknown"]:
            lines.append(f"- New to you: {', '.join(key.split(':', 1)[-1] for key in answer['unknown'])}.")
        return "\n\n".join(["\n".join(lines), *blocks])

    def status(self) -> dict[str, Any]:
        return {"this_turn": dict(self._this), "withs": self._withs().status()}


def _file() -> Path:
    # Scores from different models are not comparable, and the worker's
    # cortex is the scorer on the desktop.
    return state_root() / "data" / "together" / "what_it_asks_of_her.cortex.json"


def _stored() -> dict[str, Any]:
    try:
        held = json.loads(_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        record_degradation(
            "together", exc, severity="warning",
            action="what the answers usually score is learned again from her stored messages",
        )
        return {}
    return held if isinstance(held, dict) else {}


def who_it_is(person: str, sent_from: str) -> str:
    """The key her record keeps a person under: by name, or by where they write from."""
    return f"person:{str(person or '').strip() or 'someone at ' + str(sent_from or '').strip()}"


#: Readings under way, held so they are not collected before they finish.
_HEARING: set[Any] = set()


#: What the runtime calls its own turns when it names them (core/learning/live_learner.py ``a_persons_turn``): a turn
#: from one is not somebody's. LIVE 20 October 2026, a watching asked for by Bryan was watched with "person:someone at
#: desktop_influence_probe", the autonomy conductor's experiment, the last "person" she had heard from.
_ITS_OWN_TURNS = frozenset({"probe", "benchmark"})


def its_own_turn(origin: str) -> bool:
    """Whether a turn, or a person in her record, is one of the runtime's own: a probe or a benchmark of it."""
    from core.runtime.background_policy import origin_tokens

    return bool(origin_tokens(origin) & _ITS_OWN_TURNS)


def who_she_is_with(record: Any) -> str:
    """Who a turn with no name on it is from: whoever she is talking with, or the last person she met."""
    people = [thing for thing in getattr(record, "_things", {}).values()
              if thing.kind == "person" and not its_own_turn(thing.key)]
    if not people:
        return who_it_is("", "")
    talking = [thing for thing in people if thing.under_way]
    return max(talking or people, key=lambda thing: (thing.last, thing.key)).key


def hear_in_the_background(said: str, *, person: str = "", sent_from: str = "") -> None:
    """Start reading what somebody said while the turn it belongs to is prepared.

    The reading waits on her fast model for about half a second, and her
    reply is seconds away, so it runs beside the turn and is there when her
    model is handed the turn. A turn that outruns it is answered without it.
    A turn that arrives with no name and no place on it, as one through the
    kernel does, is from whoever she is with.
    """
    if its_own_turn(sent_from):
        return
    taking = install()
    try:
        who = who_it_is(person, sent_from) if (person or sent_from) else who_she_is_with(taking._her_record())
        task = asyncio.get_running_loop().create_task(taking.heard(said, who=who), name="what_it_asks_of_her")
    except (RuntimeError, ImportError, AttributeError, OSError) as exc:
        record_degradation(
            "together", exc, severity="warning",
            action="one utterance was not read for what it asks of her",
        )
        return
    _HEARING.add(task)
    task.add_done_callback(_HEARING.discard)


async def heard_in_a_turn(said: str, client: Any) -> dict[str, Any]:
    """Read what a person said, inside the turn that answers it, with the client that turn holds.

    This is the door on the desktop. The turn's client is the only one its
    worker will answer while the turn runs, and the reading is taken before
    anything is looked up or written for the reply.
    """
    taking = install()
    read = await taking.heard(said, who=who_she_is_with(taking._her_record()), client=client)
    await taking.take_what_it_needs()
    return read


async def _the_screen_now() -> str:
    """The screen as her own screen reading takes it (core/brain/observable_registry.py), or nothing."""
    try:
        from core.brain.observable_grounding import OBSERVABLES

        screen = next((item for item in OBSERVABLES if item.name == "screen"), None)
        if screen is None:
            return ""
        return str(await asyncio.wait_for(screen.read(""), timeout=screen.timeout_s) or "").strip()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        record_degradation("together", exc, severity="warning", action="the screen was not read for this turn")
        return ""


def _what_is_in_front(screen: str) -> str:
    """The first line of a screen reading, which names what is in front."""
    return next((line.strip() for line in screen.splitlines() if line.strip()), "")


def _how_much_changed(held: Any, screen: str) -> float | None:
    """How different this screen reading is from the last one the with saw, in [0, 1]."""
    words = set(screen.lower().split())
    last = held._screen_words
    held._screen_words = words
    if last is None or not (words | last):
        return None
    return 1.0 - len(words & last) / len(words | last)


async def asked_to_do_something(said: str) -> bool:
    """Whether what was said may ask her to do something, so that a capability could be what it needs.

    False only when it was read and read as a telling or a greeting: neither
    is carried out with a tool. An invitation may be: watching a video with
    somebody needs her senses on it. And her cortex reads some requests as
    invitations, so letting an invitation skip this would leave a request
    undone (13 October: three of twelve on a fresh set). A reading still
    under way is waited for, since it is under a second and what it saves is
    a ten-second pass of her large model over the catalogue of what she can
    do. Anything unread counts as possibly a request.
    """
    wanted = " ".join(str(said or "").split())
    taking = get_taking_it_up()
    if taking._this.get("said") != wanted and _HEARING:
        await asyncio.gather(*list(_HEARING), return_exceptions=True)
    read = taking._this
    if read.get("said") != wanted:
        return True
    return (read.get("proposal") or {}).get("kind") not in {"tell", "greet"}


_TAKING: TakingItUp | None = None


def get_taking_it_up() -> TakingItUp:
    global _TAKING
    if _TAKING is None:
        _TAKING = TakingItUp()
    return _TAKING


def install() -> TakingItUp:
    """Let the reading of a turn travel to her model with the other readings."""
    from core.brain import what_she_is_given

    taking = get_taking_it_up()
    if taking.of_this_turn not in what_she_is_given.OF_THIS_TURN:
        what_she_is_given.OF_THIS_TURN.append(taking.of_this_turn)
    return taking
