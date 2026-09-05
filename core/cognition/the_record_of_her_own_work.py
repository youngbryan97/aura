"""What she did, what it cost, and what it was worth — kept, because a decision needs it.

The previous mandate asked whether the language of her future learning could be
a product of experience. This one asks whether she DECIDES to change it, and
that is a different kind of question with a different prerequisite. A system
cannot choose between courses of action whose values it cannot estimate, and it
cannot estimate them from what it does not record.

So this is the prerequisite, and it is deliberately small. An episode is one
occasion of trying to say something: what was asked, which route answered it,
what that cost in candidates walked, what it used, and what — if anything — was
admitted as a result. Everything the decision rule needs is a statistic of
these, and nothing here is a category of opportunity.

Three statistics, and each answers a question that was previously unanswerable
from inside:

    how often has this come up      the recurrence estimate, which is what
                                    turns "worth doing once" into "worth
                                    carrying"
    what has this route cost        the attribution, which is what makes
                                    "which part of me is slow" a fact
    when was this last used         the disuse, which is what makes dropping
                                    something a measurement

Bounded, because memory is. The ring keeps the most recent episodes and the
counts survive the episodes they were taken from, so a recurrence seen a
thousand episodes ago still counts even though the episode itself is gone. That
is the shape finite memory forces: keep the statistics, forget the instances.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from core.runtime.errors import record_degradation

__all__ = [
    "Episode",
    "HOW_MANY_CASES_ARE_KEPT",
    "HOW_MANY_EPISODES_ARE_KEPT",
    "attribution",
    "episodes",
    "forget_the_record",
    "how_long_since",
    "how_much_is_unwritten",
    "how_often",
    "keep_the_record",
    "note_a_step",
    "note_an_episode",
    "other_families",
    "note_a_use",
    "recall_the_record",
    "remember_what_she_had",
    "start_counting_again",
    "steps_walked",
    "the_record",
    "what_it_has_cost",
]

logger = logging.getLogger("Aura.TheRecordOfHerOwnWork")

#: How many episodes are kept whole. The counts outlive them, so a structure
#: seen long ago still counts; what is lost is the instance, not the statistic.
#: Read off what a decision needs rather than chosen: the estimates below use
#: counts, and only the attribution reads whole episodes.
HOW_MANY_EPISODES_ARE_KEPT = 512

#: Set to send it somewhere else. Left alone in the live runtime; a test that
#: wants its own file names one here.
#: How many episodes keep their cases. Small: the cases are the only part of
#: an episode that is not a number, and a probe wants a handful of families
#: rather than a corpus.
HOW_MANY_CASES_ARE_KEPT = 24

_KEPT_AT: Path | None = None


def _kept_at() -> Path:
    """Where it goes.

    Read every time rather than fixed at import, because a test run has its own
    state root and a path resolved once would aim a test's writes at the live
    instance — which the ownership guard then refuses, so the write is lost and
    the persistence is never actually exercised.
    """
    if _KEPT_AT is not None:
        return _KEPT_AT
    from core.runtime.state_ownership import state_root

    return state_root() / "the_record_of_her_own_work.json"


@dataclass(frozen=True, slots=True)
class Episode:
    """One occasion of trying to say something, and what it cost."""

    #: What was asked, as a key that recurs when the same shape recurs.
    family: str
    #: Which route answered it, or nothing where none did.
    route: str | None
    #: Candidates walked. The unit everything here is priced in.
    walked: int
    #: Library entries the answer was built from.
    used: tuple[str, ...] = ()
    #: What was admitted because of it, where anything was.
    admitted: str | None = None
    #: What was TRIED, whether or not it worked. ``route`` names the action
    #: only when the change was kept, so a family where everything she has was
    #: tried and nothing held was indistinguishable from one she never tried.
    #: The first of those calls for a new operator; the second calls for the
    #: one she already has.
    tried: str | None = None
    #: The cases themselves, for a few episodes, so a change can be judged on
    #: something other than the occasion that provoked it.
    #:
    #: Judging a change on the trigger sample is how every compressor comes to
    #: overfit the incident that woke it: the thing was chosen because it
    #: helped there, so of course it helps there. Held-out families are the
    #: only honest answer, and holding them out means keeping some.
    about: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...] = ()
    when: float = field(default_factory=time.monotonic)

    def describes(self) -> str:
        said = self.route or "nothing"
        return f"{self.family}: {said} after {self.walked:,}"


@dataclass
class Record:
    """Everything a decision about developing needs, and nothing else."""

    kept: list[Episode] = field(default_factory=list)
    #: How often each family has come up, past the episodes still held.
    families: Counter = field(default_factory=Counter)
    #: How often each library entry has been used.
    uses: Counter = field(default_factory=Counter)
    #: Which episode each entry was last used at, counted in episodes.
    last_used: dict[str, int] = field(default_factory=dict)
    #: How many episodes there have ever been.
    seen: int = 0

    def note(self, episode: Episode) -> None:
        self.seen += 1
        self.families[episode.family] += 1
        for name in episode.used:
            self.uses[name] += 1
            self.last_used[name] = self.seen
        self.kept.append(episode)
        # Only the newest few keep their cases; the rest keep their numbers.
        holding = [one for one in self.kept if one.about]
        for old in holding[:-HOW_MANY_CASES_ARE_KEPT]:
            self.kept[self.kept.index(old)] = Episode(
                family=old.family,
                route=old.route,
                walked=old.walked,
                used=old.used,
                admitted=old.admitted,
                # Dropped here once, which turned every compacted episode
                # from "everything was tried and nothing held" into "nothing
                # was ever tried" — the two states this field exists to keep
                # apart, and they call for opposite actions.
                tried=old.tried,
                when=old.when,
            )
        if len(self.kept) > HOW_MANY_EPISODES_ARE_KEPT:
            # The instance goes, the counts stay. That is what finite memory
            # forces, and it is why the counts are kept beside the ring rather
            # than computed from it.
            del self.kept[: len(self.kept) - HOW_MANY_EPISODES_ARE_KEPT]


_RECORD = Record()

#: Candidates walked since the counter was last reset. The one unit everything
#: about developing is priced in, and it has one home so that every search
#: reports to the same place.
#:
#: Without this the positional search spent thousands of candidates and
#: reported none, so every answer it gave priced at nothing, the ceiling was
#: nothing, and no change was ever worth making on the path that answers most
#: questions.
_WALKED = [0]


def note_a_step(how_many: int = 1) -> None:
    """One more candidate walked. Called from inside a search, not around it."""
    _WALKED[0] += max(0, int(how_many))


def steps_walked() -> int:
    return _WALKED[0]


def start_counting_again() -> int:
    """Reset the counter and give back what it held."""
    was = _WALKED[0]
    _WALKED[0] = 0
    return was


def the_record() -> Record:
    """The record itself, for anything that needs more than a statistic."""
    _remember_what_she_had()
    return _RECORD


def episodes() -> tuple[Episode, ...]:
    _remember_what_she_had()
    return tuple(_RECORD.kept)


def note_an_episode(
    family: str,
    *,
    route: str | None,
    walked: int,
    used: Sequence[str] = (),
    admitted: str | None = None,
    tried: str | None = None,
    about: Sequence[Any] = (),
) -> Episode:
    """Write down one occasion. Called from the answering path, not from a test."""
    made = Episode(
        family=str(family),
        route=route,
        walked=max(0, int(walked)),
        used=tuple(str(one) for one in used),
        admitted=admitted,
        tried=tried if tried is not None else route,
        about=tuple(
            (tuple(before), tuple(after)) for before, after in (about or ())
        ),
    )
    _remember_what_she_had()
    _RECORD.note(made)
    _maybe_write()
    return made


def note_a_use(name: str) -> None:
    """Record that something was used, where no whole episode is being written.

    A library entry used inside ordinary cognition is used, and counting it
    only when an episode is written would make everything look disused.
    """
    _remember_what_she_had()
    _RECORD.uses[str(name)] += 1
    _RECORD.last_used[str(name)] = _RECORD.seen


def how_often(family: str) -> int:
    """How many times this shape has come up. The recurrence estimate."""
    _remember_what_she_had()
    return int(_RECORD.families.get(str(family), 0))


def how_long_since(name: str) -> int | None:
    """Episodes since this entry was last used, or nothing if it never was."""
    _remember_what_she_had()
    at = _RECORD.last_used.get(str(name))
    return None if at is None else max(0, _RECORD.seen - at)


def what_it_has_cost(route: str) -> int | None:
    """What this route has cost on average, or nothing where it never ran.

    The measured cost of a developmental action, so nothing has to estimate
    what has already been observed.
    """
    _remember_what_she_had()
    spent = [one.walked for one in _RECORD.kept if one.route == route]
    return int(round(sum(spent) / len(spent))) if spent else None


def other_families(than: str) -> list[tuple[str, tuple]]:
    """Families she has met that are not this one, with their cases.

    What a change is judged on. A change chosen because it helps the family in
    hand will help the family in hand; whether it helps anything else is the
    question, and this is what answers it.
    """
    _remember_what_she_had()
    found: dict[str, tuple] = {}
    for one in _RECORD.kept:
        if one.family != than and one.about and one.family not in found:
            found[one.family] = one.about
    return sorted(found.items())


def attribution() -> dict[str, dict[str, Any]]:
    """What each route has cost and how often it has answered.

    The self-model, and it is a plain one: which part of her spends the search
    is a fact about the record rather than a thing she believes about herself.
    A route that answers rarely and costs much is a bottleneck, and that is
    readable here without anything having to say the word.
    """
    _remember_what_she_had()
    spent: Counter = Counter()
    answered: Counter = Counter()
    tried: Counter = Counter()
    for one in _RECORD.kept:
        where = one.route or "nothing answered"
        spent[where] += one.walked
        tried[where] += 1
        if one.route is not None:
            answered[where] += 1
    return {
        where: {
            "walked": int(spent[where]),
            "answered": int(answered[where]),
            "episodes": int(tried[where]),
            "each": round(spent[where] / max(1, tried[where]), 1),
        }
        for where in sorted(tried)
    }


def keep_the_record() -> bool:
    """Write it down, so a developmental history survives a restart."""
    body = {
        "seen": _RECORD.seen,
        "families": dict(_RECORD.families),
        "uses": dict(_RECORD.uses),
        "last_used": dict(_RECORD.last_used),
        "kept": [
            {
                "family": one.family,
                "route": one.route,
                "walked": one.walked,
                "used": list(one.used),
                "admitted": one.admitted,
                "tried": one.tried,
                "about": [
                    [list(before), list(after)] for before, after in one.about
                ],
            }
            for one in _RECORD.kept
        ],
    }
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "the_record_of_her_own_work.keep", domain="state_mutation"
        ):
            get_file_write_gateway().ensure_directory(
                _kept_at().parent, source="the_record_of_her_own_work"
            )
            get_file_write_gateway().write_text(
                _kept_at(), json.dumps(body), source="the_record_of_her_own_work"
            )
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "the_record_of_her_own_work", exc, severity="info",
            action="keep the record of her own work",
        )
        return False


def recall_the_record() -> int:
    """Put it back. Returns how many episodes came back."""
    try:
        held = json.loads(_kept_at().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(held, dict):
        return 0
    _RECORD.seen = int(held.get("seen") or 0)
    _RECORD.families = Counter(
        {str(k): int(v) for k, v in (held.get("families") or {}).items()}
    )
    _RECORD.uses = Counter(
        {str(k): int(v) for k, v in (held.get("uses") or {}).items()}
    )
    _RECORD.last_used = {
        str(k): int(v) for k, v in (held.get("last_used") or {}).items()
    }
    _RECORD.kept = []
    for row in held.get("kept") or ():
        if not isinstance(row, dict):
            continue
        _RECORD.kept.append(
            Episode(
                family=str(row.get("family") or ""),
                route=row.get("route"),
                walked=int(row.get("walked") or 0),
                used=tuple(str(one) for one in row.get("used") or ()),
                admitted=row.get("admitted"),
                tried=row.get("tried"),
                about=tuple(
                    (tuple(pair[0]), tuple(pair[1]))
                    for pair in row.get("about") or ()
                    if isinstance(pair, list) and len(pair) == 2
                ),
            )
        )
    return len(_RECORD.kept)


def forget_the_record() -> None:
    """Start again. Used by tests, and by nothing else."""
    _RECORD.kept.clear()
    _RECORD.families.clear()
    _RECORD.uses.clear()
    _RECORD.last_used.clear()
    _RECORD.seen = 0
    with _WRITING:
        _RESTORED[0] = True
        _UNWRITTEN[0] = 0


# ── surviving a restart, without anybody remembering to ask ──────────────
#
# Both halves above existed and were correct, and nothing called either of
# them. So the developmental policy chose its next self-change from H_t, the
# accumulated evidence about her own performance, and every process restart
# set H_t back to the empty set — not all learning, because other artefacts
# persist by other means, but this one, the metacognitive history the policy
# is a function OF.
#
# The fix is not a boot caller. A seam that needs somebody to remember it is a
# seam that loses its caller again the next time the boot sequence is
# rewritten, and this one had never had a caller at all. So the record
# restores itself the first time anything asks it a question, and writes
# itself back on a cadence and at exit.
#
# The write goes to a daemon thread. The record is noted from the answering
# path, and an fsync taken there once froze the live event loop for twenty
# minutes; a thread that coalesces its work and holds nothing the answer needs
# cannot do that.

#: Episodes between write-backs. Small enough that a hard kill loses an
#: afternoon rather than a history, large enough that answering is never
#: waiting on a file.
HOW_OFTEN_IT_IS_WRITTEN = 16

_RESTORED = [False]
_UNWRITTEN = [0]
_WRITING = threading.Lock()
_WRITER: threading.Thread | None = None


def _remember_what_she_had() -> None:
    """Read the record back, once, before the first question about it.

    Idempotent and quiet. A restart with no file is a first run, which is a
    normal state and not a degradation.
    """

    global _WRITER
    with _WRITING:
        if _RESTORED[0]:
            return
        _RESTORED[0] = True
    came_back = recall_the_record()
    if came_back:
        logger.info(
            "The record of her own work came back: %d episodes, %d families, "
            "%d in total seen",
            came_back,
            len(_RECORD.families),
            _RECORD.seen,
        )
    with _WRITING:
        if _WRITER is None:
            _WRITER = threading.Thread(
                target=_write_when_asked,
                name="the-record-of-her-own-work",
                daemon=True,
            )
            _WRITER.start()
    atexit.register(_write_it_out_now)


def _write_when_asked() -> None:
    """One writer, coalescing. Never more than one save in flight."""

    while True:
        _ASKED_TO_WRITE.wait()
        _ASKED_TO_WRITE.clear()
        try:
            keep_the_record()
        except Exception as exc:  # noqa: BLE001 - a writer thread may not die
            record_degradation(
                "the_record_of_her_own_work", exc, severity="info",
                action="write the record of her own work in the background",
            )


_ASKED_TO_WRITE = threading.Event()


def _write_it_out_now() -> bool:
    """Save on this thread. Used at exit, where there is no later."""

    return keep_the_record()


def _maybe_write() -> None:
    """Ask the writer to run, at most once every HOW_OFTEN_IT_IS_WRITTEN."""

    with _WRITING:
        _UNWRITTEN[0] += 1
        if _UNWRITTEN[0] < HOW_OFTEN_IT_IS_WRITTEN:
            return
        _UNWRITTEN[0] = 0
    _ASKED_TO_WRITE.set()


def remember_what_she_had() -> None:
    """Restore the record now, for a boot that would rather not wait.

    Nothing has to call this. Every reader and writer above does it already;
    this is the same thing said out loud, so a boot sequence can pay the read
    once rather than on the first question.
    """

    _remember_what_she_had()


def how_much_is_unwritten() -> int:
    """Episodes noted since the last write-back was asked for."""

    with _WRITING:
        return int(_UNWRITTEN[0])
