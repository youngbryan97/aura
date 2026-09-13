"""Where the run stands, and which moves are worth making from here.

The options a screen offers, the ways back out of it, the pacing that stops her
hammering a control that is already working, and — the part that matters most —
the moves she will not make, because a pursuit that will try anything is a
pursuit that will eventually try the destructive thing. What is here is
judgement about a run rather than contact with the screen.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from core.cognition.something_she_keeps_true import (
    what_it_rules_out,
    what_to_hold_now,
)
from core.cognition.what_is_still_open import what_is_still_open
from core.cognition.what_would_have_to_be_true import a_way_to_get_there
from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation
from core.runtime.watched_goal import BROWSERS

from .screen_pursuit_surface import (
    LABEL_REACH,  # noqa: F401
    PRESSABLE_KEYS,
    _bound_to_a_window,  # noqa: F401
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _value_is_on_screen,  # noqa: F401
    labelled_by,  # noqa: F401
    )

logger = logging.getLogger("Aura.ScreenPursuit")


#: The moves offered when a caller does not name its own. Arrow keys are the
#: universal keyboard affordance: they mean something on a board, a list, a
#: map, a carousel, a form. A caller with a richer vocabulary passes its own.
DEFAULT_MOVES: tuple[str, ...] = ("up", "down", "left", "right")


#: How often language is consulted on a run of routine moves.
#:
#: A board changes a little each move, so re-reasoning every single one buys
#: little and costs the whole cycle: measured live, a language pass took
#: about eight seconds and a decision from evidence takes none, on a loop
#: that needs hundreds of moves. Language is asked when the answer is most
#: likely to differ — the first move, after a restart, when what she is doing
#: has stopped working — and periodically in between so the commentary stays
#: hers rather than a run of bandit statistics.
LANGUAGE_EVERY = 5


def screen_options(keys: Sequence[str] = DEFAULT_MOVES) -> list[Any]:
    """The moves really available on a screen, each carrying its own test.

    An option states what should be different once it lands, so the check is
    made by measurement rather than by asking the same faculty that chose.
    For a keypress the honest claim is narrow: the view is not what it was.
    A key that changes nothing is a key that did nothing, whatever the
    keystroke's own receipt said.
    """
    from core.agency.deliberate_action import ActionOption, Expectation

    options: list[Any] = []
    for key in keys:
        name = str(key).strip().lower()
        if name not in PRESSABLE_KEYS:
            continue
        options.append(
            ActionOption(
                name=name,
                detail=f"press {name}",
                expectation=Expectation(
                    changed=True,
                    describes=f"the view to be different after {name}",
                ),
            )
        )
    return options


#: Controls that begin a task again, by the words they are usually labelled
#: with. Matched against what is really on screen — never inferred, and never
#: clicked unless the run has actually stopped getting anywhere.
RESTART_LABELS = ("new game", "restart", "play again", "try again", "start over", "reset")


#: The two ways out of an impasse that are not "keep pressing".
START_OVER = "start over"


SEE_IT_THROUGH = "see it through"


def restart_controls(observation: dict[str, Any]) -> frozenset[str]:
    """Every control on screen that would begin the task again, by its words.

    The set rather than the first one, because what says a task has finished
    is a way back that was NOT there a moment ago. A game that keeps "New
    Game" above the board all game long puts "Try again" over it when it
    ends, and only the second of those is news.
    """
    found: set[str] = set()
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip()
        if not text or len(text) > 40:
            continue
        lowered = text.lower()
        if any(label in lowered for label in RESTART_LABELS):
            found.add(lowered)
    return frozenset(found)


def a_run_she_can_carry(
    named: Sequence[str],
    from_here: Any,
    foresee: Any,
    pick: Any,
    choices: Sequence[str],
    acted: int,
    changed: int,
) -> list[str]:
    """The moves after this one that are still moves about the situation then.

    A ranking is a judgement about alternatives at one moment. Sent as a
    sequence it is four answers to the same question, executed in order: on a
    page where a key may simply do nothing, the second-best thing to try is
    the next thing to try, and on a board the second-best move is a move for
    a board that no longer exists.

    So a named run keeps going while each step is still what she would choose
    from the board the step before it leaves. That is the same judgement
    carried forward, which is what committing without words claims to be, and
    it is checkable.

    Where she cannot say what the board becomes, a run outlives its own first
    act only where acting here has been SEEN to change nothing. ``acted`` and
    ``changed`` are her own record of that, and the difference between "no
    evidence" and "evidence of nothing" is the whole of it: read as the same
    thing, a fresh world gets four keys a cycle from the first move, no pair
    she learns from spans one act, so nothing is ever recorded as having
    changed — and the condition that lets her go four at a time is the one
    her going four at a time keeps true.

    LIVE 2026-09-04: a ranking of four arrow keys went out as a four-move
    plan every cycle, down then up then left then right, on a board where
    each of them changed everything. Sixty moves a game, no rule ever formed,
    and every pair the learner was handed named an act that did not produce
    it.
    """
    if not named:
        return []
    if foresee is None or from_here is None or pick is None:
        if acted <= 0:
            # Nothing known about this place at all. One act, then look.
            return []
        return [] if changed > 0 else list(named)
    kept: list[str] = []
    where = from_here
    for step in named:
        if pick(where, list(choices)) != step:
            break
        went = foresee(where, step)
        if went is None or went == where:
            break
        kept.append(step)
        where = went
    return kept


def a_way_back_that_was_not_there(
    now: frozenset[str], at_the_start: frozenset[str] | None
) -> bool:
    """Whether a way to begin the task again has APPEARED since it began.

    The appearing is the whole signal. A control that has been on screen
    since before she made a move is furniture — every form with a Reset,
    every wizard with Start Over, every game with a permanent New Game — and
    reading its mere presence as an ending finishes the task on cycle one.

    ``at_the_start`` of None means nothing has been compared against yet, so
    nothing can have appeared.
    """
    if at_the_start is None:
        return False
    return bool(now - at_the_start)


def restart_control(observation: dict[str, Any]) -> tuple[str, float, float] | None:
    """A control on screen that would begin the task again, if there is one."""
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip()
        if not text or len(text) > 40:
            continue
        lowered = text.lower()
        if not any(label in lowered for label in RESTART_LABELS):
            continue
        try:
            return (
                text,
                float(region.get("center_x", region.get("x"))),
                float(region.get("center_y", region.get("y"))),
            )
        except (TypeError, ValueError):
            continue
    return None


def ways_out(observation: dict[str, Any], *, ended: bool = False) -> list[Any]:
    """What she can do about being stuck, as options she chooses between.

    A loop whose only moves are inside the task can only press harder at
    something that has stopped working. These are moves about the task: begin
    it again knowing what she now knows, or finish it badly on purpose,
    because the ending is where the evidence about how it goes wrong is.

    Offered only once the in-task moves have demonstrably stopped working, so
    an ordinary run never sees them and nothing gets restarted casually.
    """
    from core.agency.deliberate_action import ActionOption, Expectation

    options: list[Any] = [
        ActionOption(
            name=SEE_IT_THROUGH,
            detail="keep playing this out and learn from how it ends",
            # Needing a reason in words protects live work from being thrown
            # away on a ranking. Where nothing answers any more there is no
            # live work to protect, and the alternative to choosing is
            # pressing keys into something that has finished.
            needs_words=not ended,
            expectation=Expectation(
                changed=False, describes="to reach the end of this attempt and know why it failed"
            ),
        )
    ]
    control = restart_control(observation)
    if control is not None:
        label, x, y = control
        options.insert(
            0,
            ActionOption(
                name=START_OVER,
                params={"label": label, "x": x, "y": y},
                needs_words=not ended,
                detail=f"begin again with {label!r}, knowing what this attempt taught",
                expectation=Expectation(changed=True, describes="a fresh start on the same task"),
            ),
        )
    return options


#: What she can do when her voice falls behind her hands. Moves about
#: herself rather than about the task, in the same shape as the ways out of
#: an impasse: offered only when the situation is real, chosen through the
#: ordinary deliberation, and recorded with a reason.
SLOW_DOWN = "slow down"


SAY_LESS = "say less"


PRESS_ON = "press on"


def narration_backlog() -> dict[str, int]:
    """How far behind the voice is. Empty when there is no surface to speak to."""
    try:
        from core.perception.ambient_presence import get_ambient_presence

        return dict(get_ambient_presence().narration_backlog())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return {}


def pacing_options(backlog: dict[str, int]) -> list[Any]:
    """What she can do about acting faster than she can speak.

    Two faculties running at once will not run at the same speed. Noticing
    that is not enough on its own — noticing without a lever is a status
    line — so each of these is something she can actually do: wait for the
    voice to catch up, say less per move, or carry on and let some of it go
    unsaid. Any of the three is a defensible answer, which is why it is a
    decision and not a rule.
    """
    from core.agency.deliberate_action import ActionOption, Expectation

    waiting = int(backlog.get("waiting", 0) or 0)
    if waiting <= 0:
        return []
    return [
        ActionOption(
            name=SLOW_DOWN,
            detail=f"wait for my commentary to catch up — {waiting} line(s) behind",
            expectation=Expectation(changed=False, describes="my words to catch up with my hands"),
        ),
        ActionOption(
            name=SAY_LESS,
            detail="name each move without explaining it, and keep this pace",
            expectation=Expectation(changed=False, describes="to keep up by saying less"),
        ),
        ActionOption(
            name=PRESS_ON,
            detail="carry on at this pace and let some of it go unsaid",
            expectation=Expectation(changed=False, describes="to keep playing and lose some commentary"),
        ),
    ]


async def let_the_voice_catch_up(before: dict[str, int], *, patience: float = 4.0) -> None:
    """Wait for the backlog to drain, bounded, without inventing a delay.

    Sized by the thing actually being waited for rather than by a number
    somebody picked: it returns as soon as the queue is shorter than it was,
    and gives up after ``patience`` seconds so a surface nobody is reading
    cannot stall the run.
    """
    started = time.monotonic()
    was = int(before.get("waiting", 0) or 0)
    while time.monotonic() - started < patience:
        await asyncio.sleep(0.25)
        now = int(narration_backlog().get("waiting", 0) or 0)
        if now < was:
            return


def _ask_again_after(asked_at: int) -> int:
    """How many moves may pass before the question is put again.

    A first plan can wait longer than a second one, because the first is
    waiting for the screen to say which part of it is the task. The horizon
    it waits to is the one past which an approach nobody revisits is a habit.
    """
    from core.agency.standing_strategy import RECONSIDER_AFTER

    return LANGUAGE_EVERY if asked_at >= 0 else RECONSIDER_AFTER


def _time_left(began: float, max_seconds: float, deadline_at: float) -> float:
    """What is left of the budget, on whichever clock started first.

    A caller that began counting before this action did says so, and its
    deadline wins: otherwise the setup between the two is free time that the
    outer deadline is then blamed for.
    """
    now = time.monotonic()
    ends_at = began + float(max_seconds)
    if deadline_at > 0.0:
        ends_at = min(ends_at, float(deadline_at))
    return max(1.0, ends_at - now)


def _within_the_run(think: Any, ends_at: float) -> Any:
    """Her thinking, bounded by what is left of the run rather than its own budget.

    A cycle checks the clock at its top and then goes away to think. When the
    thought outlasts the run, the deadline is only noticed after it returns,
    and by then the caller outside — which has room to report and nothing
    more — has already cancelled everything. LIVE 2026-08-26: twenty-nine
    narrated moves, a 64 built into the corner, and "Operation took too long.
    Completed 0/0 steps."
    """
    if think is None or ends_at <= 0.0:
        return think

    async def bounded(objective: str, evidence: Any) -> Any:
        left = ends_at - time.monotonic()
        if left <= 1.0:
            raise TimeoutError("the run is out of time to think")
        return await asyncio.wait_for(think(objective, evidence), timeout=left)

    return bounded


def _say_what_kind_of_problem(
    knows: Any, acts: Any, state: Any, toward: str, said_already: dict[str, bool]
) -> None:
    """Name the shape of what she is in, once she has worked out enough to name it.

    Recognising the kind of problem is the general part; what it calls for is
    allowed to be as specialised as the problem is. Said out loud because a
    watcher cannot otherwise tell a mind that recognised its situation from
    one that got lucky in it.
    """
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .screen_pursuit import (
        _tell,
    )

    if said_already.get("shape"):
        return
    try:
        from core.agency.what_kind_of_problem import recognise  # noqa: PLC0415

        suits = recognise(
            acts=[getattr(option, "name", str(option)) for option in acts or ()],
            knows_how_it_moves=getattr(knows, "rules", knows),
            state=state,
            toward=toward,
        )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="acted without naming the problem"
        )
        return
    if not suits.shape.transition_known:
        return
    said_already["shape"] = True
    _tell(f"I know what kind of thing this is now: {suits.shape.named()}.")


#: How many screenfuls down she will look for the thing before deciding the
#: page does not have one. A page is taller than a screen and what she came
#: for is usually below the writing about it — six is enough to clear a
#: heading, a paragraph and an advertising rail without walking a long article
#: end to end.
SCREENFULS_TO_LOOK = 6


#: The least a reading has to hold before it counts as a thing laid out rather
#: than as prose that happens to have numbers in it.
ENOUGH_TO_BE_A_THING = 4


def _is_a_thing_laid_out(reading: Any) -> bool:
    """Whether this reading holds something arranged in rows and columns."""
    rows = int(getattr(reading, "rows", 0) or 0)
    columns = int(getattr(reading, "columns", 0) or 0)
    occupied = getattr(reading, "occupied", None)
    if rows < 2 or columns < 2 or not callable(occupied):
        return False
    return occupied() >= ENOUGH_TO_BE_A_THING


async def _bring_it_into_view(look: Any, read: Any, cannot_see: dict[str, str] | None = None) -> int:
    """Scroll down until what she came for is on screen, or the page runs out.

    A page is taller than a screen. She opened a real sliding puzzle, read the
    heading and the advertising above it, found no part of what she could see
    that answered to her, and reported truthfully that nothing on screen
    offered a move — with the board eleven screenfuls further down. LIVE
    2026-08-27, and the run ended having made none.

    Scrolling commits to nothing. It moves a view and is undone by moving
    back, which is the same line drawn around every other input she is allowed
    to try without knowing what it will do.

    Returns how far down she had to go, so a caller can say so. ``cannot_see``
    is filled in when the screen could not be read at all, which is a
    different fact from finding nothing on it.
    """
    cannot_see = {} if cannot_see is None else cannot_see
    from core.capabilities.host_automation import get_host_automation

    try:
        hands = get_host_automation()
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("screen_pursuit", exc, action="scroll to find the thing")
        return 0
    for down in range(SCREENFULS_TO_LOOK):
        seen = await look()
        if not seen.get("ok"):
            why = str(seen.get("error") or "no reason given")
            logger.info(
                "looking for what she came for, %d down: nothing could be read (%s)", down, why
            )
            cannot_see["reason"] = why
            return down
        here = read(seen)
        if _is_a_thing_laid_out(here):
            logger.info(
                "what she came for is %d screenful(s) down: %dx%d with %d thing(s) in it",
                down,
                here.rows,
                here.columns,
                here.occupied(),
            )
            return down
        # Say what WAS there. A run that ends "nothing offered a move" names
        # the symptom and hides whether she read a page with no grid on it, a
        # grid too small to count, or nothing at all.
        logger.info(
            "%d down: read %d region(s), %dx%d with %d thing(s) — not a thing laid out yet",
            down,
            len(seen.get("layout") or ()),
            here.rows,
            here.columns,
            here.occupied(),
        )
        try:
            moved = await hands.scroll(dy=-_a_screenful(hands))
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("screen_pursuit", exc, action="scroll to find the thing")
            return down
        # A scroll that did not happen will not have moved the page, so the
        # next read is the same read and the wait before it buys nothing. The
        # receipt was discarded, so a refused scroll cost the full six
        # screenfuls and six settles — measured at 2.1 seconds a cycle with
        # nothing to show for it.
        if getattr(moved, "success", True) is False:
            logger.info(
                "the page did not scroll (%s); reading what is here instead",
                getattr(moved, "error", "") or "no reason given",
            )
            return down
        await asyncio.sleep(SETTLE_AFTER_SCROLL_S)
    return SCREENFULS_TO_LOOK


#: What one scroll goes if the screen cannot be measured. Small enough that
#: nothing is stepped over on any display anyone still uses.
A_SCREENFUL_AT_LEAST = 400


#: The share of a screen one scroll moves. Not the whole of it, so a thing
#: sitting across the fold is never skipped between two readings.
MOST_OF_A_SCREEN = 0.8


#: Long enough for a page to finish moving before it is read again.
SETTLE_AFTER_SCROLL_S = 0.35


def _a_screenful(hands: Any) -> int:
    """How far one scroll should go, from the screen she is actually looking at.

    A number picked in advance is wrong on every display but one. Most of a
    screen rather than all of it, so something sitting across the fold is not
    skipped between two readings.
    """
    measure = getattr(hands, "_main_screen_visible_frame", None)
    if not callable(measure):
        return A_SCREENFUL_AT_LEAST
    try:
        height = int(measure()[3])
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, IndexError):
        return A_SCREENFUL_AT_LEAST
    return max(A_SCREENFUL_AT_LEAST, int(height * MOST_OF_A_SCREEN))


def _worth_holding(found: Any, whole: Any, seen: dict[Any, int] | None = None) -> Any:
    """The block worth carrying to the next glance.

    A reading taken before the page settles has no lattice in it — nothing
    drawn yet, or a panel over the top — and what comes back is the whole
    reading rather than a block of it. Held as though it were a block, it
    forces every later reading back to the whole page, and she never finds the
    thing at all. LIVE 2026-08-29: "reading 13x8" for an entire run on a
    four-by-four board, every comparison after the first discarded as
    unreadable, and no rule ever formed.

    And the shape she carries is the one she has SETTLED on, not the one she
    saw a moment ago. Anchoring on the last glance means one bad glance drops
    the anchor and she begins again: live 2026-08-31 the readings went 3x4,
    then unreadable, then 3x4, and one move in three could be compared with
    another. A thing does not change shape, so the shape she has read most
    often is the better guess about it than the shape she read last — and a
    single misreading no longer costs her the thing.
    """
    if found is None or whole is None:
        return None
    inside = found.rows * found.columns < whole.rows * whole.columns
    if not inside:
        return None
    # `is not None`, because an empty tally is falsy and the first shape would
    # never be recorded — so it stayed empty, and nothing was ever settled.
    if seen is not None:
        seen[(found.rows, found.columns)] = seen.get((found.rows, found.columns), 0) + 1
        settled = max(seen, key=lambda shape: (seen[shape], shape[0] * shape[1]))
        if seen[settled] > 1 and (found.rows, found.columns) != settled:
            # This glance disagrees with what it has usually been. Keep the
            # shape rather than the glance; a short reading is placed inside a
            # known shape, where a differently-shaped one cannot be compared
            # at all.
            return _AS_IT_USUALLY_IS.get(settled) or found
        _AS_IT_USUALLY_IS[settled] = found
    return found


#: The last reading of each shape she has settled on, so a glance that
#: disagrees can be placed in one rather than replace it.
_AS_IT_USUALLY_IS: dict[Any, Any] = {}


def _the_rest_of_the_run(
    first: str,
    from_here: Any,
    expect: Callable[[Any, str], Any],
    names: Sequence[str],
    choose: Callable[[Any, Sequence[str]], str],
    how_many: int,
) -> tuple[list[str], Any]:
    """The acts after this one, chosen on the board she expects rather than seen.

    Where the model is trusted this is the whole saving. Reading the screen
    costs about half a second and the thing answers about a second later, so
    a move that is read, decided and confirmed costs over two — and a game of
    five hundred moves is nineteen minutes of watching, none of which is
    thinking. A person who knows how a board moves presses several keys and
    then looks.

    Hands back the acts and the board she expects to find at the end of them,
    which is what makes the run checkable: if it is not there, the run broke,
    and she says so rather than carrying on from a board she imagined.
    """
    rest: list[str] = []
    where = expect(from_here, first)
    for _again in range(max(0, how_many - 1)):
        if where is None:
            break
        got = choose(where, names)
        if not got:
            break
        after = expect(where, got)
        if after is None or after == where:
            break
        rest.append(got)
        where = after
    return rest, where


def _the_same_thing_without(reading: Any, cell: Any) -> Any:
    """The same arrangement with one thing taken out of it.

    Used to ask what a thing rests on: take a part away and see whether what
    she is holding survives without it.
    """
    from core.perception.what_is_there import Arrangement  # noqa: PLC0415

    return Arrangement(
        rows=reading.rows,
        columns=reading.columns,
        cells=tuple(one for one in reading.cells if one is not cell),
        down_at=getattr(reading, "down_at", ()),
        across_at=getattr(reading, "across_at", ()),
    )


def _expects(knows: Any) -> Callable[[Any, str], Any] | None:
    """Her rule for what an act would do, when she has one she trusts."""
    rules = getattr(knows, "rules", None)
    expect = getattr(rules, "expect", None)
    sure = getattr(rules, "confidence", None)
    if not callable(expect) or not callable(sure):
        return None
    try:
        trusted = float(sure() or 0.0)
    except (TypeError, ValueError):
        # not a failure: a confidence that is not a number is not one.
        return None
    return expect if trusted > 0.0 else None


def _how_much_the_tally_moved(
    moving: Any, before: Any, after: Any
) -> float | None:
    """How far the score she found on the screen went, across one act.

    None when she has not found one, which is not a failure — plenty of things
    keep no score, and she has other ways of telling whether a stretch went
    well. Where there IS one it is the honest measure, because it was put
    there to be exactly that and she did not have to be told.
    """
    from core.perception.where_it_responds import places_and_text

    tally = moving.what_measures_doing_well()
    if not tally or not isinstance(before, dict) or not isinstance(after, dict):
        return None
    was, now = places_and_text(before), places_and_text(after)

    def number(seen: dict[tuple[int, int], str], at: tuple[int, int]) -> float:
        try:
            return float(str(seen.get(at, "")).replace(",", "").strip() or 0.0)
        except (TypeError, ValueError):
            # not a failure: a place showing words is not showing a number.
            return 0.0

    return sum(max(0.0, number(now, at) - number(was, at)) for at in tally)


def _by_how_much_room(
    went: Sequence[tuple[Any, bool]],
    names: Sequence[str],
    expect: Callable[[Any, str], Any],
) -> list[tuple[Any, bool]]:
    """The same situations, judged by how much room each left her.

    Whether a move went well is the caller's to say, and sometimes what it
    says does not vary. Playing 2048 for three hundred moves, "the largest
    thing did not get smaller" was true of very nearly every move she made, so
    no property could tell the good states from the bad, so she committed to
    nothing and ruled nothing out for the whole game. A measure that is always
    true is not a measure.

    What she always has instead is how much of her own future she can still
    reach from here, which is what Klyubin and Polani's empowerment is for —
    the thing to want when nobody has said what to want. It varies by
    construction, because the split is at the middle of what she has actually
    seen rather than at a number chosen for it.
    """
    if not went or not names:
        return []

    def step(one: Any, act: str) -> Any:
        return _somewhere_else(one, act, expect)

    # As far ahead as it takes for the measure to say something.
    #
    # One move ahead, nearly every 2048 board reaches four different places,
    # so the measure was as flat as the one it was replacing. How crowded a
    # thing is shows up in where she can get to, not in what she can press,
    # and how far ahead that takes is a property of the thing rather than a
    # number to pick: deepen while every situation still looks alike, and stop
    # the moment they do not. Bounded by how many acts there are, because
    # past that the tree is wider than the differences it could show.
    room: list[int] = []
    for ahead in range(1, max(2, len(names)) + 1):
        room = [
            what_is_still_open(
                place, acts=names, step=step, named=repr, ahead=ahead
            ).hers
            for place, _ in went
        ]
        if len(set(room)) > 1:
            break
    if len(set(room)) < 2:
        return list(went)
    # The better half, by rank rather than by a level. A median split cannot
    # come out all one way, which is the failure it is here to avoid.
    ranked = sorted(range(len(room)), key=lambda one: room[one])
    better = set(ranked[len(ranked) // 2 :])
    return [
        (place, at in better) for at, (place, _) in enumerate(went)
    ]


def _somewhere_else(place: Any, act: str, expect: Callable[[Any, str], Any]) -> Any:
    """Where an act leaves things, or nothing when it leaves them as they were."""
    try:
        went_to = expect(place, act)
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
        return None
    if went_to is None or went_to == place:
        # not a failure: a move that changes nothing is not a move she has.
        return None
    return went_to


def _in_the_same_grid(lattice: Any, before: Any, after: Any) -> bool:
    """Whether both readings were laid into the grid she is holding.

    Before she has one, a reading works out its own — and a reading of a game
    that has not started yet puts the score, the best score, the title and the
    instructions in it, because at that moment they are the only things there
    to make a grid out of. Teaching the rule from those is teaching it that
    pressing down moves the New Game button.

    LIVE 2026-08-31, a hundred and fifty moves of the real game: every reading
    after the third was four by four and the rule still only matched three in
    four, because the first three were seven columns of furniture and every
    comparison that touched one of them failed. A rule learned across two
    different grids is not a rule about either.
    """
    if lattice is None or not getattr(lattice, "held", False):
        return False
    theirs = getattr(lattice, "down_at", ()), getattr(lattice, "across_at", ())
    return all(
        (getattr(one, "down_at", ()), getattr(one, "across_at", ())) == theirs
        for one in (before, after)
        if one is not None
    )


def _both_of_the_thing(app: str, before: Any, after: Any) -> bool:
    """Whether both readings really are of the thing she is working in.

    A reading that could not find the window is a reading of the WHOLE
    DESKTOP, and it comes back looking like any other reading — same shape,
    same fields, text and positions and all. Handed to the part of her that
    works out where the thing's places are, it says the places are wherever
    the dock and the menu bar and somebody's terminal happen to be.

    LIVE 2026-08-31, playing the real game: focus slipped partway through a
    hundred and thirteen moves, thirty-four readings were of the desktop, and
    what she had worked out to be a four by four board became six by eighteen.
    The rule she had already learned stopped matching, and nothing anywhere
    said anything had gone wrong, because every one of those readings was a
    perfectly good reading of something else.

    Losing the window for a moment is ordinary and recoverable. Learning from
    what was underneath it is not.
    """
    if not app:
        return True
    return all(
        str((one or {}).get("scoped_to") or "") == app and _was_of_that_window(one, app)
        for one in (before, after)
        if isinstance(one, dict)
    )


def _was_of_that_window(reading: Any, app: str) -> bool:
    """Whether the picture was taken while that application was in front.

    The rectangle was hers; the pixels in it belong to whatever was drawn
    there. A reading taken while something else was in front is a reading of
    that something else, wearing her window's coordinates.

    A reading from before this was recorded says nothing either way, and
    ``True`` is what "says nothing" has always meant here.
    """
    if not (reading or {}).get("her_window_showing", True):
        # Her window is exactly where it was and none of it is on the screen
        # that was photographed.
        return False
    who = str((reading or {}).get("in_front_then") or "")
    if not who:
        return True
    mine, theirs = app.strip().lower(), who.strip().lower()
    return mine in theirs or theirs in mine


def _moves_that_leave_her_nothing(
    reading: Any,
    names: Sequence[str],
    expect: Callable[[Any, str], Any],
    world: Any = None,
) -> tuple[str, ...]:
    """Moves after which she would have nothing left to do.

    How much of its own future a thing can still reach through its own acts is
    what Klyubin and Polani called empowerment, and the usual use of it is
    keeping options open. The floor of it is not a preference: in a game that
    ends when nothing can move, a move that leaves her nothing is the losing
    move, whatever else can be said for it.

    Counted as distinct places she could bring about rather than as moves she
    could make, because several moves that all leave the world the same are one
    option and not several.

    Her move alone is the wrong thing to look at, and looking at it alone made
    this unable to fire at all. A board like 2048 fills up when the WORLD puts
    something down, not when she slides; a slide either merges and leaves a
    gap or changes nothing, so no move of hers has a full board on the other
    side of it and no move of hers ever looked fatal. What she has to survive
    is her move and then the world's, so the world's turn is taken here when
    she has worked out what it does.

    Only ruled out when she has nothing left WHATEVER the world does. Where
    some of what the world might do leaves her stuck and some does not, that is
    a risk and not a certainty, and refusing on a risk is how a thing talks
    itself out of every move it has.
    """
    if not names:
        return ()

    def step(place: Any, act: str) -> Any:
        return _somewhere_else(place, act, expect)

    def then_the_world(place: Any) -> tuple[Any, ...]:
        might = getattr(world, "might_do", None)
        if not callable(might):
            return (place,)
        try:
            ways = might(place)
        except (AttributeError, TypeError, ValueError):
            # not a failure: a world she cannot ask is a world she takes as
            # leaving things where they are.
            return (place,)
        return tuple(way for way, _share in ways) or (place,)

    dead: list[str] = []
    for act in names:
        after = step(reading, act)
        if after is None:
            continue
        if all(
            what_is_still_open(
                then, acts=list(names), step=step, named=repr
            ).hers
            == 0
            for then in then_the_world(after)
        ):
            dead.append(act)
    return tuple(dead)


def _within_a_move(
    wanted: Callable[[Any], bool],
    reading: Any,
    names: Sequence[str],
    expect: Callable[[Any, str], Any],
) -> bool:
    """Whether what she wants is true here, or one move from here."""
    try:
        if wanted(reading):
            return True
    except (ArithmeticError, AttributeError, TypeError, ValueError):
        return False
    for act in names:
        try:
            went_to = expect(reading, act)
        except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
            continue
        if went_to is None:
            continue
        try:
            if wanted(went_to):
                return True
        except (ArithmeticError, AttributeError, TypeError, ValueError):
            continue
    return False


def _a_step_back(
    wanted: Any,
    went: Sequence[tuple[Any, bool]],
    reading: Any,
    names: Sequence[str],
    expect: Callable[[Any, str], Any],
) -> Any:
    """What would have to be true for the thing she wants to be one move away.

    Her own rule is what says whether something is near: a want is in reach of
    a situation when one of her moves from there makes it true. One move and
    not several, because the walking back supplies the depth — each thing it
    finds is a move from the next, so a chain of three is three moves deep
    without any of them being searched for together.
    """

    def in_reach(place: Any, want: Callable[[Any], bool]) -> bool:
        return _within_a_move(want, place, names, expect)

    way = a_way_to_get_there(
        wanted.holds,
        reading,
        somewhere_like=[place for place, _ in went],
        in_reach=in_reach,
        called=wanted.name,
    )
    first = way.want_first
    if first is None or first.name == wanted.name:
        return None
    logger.info("cannot hold that from here, so first: %s", first.name)
    return first


def _moves_she_will_not_make(
    she_keeps: dict[str, Any],
    went: Sequence[tuple[Any, bool]],
    reading: Any,
    names: Sequence[str],
    knows: Any,
    turn: int,
    world: Any = None,
) -> tuple[frozenset[str], str]:
    """The moves she has already ruled out, before any looking ahead.

    Two recordings of somebody playing well. Clearing 2048 in 989 moves, two
    of the four directions evict the corner and they pressed neither, all
    game. Beating a hard checkers engine three pieces down, every move that
    would have broken their back row was never a candidate. The difference was
    not that they searched further. Most of what a search considers, they were
    not considering at all.

    She needs something to hold before she can rule anything out, and it has to
    be earned: at least one move watched for each move she could make, because
    below that a property that separates the good from the bad has not been
    given the chance to fail. That floor comes from the size of the choice
    rather than from a number picked for it.

    Only moves the rules can actually predict are ruled out. Anything she
    cannot foresee the result of survives, which is why the way out and the
    ways of asking are never removed by this.
    """
    if reading is None or not names or len(went) < len(names):
        return frozenset(), ""
    rules = getattr(knows, "rules", None)
    expect = getattr(rules, "expect", None)
    if not callable(expect):
        return frozenset(), ""
    sure = getattr(rules, "confidence", None)
    if not callable(sure) or float(sure() or 0.0) <= 0.0:
        return frozenset(), ""
    if turn != she_keeps.get("at"):
        kept, why = what_to_hold_now(she_keeps.get("it"), went)
        if kept is None:
            kept, why = what_to_hold_now(
                she_keeps.get("it"), _by_how_much_room(went, list(names), expect)
            )
        she_keeps.update({"it": kept, "why": why, "at": turn})
    it = she_keeps.get("it")
    if it is None:
        return frozenset(), ""
    if not _within_a_move(it.holds, reading, list(names), expect):
        # It is not true here and no move makes it true, so it cannot be held
        # from where she is standing. Holding it anyway rules nothing out and
        # steers nothing; it is a want with no purchase on the next move.
        #
        # This is where somebody clearing 2048 stopped wanting the far thing.
        # Twenty moves from the tile they were playing for, no looking ahead
        # reaches it, and they did not try. They wanted two of the thing below
        # it instead, and every merge afterwards built the next merge's
        # precondition. So walk back from what she is holding to what would
        # put it within reach, and hold that.
        nearer = _a_step_back(it, went, reading, list(names), expect)
        if nearer is None:
            return frozenset(), ""
        she_keeps["it"] = it = nearer
    keeps, breaks = what_it_rules_out(it, reading, list(names), expect=expect)
    dead = _moves_that_leave_her_nothing(reading, list(names), expect, world)
    if dead and len(dead) < len(names):
        # A move after which she has no move is how a game like this is lost,
        # and no property she happens to be holding makes one worth making.
        # Ruled out on its own account, before the rest of the weighing.
        breaks = tuple(dict.fromkeys([*breaks, *dead]))
        keeps = tuple(one for one in keeps if one not in dead)
    if not keeps or not breaks:
        # Nothing to rule out, or everything — and refusing every move is not
        # holding something, it is being stuck.
        return frozenset(), ""
    return (
        frozenset(breaks),
        f"holding that {it.name}, so not {', '.join(sorted(breaks))}",
    )


def _the_biggest_thing_on_it(reading: Any, reporting: Sequence[tuple[int, int]] = ()) -> float:
    """The largest number among the things she is acting on.

    Without the places that only report. A score sitting inside the thing's
    own outline is larger than anything on the board almost at once, and it
    is not something she made.
    """
    leave_out = set(reporting or ())
    found = [
        float(cell.says.replace(",", ""))
        for cell in getattr(reading, "cells", ()) or ()
        if (cell.row, cell.column) not in leave_out
        and str(cell.says).replace(",", "").replace(".", "", 1).isdigit()
    ]
    return max(found, default=0.0)


def _she_got_further(made: float, furthest: float) -> str:
    """What to say when she has just built the biggest thing she has here.

    Said only when it passes what she brought in, so a second game says
    nothing until it is doing better than the first — which is what "furthest
    she has got" means and what somebody watching wants to hear.
    """
    if made <= furthest:
        return ""
    if furthest <= 0.0:
        return f"I have a {made:g} on the board."
    return f"A {made:g} — the biggest I have made here. The best before was {furthest:g}."


def _what_there_is_to_aim_at(reading: Any) -> str:
    """What to prefer one situation over another by, when nobody said.

    A request can name a process without naming a finish — "play it and work
    out how it moves" — and then there is nothing to score a future against,
    so she acts and looks for as long as the budget lasts and never uses the
    model she is building. That is a waste of the thing she just worked out.

    What she can read off the world instead is whether it counts, and how far
    it could go. A laid-out thing that combines equal pairs cannot exceed one
    doubling per place it has: sixteen places cannot hold more than two to the
    sixteenth however well it is played. That ceiling is a fact about the thing
    in front of her rather than a number anybody picked, and it is far enough
    above where she is that being nearer to it stays worth something all the
    way through — which a nearer goal does not, because arriving at one makes
    every situation after it look equally good.

    Measured 2026-08-27, played the way the loop plays it — the goal put
    through the same gate, six games each, run to a dead board:

        said "the largest"          median 128, and not one look ahead
        read the ceiling off it     median 768, max 1024, 666 looks ahead

    The first is random. Not because the words are wrong but because nothing
    downstream can use them: worth_comparing refuses a goal it cannot measure,
    the search never runs, and every move is a coin toss on a board she can
    read perfectly well.

    Where the things in front of her are not numbers, nothing here invents a
    purpose: it says so, and she goes back to acting and looking.
    """
    numbers = getattr(reading, "numbers", None)
    places = getattr(reading, "places", None)
    if not callable(numbers) or not callable(places):
        return ""
    if not numbers():
        return ""
    room = int(places() or 0)
    if room <= 0:
        return ""
    return f"{2 ** room}"


def _left_her_better_off(
    before: Any, after: Any, toward: str, approach: str
) -> bool:
    """Whether the move improved the situation, by the measure she is using.

    The same measure that ranks futures ranks what actually happened, so what
    she gets good at is what her own judgement says was worth doing rather
    than a separate opinion about it.
    """
    try:
        from core.agency.how_good_is_this import how_good

        was = how_good(before, toward=toward, approach=approach)
        now = how_good(after, toward=toward, approach=approach)
    except (ImportError, AttributeError, TypeError, ValueError) as why:
        logger.debug("could not weigh whether that left her better off: %s", why)
        return False
    return now >= was


def _what_she_is_not_reading(rules: Any) -> str:
    """Whether her own record proves a quantity she cannot see.

    "How this moves is not worked out yet" is true of two different worlds and
    says nothing about which. In one, a rule is there and she has not found it,
    and more moves are the answer. In the other, the same board and the same
    key came out two ways, so no rule reading only the board can ever fit, and
    more moves are looking where the answer cannot be.

    Watching is perception's; reading what the watching proves is not, so the
    record leaves the model whole and the question is asked here.
    """
    from core.cognition.something_she_cannot_see import what_she_cannot_see

    try:
        record = rules.what_she_saw_happen()
    except (AttributeError, TypeError):
        return ""
    if len(record) < 2:
        return ""
    found = what_she_cannot_see(
        [((seen, did), then) for seen, did, then in record]
    )
    if not found.anything:
        return ""
    if found.she_can_compute_it:
        return (
            f" — and one thing she was not reading, which runs every "
            f"{found.every} moves, so she can"
        )
    return (
        f" — and one thing she was not reading, taking {found.how_many} values "
        "in no cycle, so the world puts something there she does not control"
    )


def _say_what_she_worked_out(knows: Any, said_already: dict[str, bool]) -> None:
    """Say it the once, when she first works out how a thing moves."""
    from .screen_pursuit import (
        _tell,
    )

    rules = getattr(knows, "rules", None)
    if rules is None or said_already.get("said"):
        return
    if getattr(rules, "rule", lambda: None)() is None:
        return
    said_already["said"] = True
    logger.info("she can see ahead now: %s", rules.says())
    _tell(f"I can see what my moves do here now — {rules.says()}.")


def am_i_there(wanted: str, reading: str, page: str, window: str) -> bool:
    """Whether this is the thing she was asked to act in.

    She was asked to find something and act in it, and nothing anywhere
    checked that she had: a reading is a reading, and anything with text laid
    out in rows reads as something she could push. LIVE 2026-08-26, another
    part of her closed the browser mid-run and she played twelve moves of 2048
    into a chat window, narrating every one of them.

    Identity where there is identity — an address, a title, the name of the
    window — and the reading itself where there is not. What is not accepted
    is silence: a name she was given and cannot find anywhere is a name she
    has not arrived at.
    """
    name = " ".join(str(wanted or "").split()).lower()
    if not name:
        return True
    said = [word for word in re.split(r"[^a-z0-9]+", name) if len(word) > 2]
    if not said:
        return True
    # Identity first, and the reading only when there is no identity.
    #
    # A screen reading is of the screen, not of her window, so anything else
    # visible counts as evidence that she has arrived. LIVE 2026-08-26: the
    # word she was looking for was in a terminal on the same display, the test
    # passed, and she played into that instead. What identifies a thing — an
    # address, a title, the name of the window — cannot be borrowed from
    # somebody else's window the way words on a screen can.
    known = " ".join((str(page or ""), str(window or ""))).lower().strip()
    if known:
        return any(word in known for word in said)
    return any(word in str(reading or "").lower() for word in said)


async def _take_the_run_its_bearings(
    anchor: dict[str, str],
    *,
    expect_page: str = "",
    open_page: str = "",
    target_app: str = "",
) -> None:
    """Work out which page and which window this run belongs to.

    Done before anything depends on the answer, because everything does: what
    is brought forward, what a keystroke is bound to, and whether she is in
    the thing she was asked to act in at all.

    A run that names an application is about that application, and one that
    names a page is about that page. A browser having something open is not a
    fact about the task: the test for "is this about a page" included whether
    any page was open anywhere, which is true whenever a browser is running.

    LIVE 2026-09-04, driving a desktop game: "this run belongs to '2048 Game'
    on 'https://x.com/home'". Every cycle then checked that somebody's
    timeline was still in front, and brought the browser forward over the
    window it was driving to put it back.
    """
    from .screen_pursuit import (
        _frontmost,
        current_page_identity,
    )

    # A run in a BROWSER is about a page whether or not the caller named one:
    # the page is the thing being acted in, and the browser is only the window
    # around it. Excluding it along with desktop applications left such a run
    # with no anchor at all, so drift could not be detected — which is the
    # thing this function exists to make possible.
    about_a_page = bool(open_page or expect_page or names_any(target_app, BROWSERS))
    page = await current_page_identity() if (about_a_page or not target_app) else {}
    if not anchor["page"] and about_a_page:
        anchor["page"] = str(
            expect_page or page.get("url") or page.get("title") or ""
        ).strip()
    if not anchor["app"]:
        # The application that holds the page, when this run is about a page
        # at all. A task about a desktop application would otherwise anchor
        # itself to a browser that happens to be open behind it.
        holder = str(page.get("app") or "") if about_a_page else ""
        anchor["app"] = (
            str(target_app or "").strip() or holder or await _frontmost() or ""
        ).strip()
    if anchor["app"]:
        logger.info(
            "this run belongs to %r on %r", anchor["app"], anchor["page"][:60]
        )
