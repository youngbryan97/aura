"""Looking several moves ahead, as far as there is time to look.

Once she can try a move without making it and say which result is better, the
two compose: try a move, then try every move from there, and keep going while
it is worth the arithmetic. What comes back is not a move but an ordering over
the moves available, with the reason the best one is best.

How deep is not a setting. Each level costs branching times the last, and the
cost of one level is measured rather than assumed, so the depth is whatever
fits the time this decision is worth. That is the whole of metareasoning at
this scale: work out what thinking costs, work out what is available, spend
the second on the first.

A world that adds something of its own after every act — a dealt tile, another
person, a page that refreshes — is averaged over at every level rather than
chosen for, and a move is judged by what the situation comes to at the far end
of the search.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from core.agency.how_good_is_this import how_good, why

__all__ = [
    "forget_how_far_she_saw",
    "how_far_she_can_see",
    "how_deep_to_look",
    "look_ahead",
    "worth_finding_out",
]

logger = logging.getLogger("Aura.LookingAhead")

#: How far ahead the arithmetic still describes something that could happen
#: is not a number anybody picks — it is how much of its own the world will
#: have added by then, against the room the thing has left. See
#: :func:`as_far_as_the_world_lets_her`, which measures it.

#: What one level of looking is assumed to cost before any has been measured.
#: Deliberately generous: being slow to look deep costs a little, and looking
#: deeper than there is time for costs the move.
_UNMEASURED_LEVEL_S = 0.02

#: What a level of looking has actually cost, measured.
_A_LEVEL: dict[str, float] = {"seconds": 0.0}

#: How many of her own acts ahead the last search saw, whichever way it ran.
_SAW: dict[str, int] = {"acts": 0}


def forget_how_far_she_saw() -> None:
    """A new run starts without having looked."""
    _SAW["acts"] = 0


def how_far_she_can_see() -> int:
    """How many of her own acts ahead her last look reached. Nought before any.

    A question other parts of her decision ask about themselves. Ruling moves
    out before looking is worth it when looking is short-sighted; once she can
    see what a move leads to, a guess about which moves deserve a look costs
    more than the look.
    """
    return int(_SAW["acts"])


class _AlreadyWorkedOut:
    """What this search has already worked out, so it is not worked out twice.

    A search over a thing reaches the same situation by many routes — four
    moves and the world's replies fan out and fold back on each other
    constantly — and every arrival used to pay again for the same three
    things: what an act makes of a situation, what the situation is worth,
    and what the best line from it comes to.

    Kept for one search and thrown away with it, because all three answers
    are about the rule and the measure as they stand at this moment, and both
    can change between one move and the next.
    """

    __slots__ = ("becomes", "worth", "onward", "hits")

    def __init__(self) -> None:
        self.becomes: dict[tuple[str, str], Any] = {}
        self.worth: dict[str, float] = {}
        self.onward: dict[tuple[str, int], float] = {}
        self.hits = 0

    def what_it_becomes(self, expect: Any, state: Any, action: str) -> Any:
        key = (_reading(state), action)
        if key in self.becomes:
            self.hits += 1
            return self.becomes[key]
        made = expect(state, action)
        self.becomes[key] = made
        return made

    def what_it_is_worth(self, state: Any, **how: Any) -> float:
        key = _reading(state)
        if key in self.worth:
            self.hits += 1
            return self.worth[key]
        said = how_good(state, **how)
        self.worth[key] = said
        return said


def how_deep_to_look(
    available: int, budget_s: float, branching: int = 4, no_further_than: int = 0
) -> int:
    """How far ahead there is time to look, from what a level has cost.

    Each level multiplies the work by the branching. The answer is the deepest
    level whose cost still fits, and one is always affordable — a single level
    is the difference between choosing blind and choosing at all.

    ``no_further_than`` is where the arithmetic stops describing anything that
    could happen, which is a fact about the world rather than about the clock.
    Nought is no limit.
    """
    if available < 1 or budget_s <= 0.0:
        return 1
    a_level = _A_LEVEL["seconds"] or _UNMEASURED_LEVEL_S
    depth = 1
    spent = a_level * branching
    while no_further_than <= 0 or depth < no_further_than:
        spent = spent * branching
        if spent > budget_s:
            break
        depth += 1
    return depth


def worth_finding_out(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    ahead: dict[str, tuple[float, str]] | None = None,
    never_tried: Sequence[str] = (),
) -> dict[str, float]:
    """What each act is worth for what it would TELL her, not where it leads.

    Looking ahead asks which move is best under the rule she is using. This
    asks a different question: which move would settle which rule is right.
    They are not the same move, and early on the second is worth far more —
    a rule she is sure of improves every move after this one, and a slightly
    better position improves only this one.

    Scaled by what knowing is worth here, which is read off the futures she
    can already see: where the best and worst moves available differ by very
    little, being right about the rule is worth very little, and where they
    differ by a lot it is worth a lot. So the number comes from her own
    situation rather than from a setting, and it goes to nought by itself as
    the rule settles — at which point she stops experimenting, because there
    is nothing left to find out.

    Where she cannot see ahead at all, this is what she has: the acts are
    scored purely by what they would settle, which is the right thing to do
    when she has no model to prefer anything by.

    ``never_tried`` is the acts she has not taken here yet, and it comes
    first, because everything above needs a state she can read and rules that
    disagree about it — and in a world she has just arrived in she has
    neither. What her own acts do is the first thing there is to find out,
    and it needs no model, no grid and no reading. It empties itself once
    they have all been taken.

    LIVE 2026-09-04: no grid, so no rule, so nothing disagreed about
    anything, so this returned nothing, so there was no reason to vary — and
    a grid is worked out from what moves, which under one act is wherever
    that act puts things. Two hundred seconds of the same key.
    """
    fresh = [str(one) for one in never_tried if str(one) in {str(a) for a in actions}]
    if fresh:
        # A tie between all of them on the first act, which is right: any of
        # them settles as much as any other when none has been taken.
        return {one: 1.0 for one in fresh}
    settle = getattr(knows, "what_this_would_settle", None)
    if not callable(settle) or state is None:
        return {}
    told: dict[str, float] = {}
    for action in actions:
        try:
            told[action] = max(0.0, float(settle(state, action)))
        except (AttributeError, TypeError, ValueError):
            continue
    if not told or not any(told.values()):
        return {}
    values = [value for value, _ in (ahead or {}).values()]
    # What being right is worth here. With nothing to see ahead, finding out
    # is the only thing on offer and is worth one whole move.
    spread = (max(values) - min(values)) if len(values) > 1 else 1.0
    if spread <= 0.0:
        return {}
    return {action: share * spread for action, share in told.items()}


def at_the_worlds_mercy(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    *,
    toward: str = "",
    approach: str = "",
    weights: Any = None,
    world: Any = None,
) -> dict[str, float]:
    """For each act, how much of what happens next is the world's to decide.

    Looking ahead averages over what the world might do, which is right for
    working out what a move is worth and throws away the thing a person plays
    for. A position where every reply leaves her fine is not the same as one
    where the average reply leaves her fine and one of them ruins her, and the
    average cannot tell them apart.

    This is what a grip is. A hand on a wrist does not improve the position by
    itself and it is not free — it occupies a hand. What it buys is that
    whatever the other side does next, she is still where she was. Keeping the
    largest thing in a corner buys the same: wherever the world puts something
    next, the structure survives. Every account of controlling a position is
    this quantity, and none of them is about the average.

    Said as a share rather than in points, so it can be weighed beside things
    measured in other units: how far the world's replies spread, against how
    far her own choices spread. One is what happens to her and the other is
    what she does, and the ratio is how much of this position is hers.
    """
    might = getattr(world, "might_do", None)
    expect = getattr(knows, "expect", None)
    if not callable(might) or not callable(expect):
        return {}
    how = {"toward": toward, "approach": approach, "weights": weights}
    spreads: dict[str, float] = {}
    mine: list[float] = []
    for action in actions:
        try:
            future = expect(state, action)
        except (AttributeError, TypeError, ValueError):
            continue
        if future is None:
            continue
        mine.append(how_good(future, **how))
        try:
            ways = might(future)
        except (AttributeError, TypeError, ValueError):
            ways = ()
        seen = [how_good(way, **how) for way, _share in ways or ()]
        spreads[action] = (max(seen) - min(seen)) if len(seen) > 1 else 0.0
    if not spreads or not any(spreads.values()):
        return {}
    # Against how much her own choice moves things, so the number says how
    # much of what happens is the world's doing rather than hers.
    hers = (max(mine) - min(mine)) if len(mine) > 1 else 0.0
    return {
        action: spread / (spread + hers) if (spread + hers) > 0 else 0.0
        for action, spread in spreads.items()
    }


def whether_to_take_the_wide_option(
    left: float, gaining: float, *, against_a_clock: bool = True
) -> float:
    """Which way to lean on a position the world could swing, from -1 to 1.

    A fighter takes a shot at nineteen seconds that he would not take at
    three minutes, and he takes it when he is behind and not when he is
    ahead. Neither is recklessness or caution as a temperament: the value of
    an uncertain outcome depends on how much time is left to recover from it
    and on whether the sure thing is winning.

    Both come from the run rather than from a setting. ``left`` is the share
    of the budget still to go, and ``gaining`` is what she has been getting
    per act — positive when the way she is playing is working.

    Ahead with time to spare, an uncertain position is worth avoiding: she
    only needs to keep doing what works. Behind with the clock going, it is
    worth seeking, because the average outcome of what she is doing is
    already a loss and the spread is the only thing that contains a win.
    """
    # Only against a clock. A goal that names its own end is kept at until it
    # is met, and its budget is set far past what it needs, so no deadline is
    # being raced and the trade of what a move is worth against how widely it
    # could land has no reason to be made either way. Taken anyway, the whole
    # budget read as time to spare and she leaned fully away from every swing
    # all game: played out in her own model with her live search, 3 of 6
    # games reached 2048 leaning at -0.99 and 6 of 6 without (2026-09-24).
    if not against_a_clock:
        return 0.0
    share = max(0.0, min(1.0, float(left)))
    # Losing counts for more the less time is left to recover from it: the
    # same deficit is a reason to steady early and a reason to gamble late.
    urgency = 1.0 - share
    if gaining > 0:
        return -share
    if gaining < 0:
        return urgency
    return 0.0


def look_ahead(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    *,
    toward: str = "",
    approach: str = "",
    budget_s: float = 0.5,
    world: Any = None,
    weights: Any = None,
    depth: int = 0,
    settles_how_far: bool = True,
    no_deeper_than: int = 0,
) -> dict[str, tuple[float, str]]:
    """Every move available, scored by where it leads and how sure that is.

    ``settles_how_far`` is False for a quick look taken inside another
    decision, whose shallow depth says nothing about how far she can see.

    ``knows`` is anything that can say what a state would become — the rules
    she worked out by watching. When it cannot, this returns nothing, which is
    the honest answer and not a failure.

    ``no_deeper_than`` is how far her model of this world has been measured to
    carry. Zero means nothing has been measured and the clock is the only
    bound; anything else stops the search where her predictions stop beating
    "nothing changed", because past that a level is fiction.

    ``world`` is what the world does on its own between her acts, if she has
    worked that out. Without it, the search takes the best continuation at
    every level — which quietly assumes the world will cooperate, and plans a
    future that cannot happen. With it, each level averages over what the
    world might do instead, which is the difference between a plan and a wish.
    """
    if not actions or state is None or knows is None:
        return {}
    expect = getattr(knows, "expect", None)
    if not callable(expect):
        return {}
    trust = float(getattr(knows, "confidence", lambda: 0.0)() or 0.0)
    if trust <= 0.0:
        return {}
    # On a compiled world when the rule allows one: the same judgement, fast
    # enough that the clock buys the depth it is asking for.
    fast = _through_a_compiled_world(
        knows, state, actions,
        toward=toward, approach=approach, budget_s=budget_s,
        world=world, weights=weights, depth=depth, settles_how_far=settles_how_far,
        no_deeper_than=no_deeper_than,
    )
    if fast is not None:
        return fast

    # NOT bound to what the superlative names right now. Measured, and it
    # costs half her play. See `bound_to` for the reasoning and the numbers
    # that refute it: a superlative in a line is meant to float.

    started = time.monotonic()
    fixed_depth = bool(depth)
    # What a level really costs is her acts times the world's replies to each
    # of them. Counting only her own acts understated it by the whole of the
    # world's fan-out, so the depth the clock allowed was many times what the
    # clock could afford.
    depth = depth or how_deep_to_look(
        len(actions),
        budget_s,
        branching=max(2, len(actions)) * max(1, _how_many_ways(world, state)),
    )
    known = _AlreadyWorkedOut()
    here_now = _reading(state)

    def dead_end() -> float:
        weighed = weights if weights is not None else _default_weights()
        return -(1.0 + sum(abs(float(value)) for value in weighed.values()))

    def one_pass(how_far: int) -> dict[str, tuple[float, str]]:
        found: dict[str, tuple[float, str]] = {}
        for action in actions:
            future = known.what_it_becomes(expect, state, action)
            if future is None or _reading(future) == here_now:
                # A move that would change nothing has not gone anywhere.
                #
                # Scored like any other, it collects the value of the
                # situation it left alone — so standing still outscores every
                # move that costs something to make. Measured against a null
                # on 2026-08-26: choosing this way was WORSE than choosing at
                # random, with 78% of moves doing nothing.
                #
                # Ruling one out before making it is the whole point of being
                # able to try a move without making it.
                continue
            value = _what_it_leads_to(
                expect, future, actions, how_far,
                toward=toward, approach=approach, world=world,
                weights=weights, known=known, knows=knows,
                been=frozenset({here_now}), dead=dead_end(),
            )
            found[action] = (value, why(future, toward=toward, approach=approach))
        return found

    # Deeper while the clock allows, rather than a guess at how deep it will
    # allow.
    #
    # What a level costs is a projection from an average, and it is wrong by
    # whatever the world's fan-out and what has already been worked out do to
    # it — the two together were a factor of tens. Going one level deeper and
    # stopping when the time is gone is measured by construction, and the
    # deeper pass is nearly free because the shallower one is still
    # remembered. What she hands back is always a completed pass.
    scored = one_pass(depth)
    if not fixed_depth and scored:
        ends_at = started + max(0.0, budget_s)
        a_pass = time.monotonic() - started
        # No ceiling but the clock. There used to be one — the room left, over
        # how often the world adds something — on the reasoning that past it
        # the arithmetic describes a board that cannot exist. It does not: a
        # merge makes room, and the world's replies at every level are worked
        # out from the room there actually is. What the ceiling did was stop
        # the search at one level on exactly the crowded boards where a second
        # level decides whether the game goes on. Measured 2026-09-17: one
        # level on a board with one place free, whatever the budget.
        while a_pass > 0.0:
            # Only a level there is time to FINISH. A pass abandoned halfway
            # is a pass that cost the budget and answered nothing.
            branching = max(2, len(actions)) * max(1, _how_many_ways(world, state))
            if time.monotonic() + a_pass * branching > ends_at:
                break
            deeper_at = time.monotonic()
            deeper = one_pass(depth + 1)
            if not deeper:
                break
            depth += 1
            scored = deeper
            a_pass = time.monotonic() - deeper_at

    spent = time.monotonic() - started
    if settles_how_far:
        _SAW["acts"] = int(depth) if scored else 0
    if scored and depth and not fixed_depth:
        _a_level_took(spent / float(depth))
    logger.debug(
        "looked %d ahead over %d move(s) in %.3fs (%d already worked out)",
        depth, len(actions), spent, known.hits,
    )
    return scored


def _how_many_ways(world: Any, state: Any) -> int:
    """How many replies the world has to one act, as it stands here."""
    might = getattr(world, "might_do", None)
    if not callable(might):
        return 1
    try:
        return max(1, len(might(state)))
    except (AttributeError, TypeError, ValueError):
        return 1


def _default_weights() -> dict[str, float]:
    from core.agency.how_good_is_this import AS_GOOD_A_GUESS_AS_ANY  # noqa: PLC0415

    return AS_GOOD_A_GUESS_AS_ANY


def _what_it_leads_to(
    expect: Any,
    state: Any,
    actions: Sequence[str],
    depth: int,
    *,
    toward: str,
    approach: str,
    world: Any = None,
    weights: Any = None,
    known: Any = None,
    knows: Any = None,
    been: frozenset[str] = frozenset(),
    dead: float = -1.0,
) -> float:
    """What a situation her act has just made comes to, ``depth`` of her acts on.

    Judged at the far end only. With one act left, it is what the situation
    is worth. With more, it is her best from each way the world might answer,
    averaged by how often each way happens — her move is hers to pick, the
    world's is not.

    Adding each level's worth to the level below counted a situation that
    looks good early once for every level it was passed through, which
    rewards looking good over ending well. Measured 2026-09-17 on a sliding
    board with the same terms and weights: judged at the far end, 2048 in
    fifteen games of sixteen; with each level added in, three of six.
    """
    if depth <= 1:
        return (
            known.what_it_is_worth(
                state, toward=toward, approach=approach, weights=weights,
                knows=knows, acts=actions,
            )
            if known is not None
            else how_good(
                state, toward=toward, approach=approach, weights=weights,
                knows=knows, acts=actions,
            )
        )
    ways = ()
    might = getattr(world, "might_do", None)
    if callable(might):
        try:
            ways = might(state)
        except (AttributeError, TypeError, ValueError):
            ways = ()
    if not ways:
        return _best_from(
            expect, state, actions, depth - 1,
            toward=toward, approach=approach, world=world, weights=weights,
            known=known, knows=knows, been=been, dead=dead,
        )
    return sum(
        share
        * _best_from(
            expect, way, actions, depth - 1,
            toward=toward, approach=approach, world=world, weights=weights,
            known=known, knows=knows, been=been, dead=dead,
        )
        for way, share in ways
    )


def _best_from(
    expect: Any,
    state: Any,
    actions: Sequence[str],
    depth: int,
    *,
    toward: str,
    approach: str,
    world: Any = None,
    weights: Any = None,
    known: Any = None,
    knows: Any = None,
    been: frozenset[str] = frozenset(),
    dead: float = -1.0,
) -> float:
    """The best this could still come to, with ``depth`` of her acts to make.

    ``been`` is the line already walked to get here. A future already on it
    is not reached: going back somewhere is not progress, and a search that
    scores it as progress prefers pacing to arriving.

    That could not happen in the world this was written for, where every move
    is irreversible, so it was never exposed there. In a world where a move
    can be undone it is severe. Measured on a sealed world with a reading
    that rises towards the goal: she climbed the reading correctly to a
    ridge, and then stepped back and forth between the same two squares for
    the rest of the budget, because the line that stepped back could step
    forward again and collect the higher reading a second time. Two squares,
    eighty moves, a perfectly correct model of the world, and nought arrivals.

    A situation where no act changes anything is worth ``dead``, below
    anything a live situation can be worth. One whose only acts go back along
    the line already walked is worth what it is worth: the line ends there.
    """
    here_now = _reading(state)
    if known is not None:
        # The same situation, the same distance from the end, is the same
        # answer — for the same line. A search folds back on itself constantly
        # and this is most of what it costs. The line is part of the key
        # because the value of a state depends on what is now behind her.
        remembered = known.onward.get((here_now, depth, been))
        if remembered is not None:
            known.hits += 1
            return remembered
    walked = been | {here_now}
    best: float | None = None
    somewhere_to_go = False
    for action in actions:
        future = (
            known.what_it_becomes(expect, state, action)
            if known is not None
            else expect(state, action)
        )
        if future is None or _reading(future) == here_now:
            continue
        somewhere_to_go = True
        if _reading(future) in walked:
            continue
        value = _what_it_leads_to(
            expect, future, actions, depth,
            toward=toward, approach=approach, world=world, weights=weights,
            known=known, knows=knows, been=walked, dead=dead,
        )
        best = value if best is None or value > best else best
    if best is not None:
        found = best
    elif somewhere_to_go:
        # Every act from here goes back along the line she came by. The line
        # ends here rather than being closed off, and it comes to what this
        # situation is worth.
        found = _what_it_leads_to(
            expect, state, actions, 1,
            toward=toward, approach=approach, world=world, weights=weights,
            known=known, knows=knows, been=walked, dead=dead,
        )
    else:
        found = dead
    if known is not None:
        known.onward[(here_now, depth, been)] = found
    return found


def _through_a_compiled_world(
    knows: Any,
    state: Any,
    actions: Sequence[str],
    *,
    toward: str,
    approach: str,
    budget_s: float,
    world: Any,
    weights: Any,
    depth: int,
    settles_how_far: bool = True,
    no_deeper_than: int = 0,
) -> dict[str, tuple[float, str]] | None:
    """The same search on a compiled world, or None when the world cannot be one.

    Her own terms and weights judge every situation. The terms that need the
    whole arrangement — whether her stated line still holds, anything she
    invented — are asked of it, once per situation the search meets.
    """
    from core.agency.a_world_compiled import compiled, search  # noqa: PLC0415
    from core.agency.how_good_is_this import INVENTED, a_line_judge  # noqa: PLC0415

    made = compiled(knows, world, state, actions)
    if made is None:
        return None
    weighed = weights if weights is not None else _default_weights()
    wants_line = bool(str(approach or "").strip()) and bool(float(weighed.get("line", 0.0) or 0.0))
    line_holds = a_line_judge(approach) if wants_line else None
    # Her stated line decides between the moves the search cannot tell apart,
    # and does not outweigh what the search can see. A line is a claim about
    # what lies past the horizon; on a world she can search several moves
    # deep, most of that is in front of her. Scored at full weight inside the
    # search it pulled every situation toward the line whatever came of it —
    # measured 2026-09-17 through her whole loop on the same seed, a line
    # about a corner took her from a 1024 in one game to three games lost and
    # nothing past 1024. So the search judges without it, and the line then
    # orders the moves that finished level with the best.
    invented = dict(INVENTED)
    shapes: dict[tuple[int, ...], Any] = {}
    values: dict[tuple[int, ...], float] = {}

    def as_arrangement(board: tuple[int, ...]) -> Any:
        found = shapes.get(board)
        if found is None:
            found = made.arrangement(board, like=state)
            shapes[board] = found
        return found

    def worth(board: tuple[int, ...]) -> float:
        known = values.get(board)
        if known is not None:
            return known
        # How many ways are left to move is what the search itself works out,
        # level by level, and a situation with none is scored as closed. As a
        # term it is one move's look at the same question, and at full weight
        # it outweighed everything else: measured 2026-09-17 on four games, a
        # 512 in every one with it, and 1024, 1024, 2048, 2048 without.
        said = made.terms(board, toward=toward, actions=actions, freedom=False)
        for name, measure in invented.items():
            try:
                said[name] = float(measure.read(as_arrangement(board)))
            except (AttributeError, TypeError, ValueError):
                continue
        value = sum(said[name] * float(weighed.get(name, 0.0)) for name in said)
        values[board] = value
        return value

    dead = -(1.0 + sum(abs(float(value)) for value in weighed.values()))
    started = time.monotonic()
    scored, reached = search(
        made, state, actions, budget_s=budget_s, worth=worth, dead=dead, fixed_depth=depth,
        no_deeper_than=no_deeper_than,
    )
    if settles_how_far:
        _SAW["acts"] = int(reached) if scored else 0
    logger.debug(
        "looked %d ahead on a compiled world over %d move(s) in %.3fs",
        reached, len(actions), time.monotonic() - started,
    )
    if line_holds is not None and len(scored) > 1:
        from core.agency.worth_thinking_about import TOO_CLOSE_TO_CALL  # noqa: PLC0415

        values = [value for value, _after in scored.values()]
        best = max(values)
        spread = best - min(values)
        trust = float(getattr(knows, "confidence", lambda: 0.0)() or 0.0)
        # Level with the best means inside what her own model could have got
        # wrong about the difference, and never less than the smallest
        # difference that means anything.
        level = max(TOO_CLOSE_TO_CALL, (1.0 - max(0.0, min(1.0, trust))) * spread)
        scored = {
            action: (
                value + (level * line_holds(as_arrangement(after)) if best - value <= level else 0.0),
                after,
            )
            for action, (value, after) in scored.items()
        }
    return {
        action: (value, why(as_arrangement(after), toward=toward, approach=approach))
        for action, (value, after) in scored.items()
    }


def _a_level_took(seconds: float) -> None:
    """Record what a level of looking cost, so the next depth is chosen from it."""
    spent = float(seconds or 0.0)
    if spent <= 0.0:
        return
    before = _A_LEVEL["seconds"]
    _A_LEVEL["seconds"] = spent if before <= 0.0 else before * 0.7 + spent * 0.3


def _reading(state: Any) -> str:
    """A state as a thing that can be compared to another state."""
    as_text = getattr(state, "as_text", None)
    return as_text() if callable(as_text) else str(state)
