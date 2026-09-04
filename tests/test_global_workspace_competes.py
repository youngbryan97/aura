"""The global workspace must actually compete.

The class docstring says the refractory mechanism "prevents the same subsystem
from dominating every cycle and forces genuine competition". It did the
opposite. Every LOSER was inhibited for a tick, and inhibition is checked in
``submit()``, so the sources that had just lost could not bid on the next tick.
The winner then ran unopposed, won again, and inhibited them again.

Measured before the fix: four sources bidding every tick at 0.90 / 0.88 / 0.86
/ 0.84 gave the top source **24 wins out of 24** while the other three had half
their submissions refused. A two-point priority difference bought a permanent
monopoly of the broadcast — which means that in steady state there was no
competition and no global workspace, only "highest-priority source always
wins".

Hard-inhibiting the winner instead was tried and is worse in a subtler way: an
urgent source at 0.99 and an idle one at 0.20 alternated 50/50, because a hard
block ignores how much stronger the bid was, and a source bidding alone won
only half its ticks. The mechanism that works is adaptation — fatigue the
recent winner's effective priority and let it recover.

That fix was real but incomplete, and the assertions written with it are the
reason it looked finished. They asked for ``top_share < 0.75`` and two distinct
winners — both of which are true of a perfect **duopoly**, which is what the
mechanism actually settled into: a and b splitting 24 ticks 12/12 in a strict
a-b-a-b alternation while c and d won nothing at all. A monopoly of one had
become a cartel of two and the regression test applauded.

The cause was structural rather than a tuning miss. Adaptation is a leaky
integrator, so a source winning fraction ``f`` of ticks is in equilibrium when
``g·f = r``; a fixed recovery rate therefore pins the sustainable share at
``r/g`` and lets exactly ``g/r`` sources rotate — two, for every possible field
size. ``GlobalWorkspace._fatigue_recovery`` derives ``r = g/n`` instead, making
the equilibrium share ``1/n``.

These regimes define correct behaviour, and they are deliberately in tension:
anything that widens the rotation can flatten real urgency, and anything that
sharpens urgency can starve the field. Both halves are asserted here, via
``core.verify.dynamics``, over a trajectory rather than a single tick.
"""

from __future__ import annotations

import asyncio
import collections

import pytest

from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace
from core.verify.dynamics import competition_health, no_limit_cycle

TICKS = 24


async def _run(sources: dict[str, float], ticks: int = TICKS):
    workspace = GlobalWorkspace()
    wins: collections.Counter[str] = collections.Counter()
    refused: collections.Counter[str] = collections.Counter()
    sequence: list[str] = []
    for tick in range(ticks):
        for name, priority in sources.items():
            admitted = await workspace.submit(
                CognitiveCandidate(
                    content=f"{name}@{tick}", source=name, priority=priority
                )
            )
            if not admitted:
                refused[name] += 1
        winner = await workspace.run_competition()
        if winner is not None:
            wins[winner.source] += 1
            sequence.append(winner.source)
    return wins, refused, sequence


def test_a_two_point_gap_does_not_buy_a_monopoly():
    """The regression, asserted as a property instead of a symptom.

    Checking "the top source is under 75%" is what let the duopoly through.
    The real requirement is the opposing pair: nobody is locked out, *and*
    bidding higher still wins more often. Neither half is sufficient alone —
    ignoring the bids entirely satisfies the first, and a monopoly satisfies
    the second.
    """
    bids = {"memory": 0.90, "drive": 0.88, "perception": 0.86, "curiosity": 0.84}
    wins, _, _ = _sync(bids)
    total = sum(wins.values())
    shares = {name: wins.get(name, 0) / total for name in bids}

    findings = competition_health(
        shares,
        bids,
        subject="global_workspace",
        # Four near-equal sources over 24 ticks: an even rotation gives each
        # 0.25, so a floor of 0.10 fails a source that is being crowded out
        # while tolerating the integer remainder of 24/4.
        min_share=0.10,
        min_normalised_entropy=0.90,
        # One tick of 24 is 0.042; two sources can legitimately differ by that
        # much without arbitration being broken.
        order_tolerance=0.10,
    )
    assert not findings, "workspace competition is unhealthy:\n" + "\n".join(
        f"  {f}" for f in findings
    )


def test_the_field_does_not_collapse_to_a_cartel():
    """Every source that bids competitively must reach broadcast.

    This is the duopoly regression specifically: c and d measured 0.000 while
    the previous assertions passed.
    """
    bids = {"a": 0.90, "b": 0.88, "c": 0.86, "d": 0.84}
    wins, _, _ = _sync(bids)
    silent = sorted(name for name in bids if wins.get(name, 0) == 0)
    assert not silent, f"sources never reached broadcast in {TICKS} ticks: {silent}"


def test_arbitration_does_not_lock_into_a_short_cycle():
    """A fixed repeating order means arbitration stopped reading its inputs.

    Entropy alone cannot see this: a strict a-b-a-b alternation scores a
    perfect rotation between two sources.
    """
    _, _, sequence = _sync({"a": 0.90, "b": 0.88, "c": 0.86, "d": 0.84})
    findings = no_limit_cycle(
        sequence, max_period=2, min_repeats=6, subject="global_workspace"
    )
    assert not findings, "\n".join(str(f) for f in findings)


def test_identical_bids_are_recorded_as_a_tie_impasse():
    """A decision nothing discriminates must be counted, not just made.

    Exact-equality detection finds nothing here, and why is the point:
    effective_priority scales salience by (1 - 0.03*age), so four sources
    bidding an identical 0.70 came out ~2e-6 apart purely from submission
    timing — measured, 0 exact ties in 12 ticks. The workspace was settling
    those by sub-microsecond arrival order while presenting it as a priority
    difference. Detection compares against that mechanism's own noise floor.
    """

    async def scenario():
        workspace = GlobalWorkspace()
        for tick in range(TICKS):
            for name in ("a", "b", "c", "d"):
                await workspace.submit(
                    CognitiveCandidate(
                        content=f"{name}@{tick}", source=name, priority=0.70
                    )
                )
            await workspace.run_competition()
        return workspace.get_snapshot()

    snapshot = asyncio.run(scenario())
    assert snapshot["tie_impasses"] > 0, (
        "four identical bids produced no recorded tie; the impasse is being "
        "resolved by arrival order and reported as a decision"
    )
    assert len(snapshot["last_tie"]) > 1


def test_a_tie_is_not_settled_by_who_submitted_first():
    """The outcome of indistinguishable bids must not depend on arrival order.

    This is the property, and it is stronger than "the shares look fair": the
    same four sources are submitted in a fixed order and in a rotating one, and
    the resulting distribution has to be the same. Under the previous
    arrival-order resolution it could not have been, because arrival order was
    the entire input.
    """

    async def scenario(rotate: bool):
        workspace = GlobalWorkspace()
        names = ["a", "b", "c", "d"]
        wins: collections.Counter[str] = collections.Counter()
        for tick in range(48):
            order = names[tick % 4 :] + names[: tick % 4] if rotate else names
            for name in order:
                await workspace.submit(
                    CognitiveCandidate(
                        content=f"{name}@{tick}", source=name, priority=0.70
                    )
                )
            winner = await workspace.run_competition()
            if winner is not None:
                wins[winner.source] += 1
        total = sum(wins.values())
        return {name: wins.get(name, 0) / total for name in names}

    fixed = asyncio.run(scenario(rotate=False))
    rotated = asyncio.run(scenario(rotate=True))
    assert fixed == rotated, (
        f"submission order changed the outcome: fixed={fixed} rotated={rotated}"
    )


def test_identical_bids_share_the_broadcast_evenly():
    """Nothing distinguishes these sources, so nothing should favour one."""
    bids = {"a": 0.70, "b": 0.70, "c": 0.70, "d": 0.70}
    wins, _, sequence = _sync(bids, ticks=48)
    total = sum(wins.values())
    shares = {name: wins.get(name, 0) / total for name in bids}
    findings = competition_health(
        shares,
        bids,
        subject="global_workspace/tied",
        min_share=0.15,
        min_normalised_entropy=0.95,
        # Every bid is equal, so no ordering is being asserted here.
        order_tolerance=1.0,
    )
    findings += no_limit_cycle(sequence, max_period=1, min_repeats=8)
    assert not findings, "\n".join(str(f) for f in findings)


def _three_bid_scenario(gap_s: float = 0.0) -> dict:
    """Three clearly different bids, optionally submitted slowly."""

    async def scenario():
        workspace = GlobalWorkspace()
        for tick in range(TICKS):
            for name, priority in (("a", 0.90), ("b", 0.70), ("c", 0.50)):
                await workspace.submit(
                    CognitiveCandidate(
                        content=f"{name}@{tick}", source=name, priority=priority
                    )
                )
                if gap_s:
                    await asyncio.sleep(gap_s)
            await workspace.run_competition()
        return workspace.get_snapshot()

    return asyncio.run(scenario())


def test_the_competition_is_decided_by_cognition_not_by_the_clock():
    """The flake, and why it was more than a flake.

    This regression passed about half the time. The tie floor was 0.03 x the
    wall-clock span between the first and last submission, so how far apart
    two priorities had to be before they counted as different was set by
    whatever else the machine was doing — a garbage collection between two
    submits widened it, a quiet moment narrowed it.

    Underneath that was a worse one: `effective_priority` read the clock
    itself and was used as a sort key, so the comparator changed between
    comparisons. A comparison function that is not consistent does not
    produce a jittered ordering, it produces an arbitrary one.

    Both are fixed by making a competition a single instant: the clock is
    read once before any comparison, every candidate is aged against that,
    and the tie test runs on cognitive priority with recency left out. So the
    same bids must now give the same answer every time.
    """
    runs = [_three_bid_scenario()["tie_impasses"] for _ in range(8)]
    assert len(set(runs)) == 1, (
        f"the same three bids gave different results across runs: {runs}. "
        "Something outside the cognitive state is deciding the competition"
    )


def test_slowing_the_submissions_down_changes_nothing():
    """The direct test of the defect that was reported.

    Ten milliseconds between submissions is an eternity next to the
    microseconds the old floor was built from. If arrival timing still
    reached the tie decision, this would part company with the fast run.
    """
    fast = _three_bid_scenario()["tie_impasses"]
    slow = _three_bid_scenario(gap_s=0.01)["tie_impasses"]
    assert fast == slow, (
        f"submitting the same bids more slowly changed the tie count "
        f"({fast} -> {slow}); arrival time is still deciding a semantic "
        "competition"
    )


def test_a_tie_is_only_ever_reported_for_bids_that_are_actually_equal():
    """The original intent of this test, in the form that survives.

    It asserted that three different bids produce no ties at all, which is a
    stronger claim than it looks: fatigue exists precisely to stop one source
    monopolising, so adaptation is DESIGNED to drive a winner down toward its
    rivals. When it succeeds exactly, that is a real tie and recording it is
    what the impasse mechanism is for.

    What must never happen is a tie between bids that cognitive priority
    still separates. That is what this checks, and it is what the timing
    floor used to violate.
    """

    async def scenario():
        workspace = GlobalWorkspace()
        seen: list[tuple[tuple[str, ...], dict[str, float]]] = []
        for tick in range(TICKS):
            live = []
            for name, priority in (("a", 0.90), ("b", 0.70), ("c", 0.50)):
                candidate = CognitiveCandidate(
                    content=f"{name}@{tick}", source=name, priority=priority
                )
                live.append(candidate)
                await workspace.submit(candidate)
            ties_before = workspace.get_snapshot()["tie_impasses"]
            await workspace.run_competition()
            snapshot = workspace.get_snapshot()
            if snapshot["tie_impasses"] > ties_before:
                # The values the workspace itself compared. Reconstructing
                # them here was wrong: fatigue recovers inside the same call,
                # so a snapshot taken beforehand names different numbers.
                seen.append((workspace._last_tie, snapshot["last_tie_values"]))
        return seen

    for tied_sources, adjusted in asyncio.run(scenario()):
        values = [adjusted[source] for source in tied_sources if source in adjusted]
        assert len(values) > 1, tied_sources
        spread = max(values) - min(values)
        assert spread <= 1e-9, (
            f"{tied_sources} were called tied while cognitive priority still "
            f"separated them by {spread:.3g}: {adjusted}"
        )


def test_losing_a_bid_does_not_silence_the_next_one():
    """Being outbid is a reason to bid again, not to be excluded."""
    _, refused, _seq = _sync({"a": 0.90, "b": 0.88, "c": 0.86})
    assert not refused, f"sources were refused submission after losing: {dict(refused)}"


def test_priority_still_dominates():
    """Adaptation must not flatten a real difference in urgency."""
    wins, _, _seq = _sync({"urgent": 0.99, "idle": 0.20})
    assert wins.get("urgent", 0) == TICKS, (
        f"an urgent source lost ticks to an idle one: {dict(wins)}"
    )


def test_a_clear_gap_is_respected():
    """0.30 apart is not a near-tie; the stronger source should hold."""
    wins, _, _seq = _sync({"strong": 0.90, "weak": 0.60})
    assert wins.get("strong", 0) == TICKS, dict(wins)


def test_a_lone_source_is_never_silenced():
    """With nothing else to attend to, the only bid must win every tick.

    The hard-refractory attempt failed exactly here: 12 of 24.
    """
    wins, _, _seq = _sync({"only": 0.9})
    assert wins.get("only", 0) == TICKS, dict(wins)


def test_ignition_requires_crossing_the_threshold():
    """Sub-threshold content wins the slot but must not ignite."""

    async def scenario():
        workspace = GlobalWorkspace()
        for name, priority in (("a", 0.2), ("b", 0.3)):
            await workspace.submit(
                CognitiveCandidate(content="x", source=name, priority=priority)
            )
        await workspace.run_competition()
        return workspace.is_ignited(), workspace.get_ignition_level()

    ignited, level = asyncio.run(scenario())
    assert ignited is False
    assert level < GlobalWorkspace._IGNITION_THRESHOLD


def test_ignition_fires_above_the_threshold():
    async def scenario():
        workspace = GlobalWorkspace()
        await workspace.submit(
            CognitiveCandidate(content="x", source="urgent", priority=0.95)
        )
        await workspace.run_competition()
        return workspace.is_ignited()

    assert asyncio.run(scenario()) is True


def _sync(sources: dict[str, float], ticks: int = TICKS):
    return asyncio.run(_run(sources, ticks))


@pytest.mark.parametrize(
    "attribute", ["_WINNER_FATIGUE", "_MAX_FATIGUE"]
)
def test_the_adaptation_constants_exist(attribute: str) -> None:
    """Named constants, so the tuning is visible rather than buried."""
    assert isinstance(getattr(GlobalWorkspace, attribute), float)
