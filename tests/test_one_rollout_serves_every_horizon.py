"""One set of clamped rollouts per cut serves every horizon on the ladder.

The spectrum scored each horizon with its own sweep, so every cut's four arms
ran again for every lag although one trajectory holds all of them. These pin
the shared sweep: rollouts do not multiply with the ladder, each horizon is
decided on its own, a cut stops drawing only when every horizon it reached is
decided, and a horizon past the end of the rollouts is reported unreached
rather than scored as the last frame under another name.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject import v25_cut
from core.subject.v25_cut import ANCHOR_STEP, MINIMUM_ANCHORS, OPENING_ANCHORS, sweep_cuts_over_lags

DOMAINS = ("A", "B", "C")  # three bipartitions


def _anchors(count: int) -> list:
    return [SimpleNamespace(snapshot=None, current=np.zeros(2)) for _ in range(count)]


@pytest.fixture
def rig(monkeypatch):
    """A fake collector and decision rule, so only the sweep's own logic runs."""
    calls: list[tuple] = []
    decided_at: dict[tuple[str, int], bool] = {}
    unreached_lags: set[int] = set()

    async def collect(runtime, anchors, conditions, *, left, right, turns, lags):
        calls.append(("".join(left) + "|" + "".join(right), tuple(lags), len(anchors)))
        n = len(anchors)
        out = {}
        for lag in lags:
            reached = np.zeros((n, 1)) if lag in unreached_lags else np.ones((n, 1))
            out[lag] = {
                "context": np.zeros((n, 2)),
                "intact": np.zeros((n, 2)),
                "cut": np.zeros((n, 2)),
                "sham_a": np.zeros((n, 2)),
                "sham_b": np.zeros((n, 2)),
                "reached": reached,
                "_name": "".join(left) + "|" + "".join(right),
                "_lag": lag,
            }
        return out

    decisions: list[tuple[str, int]] = []

    def decide(slot, *, tau_seconds, seed, alpha):
        key = (slot["_name"], slot["_lag"])
        decisions.append(key)
        lower = 1.0 if decided_at.get(key, False) else -1.0
        estimate = SimpleNamespace(raw_rate=0.0, sham_rate=0.0, rate=0.0)
        return estimate, 0.0, lower, 0.5

    monkeypatch.setattr(v25_cut, "collect_partition_samples", collect)
    monkeypatch.setattr(v25_cut, "decide_cut", decide)
    return SimpleNamespace(calls=calls, decided_at=decided_at, unreached=unreached_lags, decisions=decisions)


@pytest.mark.asyncio
async def test_rollouts_do_not_multiply_with_the_ladder(rig) -> None:
    reports = await sweep_cuts_over_lags(
        None, _anchors(OPENING_ANCHORS), [], lags=(1, 2, 4, 8),
        frame_seconds=0.03, rounds=1, domains=DOMAINS,
    )
    assert len(rig.calls) == 3, "one collection per cut, not one per cut per horizon"
    assert all(lags == (1, 2, 4, 8) for _, lags, _ in rig.calls)
    assert sorted(reports) == [1, 2, 4, 8]
    assert all(len(report.verdicts) == 3 for report in reports.values())


@pytest.mark.asyncio
async def test_each_horizon_is_decided_on_its_own(rig) -> None:
    for name in ("A|BC", "AB|C", "AC|B"):
        rig.decided_at[(name, 1)] = True
    reports = await sweep_cuts_over_lags(
        None, _anchors(OPENING_ANCHORS), [], lags=(1, 2),
        frame_seconds=0.03, rounds=1, domains=DOMAINS,
    )
    assert not reports[1].undecided
    assert len(reports[2].undecided) == 3


@pytest.mark.asyncio
async def test_a_cut_keeps_drawing_while_any_horizon_is_open(rig) -> None:
    for name in ("A|BC", "AB|C", "AC|B"):
        rig.decided_at[(name, 1)] = True
        rig.decided_at[(name, 2)] = True
    rig.decided_at.pop(("AB|C", 2))
    await sweep_cuts_over_lags(
        None, _anchors(OPENING_ANCHORS + ANCHOR_STEP), [], lags=(1, 2),
        frame_seconds=0.03, rounds=2, domains=DOMAINS,
    )
    second_round = [call for call in rig.calls if call[2] == OPENING_ANCHORS + ANCHOR_STEP]
    assert [name for name, _, _ in second_round] == ["AB|C"]
    # And the horizon that was already decided is not scored again.
    assert rig.decisions.count(("AB|C", 1)) == 1


@pytest.mark.asyncio
async def test_a_horizon_past_the_rollouts_is_unreached_not_scored(rig) -> None:
    rig.unreached.add(8)
    reports = await sweep_cuts_over_lags(
        None, _anchors(OPENING_ANCHORS), [], lags=(1, 8),
        frame_seconds=0.03, rounds=1, domains=DOMAINS,
    )
    assert not any(lag == 8 for _, lag in rig.decisions)
    assert len(reports[8].undecided) == 3
    assert all("not reached" in v.note for v in reports[8].verdicts)
    assert not reports[8].irreducible


@pytest.mark.asyncio
async def test_too_few_anchors_refuses(rig) -> None:
    with pytest.raises(ValueError):
        await sweep_cuts_over_lags(
            None, _anchors(MINIMUM_ANCHORS - 1), [], lags=(1,),
            frame_seconds=0.03, domains=DOMAINS,
        )


@pytest.mark.asyncio
async def test_an_empty_ladder_refuses(rig) -> None:
    with pytest.raises(ValueError):
        await sweep_cuts_over_lags(
            None, _anchors(OPENING_ANCHORS), [], lags=(),
            frame_seconds=0.03, domains=DOMAINS,
        )
