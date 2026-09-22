"""ISC-v5's irreducibility line, run to the design written down before it was read.

Three things stood between the runtime sweep and a decisive interventional run.
It read each of its looks at the full alpha, so a cut that costs nothing could
be decided at any of them and the chance grew with every round. It stopped at
twenty-four anchors, which the controls showed cannot decide even the star. And
it kept drawing anchors while any horizon on the ladder was undecided, where
the one-step horizon decides nothing in a system that updates once a step, so
every cut would have drawn to the last look. See docs/ISC_V5_PREREGISTRATION.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject import isc_v5, v25_cut
from core.subject.v25_cut import anchor_schedule, merge_sweeps, playback_decided, sweep_cuts_over_lags

pytestmark = pytest.mark.unit

DOMAINS = ("A", "B", "C")  # three bipartitions


def _anchors(count: int) -> list:
    return [SimpleNamespace(snapshot=None, current=np.zeros(2)) for _ in range(count)]


def test_the_schedule_is_the_looks_and_the_level_is_divided_by_them() -> None:
    assert anchor_schedule((8, 16, 32, 64, 96, 128), rounds=2, available=200) == ((8, 16, 32, 64, 96, 128), 6)
    assert anchor_schedule((), rounds=3, available=200) == ((8, 16, 24), 3)
    # Fewer anchors than the design asks for: the looks are capped, the level is not raised.
    assert anchor_schedule((8, 16, 32, 64, 96, 128), rounds=2, available=40) == ((8, 16, 32, 40), 6)


@pytest.fixture
def rig(monkeypatch):
    levels: list[float] = []
    draws: list[int] = []
    decided_at: dict[tuple[str, int], int] = {}
    collected: list[tuple[str, int]] = []

    async def collect(runtime, anchors, conditions, *, left, right, turns, lags, offset=0, untouched=None):
        name = "".join(left) + "|" + "".join(right)
        collected.append((name, offset + len(anchors)))
        n = len(anchors)
        return {
            lag: {
                "context": np.zeros((n, 2)), "intact": np.zeros((n, 2)), "cut": np.ones((n, 2)),
                "sham_a": np.zeros((n, 2)), "sham_b": np.zeros((n, 2)), "reached": np.ones((n, 1)),
                "_name": name, "_lag": lag,
            }
            for lag in lags
        }

    def decide(slot, *, tau_seconds, seed, alpha, draws=200, permutation_draws=199):
        levels.append(alpha)
        rig_draws.append(draws)
        take = len(slot["intact"])
        replay = bool(np.array_equal(slot["cut"], slot["intact"]))
        at = decided_at.get((slot["_name"], slot["_lag"]))
        lower = -1.0 if replay or at is None or take < at else 1.0
        return SimpleNamespace(raw_rate=0.0, sham_rate=0.0), 0.0, lower, 0.5

    rig_draws = draws
    monkeypatch.setattr(v25_cut, "collect_partition_samples", collect)
    monkeypatch.setattr(v25_cut, "decide_cut", decide)
    return SimpleNamespace(levels=levels, draws=draws, decided_at=decided_at, collected=collected)


@pytest.mark.asyncio
async def test_only_the_deciding_horizon_draws_anchors(rig) -> None:
    for name in ("A|BC", "AB|C", "AC|B"):
        rig.decided_at[(name, 33)] = 16
    reports = await sweep_cuts_over_lags(
        None, _anchors(128), [], lags=(1, 33), frame_seconds=1 / 33, domains=DOMAINS,
        looks=(8, 16, 32, 64, 96, 128), draws=1000, deciding=(33,),
    )
    assert max(take for _, take in rig.collected) == 16, "lag 1, which never decides, kept cuts drawing"
    assert reports[33].irreducible and not reports[1].irreducible
    assert reports[1].undecided == [] and reports[1].deciding is False


@pytest.mark.asyncio
async def test_every_look_is_read_at_alpha_over_the_looks_with_the_draws_asked_for(rig) -> None:
    reports = await sweep_cuts_over_lags(
        None, _anchors(128), [], lags=(33,), frame_seconds=1 / 33, domains=DOMAINS,
        looks=(8, 16, 32, 64, 96, 128), draws=1000, alpha=0.05,
    )
    assert rig.levels and all(level == pytest.approx(0.05 / 6) for level in rig.levels)
    assert set(rig.draws) == {1000}
    report = reports[33].as_dict()
    assert report["looks"] == [8, 16, 32, 64, 96, 128]
    assert report["alpha_per_look"] == pytest.approx(0.05 / 6) and report["draws"] == 1000
    assert report["cuts_decided"] == 0 and len(report["undecided"]) == 3
    assert max(take for _, take in rig.collected) == 128


@pytest.mark.asyncio
async def test_the_playback_control_is_scored_where_each_cut_stopped_and_never_decided(rig) -> None:
    rig.decided_at[("A|BC", 33)] = 8
    reports = await sweep_cuts_over_lags(
        None, _anchors(32), [], lags=(33,), frame_seconds=1 / 33, domains=DOMAINS,
        looks=(8, 16, 32), draws=1000,
    )
    report = reports[33].as_dict()
    assert report["playback_scored"] == 3, "one at the decision, two at the last look"
    assert report["playback_decided"] == 0


def test_the_playback_control_uses_the_real_rule_and_is_not_decided() -> None:
    rng = np.random.default_rng(5)
    n = 24
    intact = rng.normal(size=(n, 3))
    samples = {
        "context": rng.normal(size=(n, 2)),
        "intact": intact,
        "cut": intact + 2.0,
        "sham_a": intact,
        "sham_b": intact + 0.01 * rng.normal(size=(n, 3)),
    }
    assert not playback_decided(samples, tau_seconds=1.0, seed=3, alpha=0.05 / 6, draws=200)


def test_shards_run_to_different_designs_are_not_merged() -> None:
    row = {"cut": "A|BC", "anchors": 8, "decided": True}
    one = {"tau_seconds": 1.0, "verdicts": [row], "looks": [8, 16], "alpha_per_look": 0.025, "draws": 200}
    other = {"tau_seconds": 1.0, "verdicts": [{**row, "cut": "AB|C"}], "looks": [8, 16, 32], "alpha_per_look": 0.05 / 3, "draws": 200}
    with pytest.raises(ValueError, match="different designs"):
        merge_sweeps([one, other], cuts_in_full=2)


def _sweep(decided: int, **over) -> dict:
    return {
        "looks": list(isc_v5.LOOKS), "draws": isc_v5.DRAWS, "alpha_per_look": isc_v5.ALPHA / len(isc_v5.LOOKS),
        "deciding": True, "screened": False, "shard": "", "cuts_tested": 511, "cuts_in_full": 511,
        "cuts_decided": decided, "undecided": [] if decided == 511 else ["P|IAGCSMWDN"],
        "playback_decided": 0, **over,
    }


def test_the_line_passes_only_on_every_cut_at_the_preregistered_design() -> None:
    passed = isc_v5.lines(_sweep(511), playback_decided=0, nulls_that_pass=[], null_sweeps={})
    assert [line["passed"] for line in passed] == [True, True]
    one_short = isc_v5.lines(_sweep(510), playback_decided=0, nulls_that_pass=[], null_sweeps={})
    assert [line["passed"] for line in one_short] == [False, False]
    other_design = isc_v5.lines(_sweep(511, draws=200), playback_decided=0, nulls_that_pass=[], null_sweeps={})
    assert not other_design[0]["passed"] and "draws" in other_design[0]["why"]
    screened = isc_v5.lines(_sweep(511, screened=True), playback_decided=0, nulls_that_pass=[], null_sweeps={})
    assert not screened[0]["passed"]


def test_a_passing_null_decided_everywhere_or_a_decided_playback_refuses_the_null_line() -> None:
    null = {"decided": 511, "cuts": 511, "looks": list(isc_v5.LOOKS), "draws": isc_v5.DRAWS,
            "alpha_per_look": isc_v5.ALPHA / len(isc_v5.LOOKS)}
    star = isc_v5.lines(_sweep(511), playback_decided=0, nulls_that_pass=["star"], null_sweeps={"star": null})
    assert star[0]["passed"] and not star[1]["passed"] and star[1]["value"]["decided_at_every_cut"] == ["star"]
    unswept = isc_v5.lines(_sweep(511), playback_decided=0, nulls_that_pass=["hub"], null_sweeps={})
    assert not unswept[1]["passed"] and unswept[1]["value"]["unswept"] == ["hub"]
    playback = isc_v5.lines(_sweep(511), playback_decided=2, nulls_that_pass=[], null_sweeps={})
    assert not playback[1]["passed"]


def test_the_battery_reports_v5_beside_v3_and_changes_neither() -> None:
    from core.subject.battery import assemble

    plain = assemble({}).as_dict()
    assert plain["v5_criteria"] == [] and plain["isc_v5_on_this_seed"] is False
    read = assemble({"interventional_cut": {"sweep": _sweep(511), "null_sweeps": {}, "nulls_that_pass": []}}).as_dict()
    assert [line["criterion"] for line in read["v5_criteria"]] == ["partition_irreducibility", "partition_beats_nulls"]
    assert all(line["passed"] for line in read["v5_criteria"])
    assert read["criteria"] == plain["criteria"], "v5 moved a v1 line"
    assert read["v3_criteria"] == plain["v3_criteria"], "v5 moved a v3 line"


def test_the_runner_takes_its_v5_design_from_one_place() -> None:
    chosen = isc_v5.design()
    assert chosen["looks"] == [8, 16, 32, 64, 96, 128] and chosen["draws"] == 1000
    assert chosen["deciding"] == [33] and chosen["lags"] == [33, 66] and chosen["anchors"] == 128


def test_the_fingerprint_moves_with_the_design() -> None:
    from core.subject.provenance import campaign_v25

    base = dict(seed=7, rounds=4, anchors=16, history_turns=2, turns=1, cut_rounds=2, support=("P", "I"))
    plain = campaign_v25(**base)["fingerprint"]
    v5 = campaign_v25(**base, design=isc_v5.design())["fingerprint"]
    assert plain != v5
