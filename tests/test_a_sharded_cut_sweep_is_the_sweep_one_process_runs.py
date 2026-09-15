"""A cut sweep split across processes merges back into the sweep one process would run.

The authoritative carrier run scores all 511 bipartitions, and one process
scoring them in turn takes days. Each cut's rollouts depend only on the anchors,
so the cuts can be split across workers. These pin what makes a split safe:
the shards partition the cuts, a cut's bootstrap seed belongs to the cut rather
than to a queue, a verdict survives the trip through its report, and a merge is
refused unless every cut was scored exactly once.
"""

from __future__ import annotations

import pytest

from core.subject.v25_cut import CutVerdict, RecordedRate, SweepReport, merge_sweeps, shard_of
from core.subject.v25_runtime import bipartitions

pytestmark = pytest.mark.unit


def test_shards_partition_every_cut_and_keep_each_cut_s_place() -> None:
    cuts = list(bipartitions())
    assert len(cuts) == 511
    count = 7
    places = sorted(place for index in range(count) for place, _ in shard_of(cuts, index, count))
    assert places == list(range(511))
    for index in range(count):
        for place, cut in shard_of(cuts, index, count):
            assert cuts[place] == cut


def test_a_shard_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError):
        shard_of(list(bipartitions()), 3, 3)


def _verdict(name: str, lower: float, decided: bool) -> CutVerdict:
    left, _, right = name.partition("|")
    return CutVerdict(
        left=tuple(left), right=tuple(right), anchors_used=8,
        estimate=RecordedRate(raw_rate=lower + 0.2, sham_rate=0.1),
        excess=lower + 0.1, lower_bound=lower, p_value=0.01 if decided else 0.4, decided=decided,
    )


def test_a_verdict_survives_the_trip_through_its_report() -> None:
    original = _verdict("PAG|ICSMWDN", 0.031, True)
    back = CutVerdict.from_dict(original.as_dict())
    assert back.name == original.name
    assert back.lower_bound == pytest.approx(original.lower_bound)
    assert back.decided and back.estimate is not None
    assert back.as_dict()["raw_rate"] == pytest.approx(original.as_dict()["raw_rate"])


def _shard(index: str, verdicts: list[CutVerdict]) -> dict:
    return SweepReport(tau_seconds=0.030303, cuts_in_full=4, verdicts=verdicts, shard=index,
                       undecided=[v.name for v in verdicts if not v.decided], anchors_spent=16).as_dict()


def test_shards_merge_into_one_sweep_whose_weakest_cut_is_the_weakest_of_all() -> None:
    first = _shard("0/2", [_verdict("PA|IG", 0.05, True), _verdict("PI|AG", 0.02, True)])
    second = _shard("1/2", [_verdict("PG|IA", 0.01, True), _verdict("PIA|G", -0.01, False)])
    assert not SweepReport(tau_seconds=0.03, verdicts=[_verdict("PA|IG", 0.05, True)], shard="0/2").irreducible
    merged = merge_sweeps([first, second], cuts_in_full=4)
    assert len(merged.verdicts) == 4
    assert merged.weakest.name == "PIA|G"
    assert merged.undecided == ["PIA|G"]
    assert merged.anchors_spent == 32
    assert not merged.irreducible


def test_a_merge_is_refused_when_a_cut_is_missing_or_scored_twice() -> None:
    first = _shard("0/2", [_verdict("PA|IG", 0.05, True), _verdict("PI|AG", 0.02, True)])
    with pytest.raises(ValueError, match="of 4 cuts"):
        merge_sweeps([first], cuts_in_full=4)
    with pytest.raises(ValueError, match="more than one shard"):
        merge_sweeps([first, first], cuts_in_full=4)
