"""The sharded carrier run merges its workers only when they ran one experiment.

A shard worker is its own process with its own organism, and two processes do
not reach identical states from one seed. These pin the checks that make a
merge honest: anchor banks from one kind of life pass the exchangeability test
and shifted ones do not, shards that ran other horizons, another clock or an
incomplete split are refused, and the launcher gives every process the same
experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from core.subject.v25_cut import (
    CutVerdict,
    RecordedRate,
    SweepReport,
    anchors_exchangeable,
    merge_shard_payloads,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def test_banks_from_one_life_are_exchangeable_and_a_shifted_bank_is_not() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(40, 20))
    same = rng.normal(size=(40, 20))
    shifted = rng.normal(size=(40, 20)) + 1.5
    assert anchors_exchangeable(reference, same, comparisons=4, seed=2)["exchangeable"]
    check = anchors_exchangeable(reference, shifted, comparisons=4, seed=2)
    assert not check["exchangeable"]
    assert check["draws"] >= 20 / check["level"]


def test_banks_of_different_widths_are_not_measured() -> None:
    check = anchors_exchangeable(np.zeros((5, 3)), np.zeros((5, 4)))
    assert check["measured"] is False and check["exchangeable"] is False


def _payload(index: int, count: int, verdict_names: list[str], anchors: np.ndarray, **overrides) -> dict:
    verdicts = [
        CutVerdict(left=tuple(n.split("|")[0]), right=tuple(n.split("|")[1]), anchors_used=8,
                   estimate=RecordedRate(raw_rate=0.3, sham_rate=0.1), excess=0.2, lower_bound=0.05,
                   p_value=0.01, decided=True)
        for n in verdict_names
    ]
    report = SweepReport(tau_seconds=0.030303, cuts_in_full=4, verdicts=verdicts, shard=f"{index}/{count}").as_dict()
    payload = {
        "shard": f"{index}/{count}",
        "lags": [1],
        "frame_seconds": 0.030303,
        "support": ["P", "I", "A"],
        "conditions": ["conversation", "idle"],
        "anchor_states": anchors.tolist(),
        "reports": {"lag_1": report},
    }
    payload.update(overrides)
    return payload


def _merge(payloads, reference):
    return merge_shard_payloads(
        payloads, reference_anchors=reference, cuts_in_full=4, lags=[1], frame_seconds=0.030303,
        support=["P", "I", "A"], conditions=["conversation", "idle"], seed=3,
    )


def test_matching_shards_merge_into_the_whole_sweep_with_the_gate_recorded() -> None:
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(30, 12))
    payloads = [
        _payload(0, 2, ["PI|A", "PA|I"], rng.normal(size=(30, 12))),
        _payload(1, 2, ["P|IA", "PIA|X"], rng.normal(size=(30, 12))),
    ]
    reports, gate = _merge(payloads, reference)
    assert len(reports[1].verdicts) == 4
    assert gate["shards"] == 2 and gate["exchangeable"]


def test_a_shard_with_shifted_anchors_is_merged_but_marked_for_the_authority_gate() -> None:
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(30, 12))
    payloads = [
        _payload(0, 2, ["PI|A", "PA|I"], rng.normal(size=(30, 12))),
        _payload(1, 2, ["P|IA", "PIA|X"], rng.normal(size=(30, 12)) + 2.0),
    ]
    _, gate = _merge(payloads, reference)
    assert not gate["exchangeable"]
    assert "shard 1/2" in gate["why"]


@pytest.mark.parametrize(
    "override, message",
    [
        ({"lags": [8]}, "other horizons"),
        ({"frame_seconds": 0.05}, "another clock"),
        ({"conditions": ["stress"]}, "other conditions"),
    ],
)
def test_shards_that_ran_another_experiment_are_refused(override, message) -> None:
    rng = np.random.default_rng(6)
    reference = rng.normal(size=(10, 4))
    payloads = [
        _payload(0, 2, ["PI|A", "PA|I"], rng.normal(size=(10, 4))),
        _payload(1, 2, ["P|IA", "PIA|X"], rng.normal(size=(10, 4)), **override),
    ]
    with pytest.raises(ValueError, match=message):
        _merge(payloads, reference)


def test_an_incomplete_split_is_refused() -> None:
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="not every index"):
        _merge([_payload(1, 2, ["PI|A", "PA|I"], rng.normal(size=(10, 4)))], rng.normal(size=(10, 4)))


def test_the_launcher_gives_every_process_one_experiment() -> None:
    spec = importlib.util.spec_from_file_location("sharded", REPO / "tools" / "run_subject_core_v25_sharded.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = argparse.Namespace(workers=3, rounds=24, anchors=32, history_turns=8, turns=1, cut_rounds=2,
                              seed=7, domains="", conditions=0, allow_degraded=False, hours=12.0)
    plan = module.commands(args, Path("/tmp/launch"))
    names = [name for name, _ in plan]
    assert names == ["worker_0", "worker_1", "worker_2", "coordinator"]
    shared = [argv[argv.index("--rounds"):argv.index("--rounds") + 12] for _, argv in plan]
    assert all(block == shared[0] for block in shared)
    assert [argv[argv.index("--shard") + 1] for _, argv in plan[:3]] == ["0/3", "1/3", "2/3"]
    coordinator = plan[-1][1]
    assert "--from-shards" in coordinator and "--shard" not in coordinator
