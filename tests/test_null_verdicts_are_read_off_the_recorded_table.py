"""A null's verdict, recomputed from the table a campaign recorded.

ISC-v2 changes how nulls are judged, and the change is only honest if ISC-v1's
verdicts come out the same when they are recomputed. These read the three
frozen runs' recorded null tables: v1 reproduces what each run reported, v2's
comparison set leaves out the bus the reference itself cannot beat, and read
across the three seeds, the seed on which the reference failed says the
instrument failed rather than that a null passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.subject.null_verdicts import (
    REFERENCE,
    beats_the_comparison_set,
    comparison_set,
    passes_the_conjunction,
    verdict_across_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("run_026", "run_027", "run_028")


def _nulls(run: str) -> dict:
    return json.loads((ROOT / "artifacts" / "subject_core" / run / "nulls.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("run", RUNS)
def test_v1_reproduces_the_conjunction_each_run_reported(run: str) -> None:
    nulls = _nulls(run)
    recomputed = {name: passes_the_conjunction(row) for name, row in nulls["detail"].items()}
    assert recomputed == nulls["conjunction"]


def test_v2_compares_against_the_surrogates_and_nulls_that_pass_everything_else() -> None:
    table = _nulls("run_026")["detail"]
    compared = comparison_set(table)
    assert {"replay", "time_shuffle"} <= set(compared)
    assert "all_to_all" not in compared and "low_rank" not in compared
    assert REFERENCE not in compared


def test_the_reference_beats_the_v2_comparison_set_where_v1_asked_the_impossible() -> None:
    nulls = _nulls("run_026")
    reference_phi = float(nulls["detail"][REFERENCE]["phi_do"])
    beats, _ = beats_the_comparison_set(reference_phi, nulls["detail"])
    assert beats
    assert reference_phi < max(float(row["phi_do"]) for name, row in nulls["detail"].items() if name != REFERENCE)


def test_across_the_three_seeds_the_reference_failing_on_one_is_the_instrument_failing() -> None:
    verdict = verdict_across_seeds([_nulls(run)["conjunction"] for run in RUNS])
    assert verdict["seeds"] == 3
    assert verdict["instrument_failed"] is True
    assert verdict["all_nulls_fail"] is False
    assert "low_rank" in verdict["nulls_passing"]


def test_a_null_that_fails_on_every_seed_fails_and_a_steady_reference_passes() -> None:
    steady = [{REFERENCE: True, "star": False, "hub": False} for _ in range(3)]
    verdict = verdict_across_seeds(steady)
    assert verdict == {
        "seeds": 3, "nulls_passing": [], "reference_passes": True,
        "instrument_failed": False, "all_nulls_fail": True,
    }


@pytest.mark.parametrize("run", ("run_026", "run_027"))
def test_on_a_steady_seed_only_the_reference_passes_the_v2_conjunction(run: str) -> None:
    from core.subject.null_verdicts import passes_the_v2_conjunction

    table = _nulls(run)["detail"]
    passing = sorted(name for name in table if passes_the_v2_conjunction(name, table))
    assert passing == [REFERENCE]


def test_on_seed_13_the_null_suite_flip_carries_into_v2() -> None:
    """The instability section 3 is for: one seed's toy recording lets low_rank through."""
    from core.subject.null_verdicts import passes_the_v2_conjunction

    table = _nulls("run_028")["detail"]
    assert sorted(name for name in table if passes_the_v2_conjunction(name, table)) == ["low_rank"]
