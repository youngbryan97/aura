"""The questions people actually ask a spreadsheet.

LIVE, 2026-08-27: "I've got a deals export at <path>. How many of them are
approved, what do they add up to, and which region has the highest average
approved deal size?" The answerer required a breakdown column, so a plain
filtered count resolved to nothing; the filter needed the COLUMN named, and
people write "approved deals" rather than "where status is approved"; a
superlative beat the measure, so "highest average" computed a maximum and
labelled it a total; and a count could not be grouped, so "which rep has the
most deals" was unanswerable.

The model then answered from a file it had not read, was correctly rejected for
having no numbers, and the turn ended in a canned apology — for figures the
file settles exactly.

Every figure below is checked against the same data computed independently, so
the test fails if the reading drifts rather than if the wording does.
"""

from __future__ import annotations

import collections
import csv
import random
from pathlib import Path

import pytest

from core.conversation.tabular_answer import (
    answer_tabular_question,
    describe_tabular_answer,
)


@pytest.fixture
def deals(tmp_path: Path) -> Path:
    random.seed(1187)
    regions = ["North", "South", "East", "West"]
    reps = ["Ilse", "Marek", "Nour", "Priya", "Tomas", "Wren"]
    statuses = ["approved", "pending", "rejected"]
    rows = [
        {
            "id": index,
            "region": random.choice(regions),
            "rep": random.choice(reps),
            "status": random.choice(statuses),
            "amount_gbp": round(random.uniform(180, 9400), 2),
            "days_open": random.randint(1, 96),
        }
        for index in range(1, 84)
    ]
    path = tmp_path / "deals.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_a_filtered_count(deals: Path) -> None:
    """"How many are approved" — no breakdown, and the column is never named."""
    expected = sum(1 for row in _rows(deals) if row["status"] == "approved")
    answer = answer_tabular_question(deals, "How many deals are approved?")
    assert answer is not None
    assert answer.ranking[0][1] == float(expected)
    assert answer.filters == (("status", "approved"),)
    assert answer.rows_used == expected
    assert str(expected) in describe_tabular_answer(answer)


def test_a_filtered_mean(deals: Path) -> None:
    approved = [float(row["amount_gbp"]) for row in _rows(deals) if row["status"] == "approved"]
    expected = sum(approved) / len(approved)
    answer = answer_tabular_question(deals, "What is the average approved amount_gbp?")
    assert answer is not None
    assert answer.ranking[0][1] == pytest.approx(expected, abs=0.01)


def test_a_measure_beats_a_superlative(deals: Path) -> None:
    """"Highest average" is the mean, read from the top of the ranking.

    Checking max first made it the largest single row and labelled the answer
    "total". A superlative says which end to read, not what to compute.
    """
    rows = [row for row in _rows(deals) if row["status"] == "approved"]
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["region"]].append(float(row["amount_gbp"]))
    expected = max(
        ((region, sum(values) / len(values)) for region, values in grouped.items()),
        key=lambda pair: pair[1],
    )
    answer = answer_tabular_question(
        deals, "Which region has the highest average approved amount_gbp?"
    )
    assert answer is not None
    assert answer.aggregation == "mean"
    assert answer.ranking[0][0] == expected[0]
    assert answer.ranking[0][1] == pytest.approx(expected[1], abs=0.01)


def test_a_grouped_count(deals: Path) -> None:
    """"Which rep has the most deals" names no measure; frequency is the only one."""
    counted = collections.Counter(row["rep"] for row in _rows(deals))
    answer = answer_tabular_question(deals, "Which rep has the most deals?")
    assert answer is not None
    assert answer.aggregation == "count"
    assert answer.ranking[0] == (counted.most_common(1)[0][0], float(counted.most_common(1)[0][1]))
    assert "count rows" in describe_tabular_answer(answer)


def test_a_count_per_group(deals: Path) -> None:
    counted = collections.Counter(row["region"] for row in _rows(deals))
    answer = answer_tabular_question(deals, "How many deals per region?")
    assert answer is not None
    assert dict(answer.ranking) == {key: float(value) for key, value in counted.items()}


def test_an_ambiguous_measure_is_still_refused(deals: Path) -> None:
    """Three numeric columns and none of them named: no single reading.

    A wrong number served with authority is worse than no number, and that
    discipline is the reason this module can be trusted at all.
    """
    assert answer_tabular_question(deals, "What do the approved deals add up to?") is None


def test_a_value_in_two_columns_is_too_ambiguous_to_filter(tmp_path: Path) -> None:
    """A value naming its own column only works when exactly one column can mean it."""
    path = tmp_path / "t.csv"
    path.write_text(
        "stage,review,amount\n"
        "approved,approved,10\n"
        "pending,rejected,20\n"
        "approved,rejected,30\n"
    )
    answer = answer_tabular_question(path, "How many are approved?")
    assert answer is None or not answer.filters
