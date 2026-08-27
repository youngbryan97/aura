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


def test_a_message_asking_two_things_gets_two_readings(deals: Path) -> None:
    """LIVE, 2026-08-27: the regional means came back and the count did not.

    One reading resolved, the other was never attempted, and nothing said that
    half the message went unanswered.
    """
    from interface.routes.chat import _tabular_readings

    readings = _tabular_readings(
        deals,
        "how many of those are approved, and which region has the highest "
        "average approved amount_gbp?",
    )
    assert len(readings) >= 2
    joined = "\n".join(readings)
    assert "status is approved" in joined
    assert "By region" in joined


def test_the_same_figure_is_not_said_twice(deals: Path) -> None:
    """Two clauses often resolve alike, and a repeat reads as two findings."""
    from interface.routes.chat import _tabular_readings

    readings = _tabular_readings(deals, "how many are approved? how many approved deals?")
    assert len(readings) == len(set(readings))


def test_a_false_premise_about_the_table_is_settled(deals: Path) -> None:
    """Doubting a premise is worth much less than checking it.

    LIVE, 2026-08-27: "Since West came out top on average approved deal size in
    deals.csv, what's West doing that the other regions should copy?" She
    doubted it — correctly, the leader is South — and then reasoned from
    figures she had never looked at: "West often has deals sitting for a long
    time or getting rejected."
    """
    from core.conversation.tabular_answer import check_ranking_claim

    counted = collections.Counter(row["rep"] for row in _rows(deals))
    leader, most = counted.most_common(1)[0]
    runner_up = counted.most_common(2)[1][0]

    told = check_ranking_claim(
        deals, f"Since {runner_up} has the most deals, what are they doing right?"
    )
    assert told, "a false claim about this table went unchecked"
    assert leader in told and str(most) in told
    assert runner_up in told


def test_a_true_premise_is_left_alone(deals: Path) -> None:
    """Correcting a claim that is right is worse than saying nothing."""
    from core.conversation.tabular_answer import check_ranking_claim

    counted = collections.Counter(row["rep"] for row in _rows(deals))
    leader = counted.most_common(1)[0][0]
    assert check_ranking_claim(deals, f"Since {leader} has the most deals, what now?") == ""


def test_a_claim_about_something_the_table_does_not_hold(deals: Path) -> None:
    """A wrong correction served with authority is worse than no correction."""
    from core.conversation.tabular_answer import check_ranking_claim

    assert check_ranking_claim(deals, "Since Atlantis has the most deals, what now?") == ""
    assert check_ranking_claim(deals, "what should we do about the pipeline?") == ""


def test_an_ambiguous_measure_is_not_corrected(deals: Path) -> None:
    """"Deal size" fits three numeric columns, so there is no single reading."""
    from core.conversation.tabular_answer import check_ranking_claim

    assert check_ranking_claim(deals, "Since West came out top on deal size, why?") == ""


def test_a_reading_is_evidence_for_an_answer_not_a_replacement(deals: Path) -> None:
    """Correcting somebody is not answering them.

    LIVE, 2026-08-27: "Given Wren has the most deals in deals.csv, should we
    put her on the enterprise accounts?" came back "Wren is not top: Marek
    leads at 21 and Wren is at 16." — the premise correctly settled and the
    question left unanswered.
    """
    from interface.routes.chat import _serve_tabular_answer

    counted = collections.Counter(row["rep"] for row in _rows(deals))
    runner_up = counted.most_common(2)[1][0]
    asked = (
        f"Given {runner_up} has the most deals in {deals}, should we put them "
        "on the enterprise accounts?"
    )
    served = str(_serve_tabular_answer(asked, "Yes, they look like the obvious choice."))
    assert "is not top" in served, "the premise went unchecked"
    assert "obvious choice" in served, "the question went unanswered"


def test_a_reading_stands_alone_when_there_is_nothing_to_join(deals: Path) -> None:
    from interface.routes.chat import _serve_tabular_answer

    served = str(_serve_tabular_answer(f"how many are approved in {deals}?", ""))
    assert "approved" in served
