"""A question about a data file is arithmetic, so it is computed.

LIVE, 2026-08-19. Given a 60-row CSV she had already read, asked "which team
spent the most on APPROVED expenses, and how much", every draft was rejected
as arithmetic_answer_missing and the turn ended in a canned apology. The gate
was right: the question asks for a number and no draft had one. No model sums
sixty rows reliably in its head, and it should not have to.

Nothing here is specific to a column, a file or a domain. The grouping column,
the value column and any filter are read out of the question against the
file's own header, so a table nobody has seen behaves like one that ships with
the tests.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from core.conversation.tabular_answer import (
    answer_tabular_question,
    describe_tabular_answer,
)


@pytest.fixture
def expenses(tmp_path: Path) -> Path:
    path = tmp_path / "q3_expenses.csv"
    random.seed(7)
    rows = [("date", "team", "category", "amount_usd", "approved")]
    teams = ["platform", "research", "design", "ops"]
    cats = ["travel", "compute", "software", "hardware"]
    for index in range(60):
        rows.append(
            (
                f"2026-0{7 + index % 3}-{(index % 27) + 1:02d}",
                teams[index % 4],
                cats[(index * 3) % 4],
                f"{round(random.uniform(40, 4200), 2)}",
                "yes" if index % 5 else "no",
            )
        )
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def _truth(path: Path, approved_only: bool) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in csv.DictReader(path.open()):
        if approved_only and row["approved"] != "yes":
            continue
        totals[row["team"]] = totals.get(row["team"], 0.0) + float(row["amount_usd"])
    return totals


def test_the_computed_answer_matches_the_file_exactly(expenses: Path):
    answer = answer_tabular_question(
        expenses, "which team spent the most on APPROVED expenses, and how much"
    )
    assert answer is not None
    truth = _truth(expenses, approved_only=True)
    expected = max(truth.items(), key=lambda item: item[1])
    assert answer.ranking[0][0] == expected[0]
    assert abs(answer.ranking[0][1] - expected[1]) < 0.01


def test_the_word_after_which_is_the_breakdown(expenses: Path):
    """Two columns are named; only one is the grouping.

    Choosing by fewest distinct values grouped by `approved`, which has two,
    and answered a question nobody asked.
    """
    answer = answer_tabular_question(
        expenses, "which team spent the most on APPROVED expenses, and how much"
    )
    assert answer.group_column == "team"
    assert answer.value_column == "amount_usd"


def test_a_column_used_as_an_adjective_filters_to_its_affirmative(expenses: Path):
    """"APPROVED expenses" names the column, not the value."""
    answer = answer_tabular_question(
        expenses, "which team spent the most on APPROVED expenses, and how much"
    )
    assert ("approved", "yes") in answer.filters
    assert answer.rows_used < answer.rows_total


def test_without_the_adjective_every_row_counts(expenses: Path):
    answer = answer_tabular_question(expenses, "which team spent the most, and how much")
    assert answer.filters == ()
    assert answer.rows_used == answer.rows_total
    truth = _truth(expenses, approved_only=False)
    assert abs(answer.ranking[0][1] - max(truth.values())) < 0.01


def test_the_aggregation_follows_the_question(expenses: Path):
    average = answer_tabular_question(expenses, "average amount_usd by team")
    assert average.aggregation == "mean"
    counted = answer_tabular_question(expenses, "how many rows by team")
    assert counted.aggregation == "count"


def test_a_question_that_names_no_grouping_returns_nothing(expenses: Path):
    """A wrong number served with authority is worse than no number."""
    assert answer_tabular_question(expenses, "is this file any good?") is None


def test_a_file_that_is_not_a_table_returns_nothing(tmp_path: Path):
    prose = tmp_path / "notes.csv"
    prose.write_text("just some prose with no columns at all\n")
    assert answer_tabular_question(prose, "which team spent the most") is None


def test_the_description_states_what_was_counted(expenses: Path):
    described = describe_tabular_answer(
        answer_tabular_question(expenses, "which team spent the most on approved expenses")
    )
    assert "By team" in described
    assert "approved is yes" in described
    assert "of 60 rows" in described


def test_a_question_about_both_sides_of_a_column_returns_nothing(expenses: Path):
    """LIVE, 2026-08-21: asked for the gap between approved and unapproved
    spend, she served the unfiltered total by category and badged it computed.

    The negation was seen, which stopped the approved-only filter, and then no
    filter was applied at all. A question naming both senses of a column is
    asking about the difference between them, which this form cannot express.
    """
    assert (
        answer_tabular_question(
            expenses, "which team has the biggest gap between approved and unapproved spend"
        )
        is None
    )
    assert answer_tabular_question(expenses, "approved vs unapproved totals by team") is None
    # One sense alone still resolves.
    assert answer_tabular_question(expenses, "total approved spend by team") is not None
    assert answer_tabular_question(expenses, "unapproved spend by team") is not None
