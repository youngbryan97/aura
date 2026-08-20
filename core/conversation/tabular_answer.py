"""Answer a quantitative question about a data file by computing it.

LIVE DEFECT, 2026-08-19. Given a 60-row CSV she had already read, asked "which
team spent the most on APPROVED expenses, and how much", every draft was
rejected as ``arithmetic_answer_missing`` and the turn ended in a canned
apology. The gate was right: the question asks for a number and no draft had
one. No model sums sixty rows reliably in its head, and it should not have to
— the file is on disk and the answer is arithmetic.

This is the same remedy as file counts, belief history and queued work: where
the runtime can compute the answer exactly, it composes the reply from the
computation instead of asking the model to do it.

Nothing here is specific to a column, a file or a domain. The grouping column,
the value column and any filter are read out of the question against the
file's own header, so a table nobody has seen works the same way as one that
ships with the tests. When the question does not resolve to exactly one
reading of the table, it returns nothing and the model answers as usual — a
wrong number served with authority is worse than no number.
"""

from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["TabularAnswer", "answer_tabular_question", "describe_tabular_answer"]

_RECOVERABLE = (OSError, UnicodeDecodeError, csv.Error, ValueError, TypeError)

#: Beyond this the file is a database, not a question a chat turn answers.
_MAX_ROWS = 200_000
_MAX_BYTES = 32 * 1024 * 1024

#: What the question is asking the numbers to do.
_AGGREGATIONS: tuple[tuple[str, str], ...] = (
    ("total", r"\b(?:total|sum|altogether|combined|spend|spent)\b"),
    ("mean", r"\b(?:average|mean|typical|per\s+\w+\s+average)\b"),
    ("count", r"\b(?:how\s+many|count|number\s+of)\b"),
    ("max", r"\b(?:most|highest|largest|biggest|top|max(?:imum)?)\b"),
    ("min", r"\b(?:least|lowest|smallest|fewest|min(?:imum)?)\b"),
)

_DELIMITERS = {".csv": ",", ".tsv": "\t", ".txt": None}


@dataclass(frozen=True, slots=True)
class TabularAnswer:
    """One computed reading of a table."""

    path: str
    group_column: str
    value_column: str
    aggregation: str
    ranking: tuple[tuple[str, float], ...]
    filters: tuple[tuple[str, str], ...] = ()
    rows_used: int = 0
    rows_total: int = 0

    def leader(self) -> tuple[str, float] | None:
        return self.ranking[0] if self.ranking else None


def _numeric(value: object) -> float | None:
    text = str(value or "").strip().replace(",", "").replace("$", "").replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if path.stat().st_size > _MAX_BYTES:
        return [], []
    delimiter = _DELIMITERS.get(path.suffix.lower(), ",")
    body = path.read_text(encoding="utf-8", errors="replace")
    if delimiter is None:
        delimiter = "\t" if body.count("\t") > body.count(",") else ","
    reader = csv.DictReader(io.StringIO(body), delimiter=delimiter)
    header = [str(name or "").strip() for name in (reader.fieldnames or [])]
    if len(header) < 2:
        return [], []
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if index >= _MAX_ROWS:
            break
        rows.append({str(k or "").strip(): str(v or "") for k, v in row.items()})
    return header, rows


def _words(text: object) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9_]+", str(text or "").lower()) if word}


def _column_named_in(question: str, header: list[str]) -> list[str]:
    """Columns the question mentions, by name or by any word of their name."""
    asked = _words(question)
    named: list[str] = []
    for column in header:
        parts = _words(column)
        if parts and parts <= asked:
            named.append(column)
    return named


def _value_column(header: list[str], rows: list[dict[str, str]], question: str) -> str | None:
    """The column being measured: named if the question names one, else the
    only one that is reliably numeric."""
    numeric = [
        column
        for column in header
        if rows
        and sum(1 for row in rows[:50] if _numeric(row.get(column)) is not None)
        >= max(1, min(len(rows), 50) * 0.8)
    ]
    if not numeric:
        return None
    named = [column for column in _column_named_in(question, header) if column in numeric]
    if len(named) == 1:
        return named[0]
    return numeric[0] if len(numeric) == 1 else None


def _group_column(
    header: list[str], rows: list[dict[str, str]], question: str, value_column: str
) -> str | None:
    """The column the answer is broken down BY."""
    candidates = [
        column
        for column in _column_named_in(question, header)
        if column != value_column and _numeric(rows[0].get(column)) is None
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # "Which TEAM spent the most on APPROVED expenses" names two columns and
    # means only one of them as the breakdown. The word after which/what/by/per
    # is the thing being ranked; the other is a restriction on it. Choosing by
    # fewest distinct values instead grouped by `approved`, which has two, and
    # answered a question nobody asked.
    asked = str(question or "").lower()
    for cue in ("which", "what", "by", "per", "for each", "each"):
        match = re.search(rf"\b{re.escape(cue)}\s+([a-z0-9_]+)", asked)
        if not match:
            continue
        word = match.group(1)
        for column in candidates:
            if word in _words(column):
                return column
    # Otherwise the one with the fewest distinct values: a category, not a key.
    return min(candidates, key=lambda column: len({row.get(column, "") for row in rows}))


def _filters(
    header: list[str], rows: list[dict[str, str]], question: str, exclude: set[str]
) -> list[tuple[str, str]]:
    """Column/value pairs the question restricts the table to.

    Read off the table's own contents: "approved expenses" restricts approved
    to yes only because `approved` is a column and `yes` is one of its values.
    """
    asked = _words(question)
    found: list[tuple[str, str]] = []
    for column in header:
        if column in exclude or not (_words(column) <= asked):
            continue
        values = {str(row.get(column, "")).strip() for row in rows}
        if not 2 <= len(values) <= 8:
            continue
        matched = [value for value in values if value and _words(value) <= asked]
        if len(matched) == 1:
            found.append((column, matched[0]))
            continue
        # "APPROVED expenses" names the column and not the value, and means
        # the affirmative one. Without this the filter never applied and the
        # answer covered every row — the right breakdown of the wrong rows.
        affirmative = _affirmative_value(values)
        if affirmative and not _negated_near(question, column):
            found.append((column, affirmative))
    return found


#: Columns that answer yes or no, in the spellings data uses.
_AFFIRMATIVE = ("yes", "y", "true", "t", "1", "approved", "active", "paid", "done")
_NEGATIVE = ("no", "n", "false", "f", "0", "rejected", "inactive", "unpaid", "pending")


def _affirmative_value(values: set[str]) -> str | None:
    """The positive value of a two-valued column, if it has one."""
    present = {value.strip().lower(): value.strip() for value in values if value.strip()}
    if len(present) != 2:
        return None
    positives = [present[key] for key in present if key in _AFFIRMATIVE]
    negatives = [key for key in present if key in _NEGATIVE]
    return positives[0] if len(positives) == 1 and negatives else None


def _negated_near(question: str, column: str) -> bool:
    """True when the question asks for the column's NEGATIVE case."""
    lowered = str(question or "").lower()
    for word in _words(column):
        if re.search(rf"\b(?:not|un|non|never|without|excluding|rejected)\s*-?\s*{re.escape(word)}", lowered):
            return True
        if re.search(rf"\bun{re.escape(word)}\b", lowered):
            return True
    return False


def _aggregation(question: str) -> str:
    lowered = str(question or "").lower()
    for name, pattern in _AGGREGATIONS:
        if name in {"max", "min"} and re.search(pattern, lowered):
            return name
    for name, pattern in _AGGREGATIONS:
        if re.search(pattern, lowered):
            return name
    return "total"


def answer_tabular_question(path: str | Path, question: str) -> TabularAnswer | None:
    """Compute the reading this question asks for, or None if it is not clear.

    None is the honest result whenever the question does not resolve to one
    unambiguous reading of this table: a wrong number served with authority is
    worse than no number.
    """
    try:
        target = Path(str(path))
        if not target.is_file():
            return None
        header, rows = _read_rows(target)
        if not rows:
            return None
        value_column = _value_column(header, rows, question)
        if not value_column:
            return None
        group_column = _group_column(header, rows, question, value_column)
        if not group_column:
            return None
        applied = _filters(header, rows, question, {value_column, group_column})
        kept = [
            row
            for row in rows
            if all(str(row.get(column, "")).strip() == value for column, value in applied)
        ]
        if not kept:
            return None

        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for row in kept:
            amount = _numeric(row.get(value_column))
            if amount is None:
                continue
            key = str(row.get(group_column, "")).strip() or "(blank)"
            totals[key] += amount
            counts[key] += 1
        if not totals:
            return None

        aggregation = _aggregation(question)
        if aggregation == "mean":
            scored = {key: totals[key] / max(1, counts[key]) for key in totals}
        elif aggregation == "count":
            scored = {key: float(counts[key]) for key in counts}
        else:
            scored = dict(totals)
        ascending = aggregation == "min"
        ranking = tuple(
            sorted(scored.items(), key=lambda item: item[1], reverse=not ascending)
        )
        return TabularAnswer(
            path=str(target),
            group_column=group_column,
            value_column=value_column,
            aggregation="total" if aggregation in {"max", "min"} else aggregation,
            ranking=ranking,
            filters=tuple(applied),
            rows_used=len(kept),
            rows_total=len(rows),
        )
    except _RECOVERABLE as exc:
        record_degradation(
            "conversation.tabular_answer",
            exc,
            severity="debug",
            action="left the table question to the model",
            enforce_failure_policy=False,
        )
        return None


def describe_tabular_answer(answer: TabularAnswer | None) -> str:
    """The reading as a sentence, or "" when there is nothing to report."""
    if answer is None or not answer.ranking:
        return ""
    leader, amount = answer.ranking[0]
    unit = "" if answer.aggregation == "count" else ""
    lines = [
        f"{leader} — {amount:,.2f}{unit}"
        if answer.aggregation != "count"
        else f"{leader} — {int(amount)}"
    ]
    for key, value in answer.ranking[1:6]:
        lines.append(
            f"{key} — {value:,.2f}" if answer.aggregation != "count" else f"{key} — {int(value)}"
        )
    restriction = ""
    if answer.filters:
        restriction = " where " + ", ".join(
            f"{column} is {value}" for column, value in answer.filters
        )
    head = (
        f"By {answer.group_column}, {answer.aggregation} {answer.value_column}"
        f"{restriction} ({answer.rows_used} of {answer.rows_total} rows):"
    )
    return head + "\n- " + "\n- ".join(lines)
