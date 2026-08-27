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
    ("total", r"\b(?:total|sum|summed|altogether|combined|spend|spent|"
              r"adds?\s+up\s+to|added\s+up|come\s+to|comes\s+to)\b"),
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
    #: Set when the question contrasts two values of one column. Holds the
    #: column, the two values, and each group's figure on each side.
    split_column: str = ""
    sides: tuple[str, str] = ("", "")
    contrast: tuple[tuple[str, float, float], ...] = ()

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


def _value_column(
    header: list[str],
    rows: list[dict[str, str]],
    question: str,
    reserved: set[str] | None = None,
) -> str | None:
    """The column being measured: named if the question names one, else the
    only one that is reliably numeric.

    A column the question contrasts two values of is an axis, not a measure.
    Without that, "2024 vs 2025 spend by team" had two numeric columns to
    choose between, named neither, and gave up on a computable question.
    """
    held = reserved or set()
    numeric = [
        column
        for column in header
        if column not in held
        and rows
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
    header: list[str],
    rows: list[dict[str, str]],
    question: str,
    value_column: str,
    reserved: set[str] | None = None,
) -> str | None:
    """The column the answer is broken down BY."""
    held = (reserved or set()) | {value_column}
    candidates = [
        column
        for column in _column_named_in(question, header)
        if column not in held and _numeric(rows[0].get(column)) is None
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
    if found:
        return found
    # A value names its own column.
    #
    # Everything above needs the COLUMN in the question, and people do not
    # write "where status is approved" — they write "approved deals". LIVE,
    # 2026-08-27: "how many of them are approved" filtered nothing and the
    # answer covered all 83 rows when 30 were approved.
    #
    # Only when exactly one column can mean it. A word that is a value in two
    # columns is ambiguous, and an ambiguous filter is the wrong rows served
    # with authority.
    return _filter_named_only_by_value(header, rows, question, exclude)


def _filter_named_only_by_value(
    header: list[str], rows: list[dict[str, str]], question: str, exclude: set[str]
) -> list[tuple[str, str]]:
    """A single column/value pair identified by the value alone, or nothing."""
    asked = _words(question)
    candidates: list[tuple[str, str]] = []
    for column in header:
        if column in exclude:
            continue
        values = {str(row.get(column, "")).strip() for row in rows}
        if not 2 <= len(values) <= 12:
            continue
        matched = [value for value in values if value and _words(value) <= asked]
        if len(matched) == 1 and not _negated_near(question, matched[0]):
            candidates.append((column, matched[0]))
    return candidates if len(candidates) == 1 else []


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


#: Words that ask for the distance between two figures rather than their order.
_GAP_WORDS = re.compile(r"\b(?:gap|difference|differ|spread|delta|swing|apart)\b")


def _contrast_column(
    header: list[str], rows: list[dict[str, str]], question: str, exclude: set[str]
) -> tuple[str, str, str] | None:
    """The column this question contrasts two values of, and which two.

    LIVE, 2026-08-21, twice in one question. "which category has the biggest
    gap between approved and unapproved spend?" first came back as total spend
    by category — the right arithmetic for a question nobody asked, badged
    computed, and plausible because the top row is the same either way. Made
    to decline instead, the turn fell through to the model, which invented all
    three figures and served them at high confidence.

    Declining is the wrong remedy for arithmetic. The file is on disk and the
    comparison is computable, so it is computed. Two shapes are read here, and
    neither knows a domain: a question that names both values of a two-valued
    column ("2024 vs 2025"), and one that names the column and negates it
    ("approved and unapproved").
    """
    asked = _words(question)
    lowered = str(question or "").lower()
    for column in header:
        if column in exclude:
            continue
        values = [
            value
            for value in dict.fromkeys(str(row.get(column, "")).strip() for row in rows)
            if value
        ]
        if len(values) != 2:
            continue
        named = [value for value in values if _words(value) <= asked]
        if len(named) == 2:
            # Order them as the question does, so "2024 vs 2025" reads that way.
            named.sort(key=lambda value: lowered.find(value.lower()))
            return column, named[0], named[1]
        if _names_both_senses(question, column):
            affirmative = _affirmative_value(set(values))
            if affirmative:
                other = next(value for value in values if value != affirmative)
                return column, affirmative, other
    return None


def _contrast_answer(
    target: Path,
    kept: list[dict[str, str]],
    rows: list[dict[str, str]],
    question: str,
    value_column: str,
    group_column: str,
    split: tuple[str, str, str],
    applied: list[tuple[str, str]],
) -> TabularAnswer | None:
    """Each group's figure on both sides of the split, ranked by the distance."""
    column, left, right = split
    aggregation = _aggregation(question)
    sums: dict[tuple[str, str], float] = defaultdict(float)
    seen: dict[tuple[str, str], int] = defaultdict(int)
    used = 0
    for row in kept:
        side = str(row.get(column, "")).strip()
        if side not in {left, right}:
            continue
        amount = _numeric(row.get(value_column))
        if amount is None:
            continue
        key = (str(row.get(group_column, "")).strip() or "(blank)", side)
        sums[key] += amount
        seen[key] += 1
        used += 1
    groups = {key[0] for key in sums}
    if not groups:
        return None

    def figure(group: str, side: str) -> float:
        key = (group, side)
        if aggregation == "count":
            return float(seen[key])
        if aggregation == "mean":
            return sums[key] / seen[key] if seen[key] else 0.0
        return sums[key]

    rows_out = [(group, figure(group, left), figure(group, right)) for group in groups]
    by_distance = _GAP_WORDS.search(str(question or "").lower()) is not None
    rows_out.sort(
        key=lambda item: abs(item[1] - item[2]) if by_distance else item[1] - item[2],
        reverse=True,
    )
    return TabularAnswer(
        path=str(target),
        group_column=group_column,
        value_column=value_column,
        aggregation="count" if aggregation == "count" else ("mean" if aggregation == "mean" else "total"),
        ranking=tuple((group, left_figure - right_figure) for group, left_figure, right_figure in rows_out),
        filters=tuple(applied),
        rows_used=used,
        rows_total=len(rows),
        split_column=column,
        sides=(left, right),
        contrast=tuple(rows_out),
    )


def _names_both_senses(question: str, column: str) -> bool:
    """True when the question asks about a column's yes AND its no."""
    lowered = str(question or "").lower()
    if not _negated_near(lowered, column):
        return False
    return any(
        re.search(rf"\b{re.escape(word)}\b", lowered) for word in _words(column)
    )


def _negated_near(question: str, column: str) -> bool:
    """True when the question asks for the column's NEGATIVE case."""
    lowered = str(question or "").lower()
    for word in _words(column):
        if re.search(rf"\b(?:not|un|non|never|without|excluding|rejected)\s*-?\s*{re.escape(word)}", lowered):
            return True
        if re.search(rf"\bun{re.escape(word)}\b", lowered):
            return True
    return False


def _names_an_aggregation(question: str) -> bool:
    """Whether the question actually asked for one, rather than defaulting."""
    lowered = str(question or "").lower()
    return any(re.search(pattern, lowered) for _name, pattern in _AGGREGATIONS)


def _aggregation(question: str) -> str:
    """Which figure the question asks for.

    A superlative says which END of the ranking to read, not what to compute,
    and the ranking is already sorted. "Which region has the highest AVERAGE"
    means the mean per region, read from the top — checking max first made it
    the largest single row instead, and labelled the answer "total".

    So a named measure wins over a superlative, and a superlative alone still
    means the extreme value.
    """
    lowered = str(question or "").lower()
    measures = {"total", "mean", "count"}
    for name, pattern in _AGGREGATIONS:
        if name in measures and re.search(pattern, lowered):
            return name
    for name, pattern in _AGGREGATIONS:
        if re.search(pattern, lowered):
            return name
    return "total"


#: The row that stands for "all of them" when there is no breakdown.
_WHOLE_TABLE = "all rows"


def _whole_table_answer(
    target: Path,
    header: list[str],
    rows: list[dict[str, str]],
    question: str,
    value_column: str,
) -> TabularAnswer | None:
    """One figure over the whole table, filtered by what the question restricts."""
    applied = _filters(header, rows, question, {value_column})
    kept = [
        row
        for row in rows
        if all(str(row.get(column, "")).strip() == value for column, value in applied)
    ]
    if not kept:
        return None
    # "Is this file any good?" names no aggregation. _aggregation defaults to
    # "total" so a caller can ask for a breakdown without saying "total", and
    # over the WHOLE table that default invents a figure nobody asked for.
    if not _names_an_aggregation(question):
        return None
    aggregation = _aggregation(question)
    amounts = (
        [amount for row in kept if (amount := _numeric(row.get(value_column))) is not None]
        if value_column
        else []
    )
    if aggregation == "count":
        figure = float(len(kept))
    elif not amounts:
        return None
    elif aggregation == "total":
        figure = sum(amounts)
    elif aggregation == "mean":
        figure = sum(amounts) / len(amounts)
    elif aggregation == "max":
        figure = max(amounts)
    elif aggregation == "min":
        figure = min(amounts)
    else:
        # No aggregation named and no breakdown asked for: there is no single
        # reading here, and inventing one is the thing this module refuses.
        return None
    return TabularAnswer(
        path=str(target),
        group_column=_WHOLE_TABLE,
        value_column=value_column,
        aggregation=aggregation,
        ranking=((_WHOLE_TABLE, figure),),
        filters=tuple(applied),
        rows_used=len(kept),
        rows_total=len(rows),
    )


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
        # Read the contrast first: a column the question names both values of
        # is the axis being compared, so it cannot also be the measure or the
        # breakdown.
        proposed = _contrast_column(header, rows, question, set())
        reserved = {proposed[0]} if proposed else set()
        value_column = _value_column(header, rows, question, reserved)
        aggregation_asked = _aggregation(question)
        # "Which rep has the most deals" names no measure, and the only thing
        # that can be ranked without one is how often each key appears.
        counting = aggregation_asked == "count" or (
            aggregation_asked in {"max", "min"} and not value_column
        )
        if not value_column and not counting:
            return None
        # Counting rows needs no number to add up, so a count may still be
        # broken down by a column: "which rep has the most deals" is a count
        # per rep, and requiring a measure made it unanswerable.
        group_column = _group_column(header, rows, question, value_column or "", reserved)
        if not value_column and not group_column:
            return _whole_table_answer(target, header, rows, question, "")
        if not group_column:
            # An aggregate over the whole table, which is what most questions
            # about a spreadsheet actually are.
            #
            #
            # LIVE, 2026-08-27: "how many of them are approved, and what do
            # they add up to?" resolved to nothing, because a breakdown column
            # was required and there is no breakdown in that question. The
            # model then answered from a file it had not read, was correctly
            # rejected for having no numbers, and the turn ended in an apology
            # — for two figures this file settles exactly.
            return _whole_table_answer(target, header, rows, question, value_column)
        split = _contrast_column(header, rows, question, {value_column, group_column})

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
            key = str(row.get(group_column, "")).strip() or "(blank)"
            if not value_column:
                counts[key] += 1
                totals[key] += 0.0
                continue
            amount = _numeric(row.get(value_column))
            if amount is None:
                continue
            totals[key] += amount
            counts[key] += 1
        if not totals:
            return None

        aggregation = "count" if counting else aggregation_asked
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
        if split:
            return _contrast_answer(
                target, kept, rows, question, value_column, group_column, split, applied
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
    if answer.contrast:
        return _describe_contrast(answer)
    if answer.group_column == _WHOLE_TABLE:
        _leader, figure = answer.ranking[0]
        said = (
            f"{int(figure)}"
            if answer.aggregation == "count"
            else f"{figure:,.2f}"
        )
        restriction = ""
        if answer.filters:
            restriction = " where " + ", ".join(
                f"{column} is {value}" for column, value in answer.filters
            )
        subject = (
            "rows" if answer.aggregation == "count" else f"{answer.aggregation} {answer.value_column}"
        )
        return (
            f"{subject.capitalize()}{restriction}: {said} "
            f"({answer.rows_used} of {answer.rows_total} rows)."
        )
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
    measured = answer.value_column or "rows"
    head = (
        f"By {answer.group_column}, {answer.aggregation} {measured}"
        f"{restriction} ({answer.rows_used} of {answer.rows_total} rows):"
    )
    return head + "\n- " + "\n- ".join(lines)


def _describe_contrast(answer: TabularAnswer) -> str:
    """Both sides and the distance between them, so the reading is checkable."""
    left, right = answer.sides

    def figure(value: float) -> str:
        return f"{int(value)}" if answer.aggregation == "count" else f"{value:,.2f}"

    lines = [
        f"{group} — {figure(on_left)} vs {figure(on_right)} "
        f"({'+' if on_left >= on_right else '-'}{figure(abs(on_left - on_right))})"
        for group, on_left, on_right in answer.contrast[:6]
    ]
    restriction = ""
    if answer.filters:
        restriction = " where " + ", ".join(
            f"{column} is {value}" for column, value in answer.filters
        )
    head = (
        f"By {answer.group_column}, {answer.aggregation} {answer.value_column}"
        f"{restriction}, {answer.split_column} {left} vs {right}"
        f" ({answer.rows_used} of {answer.rows_total} rows):"
    )
    return head + "\n- " + "\n- ".join(lines)


#: A claim that some named thing leads on some measure: "West came out top on
#: average approved deal size", "Marek has the most deals", "South is highest".
_A_RANKING_CLAIM = re.compile(
    r"\b(?P<who>[A-Z][\w'-]*|[a-z][\w'-]{2,})\b\s+"
    r"(?:came\s+out|is|was|has|had|comes\s+out|leads?\s+on)\s+"
    r"(?:the\s+)?(?:top|highest|largest|biggest|most|best|first)\b",
    re.IGNORECASE,
)


def check_ranking_claim(path: str | Path, message: object) -> str:
    """Test a claim the message rests on against the table, or "".

    LIVE, 2026-08-27: "Since West came out top on average approved deal size in
    deals.csv, what's West doing that the other regions should copy?" She
    doubted it — correctly, the leader is South — and then reasoned from
    figures she had not looked at: "West often has deals sitting for a long
    time or getting rejected."

    A premise about a named file is checkable. Doubting one is worth much less
    than settling it, and the same enumeration that answers the question
    answers the premise.

    Returns "" whenever the claim does not resolve to one unambiguous reading,
    for the reason the rest of this module returns None: a wrong correction
    served with authority is worse than no correction.
    """
    said = str(message or "")
    claim = _A_RANKING_CLAIM.search(said)
    if not claim:
        return ""
    named = claim.group("who").strip()
    # The claim is not phrased as a question, so it is asked as one. What the
    # named thing IS tells us how to group: "West" is a value in some column,
    # and that column is the axis the claim ranks along.
    try:
        target = Path(str(path))
        header, rows = _read_rows(target)
    except _RECOVERABLE:
        return ""
    grouping = next(
        (
            column
            for column in header
            if any(str(row.get(column, "")).strip().lower() == named.lower() for row in rows)
        ),
        "",
    )
    if not grouping:
        return ""
    # Only the claim, asked as a question. Feeding the whole message back in
    # asked "which region has the Since Marek has the most deals, what is he
    # doing right?" — the rest of the sentence is about what to DO, and it is
    # not part of the claim being checked.
    stated = said[claim.start() : claim.end() + 60].split("?")[0].split(",")[0]
    answer = answer_tabular_question(path, f"which {grouping} has the {stated}")
    if answer is None or not answer.ranking or answer.group_column == _WHOLE_TABLE:
        return ""
    leader, figure = answer.ranking[0]
    if leader.strip().lower() == named.lower():
        return ""
    # Only when the thing they named is actually in this column, otherwise the
    # claim was about something this table does not hold.
    ranked = {key.strip().lower(): value for key, value in answer.ranking}
    if named.lower() not in ranked:
        return ""
    theirs = ranked[named.lower()]
    shown = (
        f"{figure:,.2f}" if answer.aggregation != "count" else f"{int(figure)}"
    )
    mine = f"{theirs:,.2f}" if answer.aggregation != "count" else f"{int(theirs)}"
    return (
        f"{named} is not top: by {answer.group_column}, {answer.aggregation} "
        f"{answer.value_column or 'rows'}, {leader} leads at {shown} and {named} "
        f"is at {mine}."
    )
