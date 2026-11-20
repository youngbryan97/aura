"""A figure she derived from what her run returned, worked out again from it (Master Todo U07-08).

LIVE 18 November 2026, asked for the three largest Python files under core/brain with their line and function counts,
she ran a script and every file, line count and function count she gave matched. The same reply ended "the gap between
#3 and #4 is a comfortable ~75% in line count". Her run had returned 10,397 and 10,215 lines for them: 1.8%. A correct
table does not make every number beside it correct.

A percentage or a multiple in her reply is a figure derived from others. The figures it can come from are the ones her
run returned that her reply also states. It stands when her run printed it, or when a comparison of those figures gives
it at the precision she wrote it to: how much larger or smaller one is than another, what share one is of another, how
many times one is another, or what share one is of a run of them. A share of a run is taken only among figures of one
kind, and the run's output says which those are by its shape: the figures in the same column of its rows, counted from
the right. Which comparison she meant is not read from her words; every one is tried, so a figure stands if any gives
it. Where fewer than two of her stated figures came from the run there is nothing to work it out from, and it is not
judged.

What is refuted goes where every refuted sentence of hers goes (core/epistemics/what_her_record_refutes.py): taken out,
with what the record says carried to her turn asked again.

Measured 20 November 2026 on the reply above: no comparison of its ten stated figures gives 75%, so the sentence is
refuted, and a true "holds 37% of the three files' lines" stands, a share of the first three of one column. A share of
two figures of any kind was tried first and dropped: 324 functions against 106 is 75.3% of the two, and the false
sentence stood. Of the integer percentages from 1 to 200, 82 are given by some comparison of those figures, so a false
figure has about a two-in-five chance of standing by coincidence: the check misses in that direction, so as never to
take out a figure some comparison of hers supports.
"""
from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["DerivedFigure", "Recomputed", "derived_figures", "figures_her_run_does_not_give", "numbers_in"]

#: A number as written: digits with thousands separators or without, and a decimal part.
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\d,]*\d)")
#: A figure derived from others: a number followed by a percent sign, or by a multiplication sign or an "x" standing
#: alone after it ("2.6x", "3×"), and not a dimension ("1920x1080").
_DERIVED = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|×|x(?![\w]))")

#: The most figures a reply's comparisons are tried over: every ordered pair of them is a handful of quotients.
_MOST_OPERANDS = 40


@dataclass(frozen=True)
class DerivedFigure:
    """A percentage or a multiple as she wrote it, and how near the true value it must be to stand."""

    value: float
    kind: str  # "percent" or "times"
    tolerance: float
    written: str


@dataclass(frozen=True)
class Recomputed:
    """A sentence of hers with a figure no comparison of her run's figures gives; read as a refuted sentence is."""

    sentence: str
    figure: DerivedFigure
    operands: tuple[float, ...]

    def said(self) -> str:
        """What the record says, for her turn asked again."""
        shown = ", ".join(_shown(value) for value in self.operands[:8])
        return (f"No comparison of the figures my run returned that I gave ({shown}) comes to "
                f"{self.figure.written}.")


def _value(whole: str, fraction: str | None) -> float:
    return float(whole.replace(",", "") + (fraction or ""))


def _shown(value: float) -> str:
    return f"{value:,.0f}" if value == int(value) else f"{value:,}"


def numbers_in(text: str) -> list[float]:
    """Every number written in ``text``, thousands separators read as such."""
    return [_value(match.group(1), match.group(2)) for match in _NUMBER.finditer(str(text or ""))]


def _tolerance(whole: str, fraction: str | None) -> float:
    """Half a unit of the last place she wrote: 1.8 is 1.75 to 1.85, 75 is 74.5 to 75.5, and a round 70 or 300 is a
    figure rounded to its tens or hundreds."""
    if fraction:
        return 0.5 * 10 ** -(len(fraction) - 1)
    digits = whole.replace(",", "")
    trailing = len(digits) - len(digits.rstrip("0"))
    return 0.5 * 10 ** min(trailing, len(digits) - 1)


def derived_figures(sentence: str) -> list[DerivedFigure]:
    """The percentages and multiples written in ``sentence``."""
    figures = []
    for match in _DERIVED.finditer(str(sentence or "")):
        whole, fraction, sign = match.group(1), match.group(2), match.group(3)
        figures.append(DerivedFigure(_value(whole, fraction), "percent" if sign == "%" else "times",
                                     _tolerance(whole, fraction), match.group(0).strip()))
    return figures


def _comparisons(operands: Sequence[float], columns: Iterable[Sequence[float]], kind: str) -> Iterable[float]:
    """Every value a comparison of ``operands`` gives, in the unit of ``kind``: each figure against each other one, and
    for a percentage each figure's share of a run of figures in one of ``columns``."""
    for a, b in itertools.permutations(operands, 2):
        if a <= 0 or b <= 0:
            continue
        if kind == "percent":
            yield abs(a - b) / b * 100.0
            yield a / b * 100.0
        else:
            yield a / b
            yield abs(a - b) / b
    if kind != "percent":
        return
    for column in columns:
        for start, end in itertools.combinations(range(len(column) + 1), 2):
            run = column[start:end]
            whole = sum(run)
            if len(run) > 1 and whole > 0:
                yield from (a / whole * 100.0 for a in run)


def _given(figure: DerivedFigure, operands: Sequence[float], columns: Sequence[Sequence[float]]) -> bool:
    return any(abs(value - figure.value) <= figure.tolerance
               for value in _comparisons(operands, columns, figure.kind))


def _columns(returned: str) -> list[list[float]]:
    """The run's figures by the column of its rows they stand in, counted from the right, each in the order printed."""
    columns: dict[int, list[float]] = {}
    for line in returned.splitlines():
        for place, value in enumerate(reversed(numbers_in(line))):
            column = columns.setdefault(place, [])
            if value not in column:
                column.append(value)
    return [columns[place] for place in sorted(columns)]


def _what_the_run_returned(receipts: Iterable[Mapping[str, Any]]) -> str:
    """What every call that came back done returned, as the call made it."""
    returned = []
    for receipt in receipts:
        if not receipt.get("ok"):
            continue
        made = receipt.get("as_made")
        text = made.get("returned") if isinstance(made, Mapping) else None
        returned.append(str(text if text else receipt.get("observed_content") or ""))
    return "\n".join(returned)


def figures_her_run_does_not_give(sentences: Iterable[str], receipts: Iterable[Mapping[str, Any]],
                                  reply: str = "") -> list[Recomputed]:
    """Each sentence with a percentage or a multiple that no comparison of her run's figures gives.

    ``reply``: her whole reply, whose stated figures are the operands; the sentences joined when it is not given.
    """
    sentences = [str(sentence) for sentence in sentences if str(sentence or "").strip()]
    returned = _what_the_run_returned(receipts)
    printed = set(numbers_in(returned))
    if not sentences or not printed:
        return []
    stated = numbers_in(reply or " ".join(sentences))
    operands = tuple(dict.fromkeys(value for value in stated if value in printed))[:_MOST_OPERANDS]
    columns = [[value for value in column if value in operands] for column in _columns(returned)]
    found: list[Recomputed] = []
    for sentence in sentences:
        for figure in derived_figures(sentence):
            if figure.value in printed or len(operands) < 2:
                continue
            if not _given(figure, operands, columns):
                found.append(Recomputed(sentence.strip(), figure, operands))
                break
    return found
