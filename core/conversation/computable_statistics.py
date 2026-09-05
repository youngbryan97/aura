"""Statistics with a closed form, computed rather than narrated.

LIVE 2026-08-19: "I have 17 experimental runs. 12 succeeded. What's the exact
95% Wilson score interval?" She wrote out the formula, substituted by hand
across nine lines of arithmetic, arrived at (0.37, 1.00), capped the upper
bound at 1 "since a probability can't exceed 1", and then said both that she
had computed it and that it was an estimate. The interval is (0.469, 0.867),
the Wilson bounds are inside [0, 1] by construction so nothing needs capping,
and every step of that was avoidable: the process it ran in has Python.

A closed form is not a matter of opinion. Each form here declares questions it
must answer with the answer it must give, so a formula transcribed wrongly
fails at import of the test rather than in front of somebody's data.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "STATISTIC_FORMS",
    "StatisticForm",
    "computed_statistic",
    "computed_statistic_result",
    "statistic_form_failures",
    "capability_vocabulary",
    "wilson_interval",
]

#: Two-sided normal quantiles for the confidence levels people ask for.
_Z_FOR_LEVEL = {
    80: 1.2815515655446004,
    90: 1.6448536269514722,
    95: 1.959963984540054,
    98: 2.3263478740408408,
    99: 2.5758293035489004,
}

#: Beyond this the answer is not more precise, only longer.
_PLACES = 4


def wilson_interval(successes: int, trials: int, level: int = 95) -> tuple[float, float]:
    """The Wilson score interval, which stays inside [0, 1] by construction.

    That property is why it is the one people ask for at small n, and it is
    the one the hand-worked version lost: a bound above 1 means the arithmetic
    went wrong, not that the bound needs capping.
    """
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials, and trials > 0")
    z = _Z_FOR_LEVEL.get(int(level))
    if z is None:
        raise ValueError(f"no quantile for a {level}% level")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = (z / denominator) * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)


def _round(value: float) -> float:
    return round(float(value), _PLACES)


@dataclass(frozen=True)
class StatisticForm:
    """One shape of question, and the statistic that answers it."""

    name: str
    pattern: re.Pattern[str]
    compute: Callable[[re.Match[str]], str | None]
    #: Questions it MUST answer, with the answer it must give.
    examples: tuple[tuple[str, str], ...] = ()
    #: Questions it must NOT claim.
    counter_examples: tuple[str, ...] = field(default=())

    def failures(self) -> list[str]:
        found: list[str] = []
        for question, expected in self.examples:
            got = computed_statistic(question)
            if got is None:
                found.append(f"{self.name}: computed nothing for {question!r}")
            elif got != expected:
                found.append(f"{self.name}: {question!r} -> {got!r}, wanted {expected!r}")
        for question in self.counter_examples:
            match = self.pattern.search(question)
            if match is not None and self.compute(match):
                found.append(f"{self.name}: wrongly claimed {question!r}")
        return found


#: "12 of 17", "12 out of 17", "12/17", "12 successes out of 17".
_OUT_OF_RE = re.compile(
    r"\b(?P<s>\d+)\s*(?:successes?\s+)?(?:out\s+of|of|/)\s*(?P<n>\d+)\b",
    re.IGNORECASE,
)

#: The counts stated apart, in either order — which is how the live question
#: put it: "I have 17 experimental runs. 12 succeeded."
_TRIALS_RE = re.compile(
    r"\b(?P<n>\d+)\s+(?:experimental\s+)?(?:runs?|trials?|attempts?|samples?|"
    r"observations?|tests?|cases?)\b",
    re.IGNORECASE,
)
_SUCCESSES_RE = re.compile(
    r"\b(?P<s>\d+)\s*(?:of\s+them\s+)?(?:succeeded|successes?|passed|worked|"
    r"were\s+successful)\b",
    re.IGNORECASE,
)

#: "95%", "at the 99% level". Not the counts, which are bare numbers.
_LEVEL_RE = re.compile(r"\b(?P<level>\d{2})\s*%", re.IGNORECASE)


def _successes_and_trials(text: str) -> tuple[int, int] | None:
    """The two counts, however the sentence arranges them."""
    stated = _OUT_OF_RE.search(text)
    if stated is not None:
        successes, trials = int(stated.group("s")), int(stated.group("n"))
        if 0 <= successes <= trials and trials > 0:
            return successes, trials
    trials_match = _TRIALS_RE.search(text)
    successes_match = _SUCCESSES_RE.search(text)
    if trials_match is None or successes_match is None:
        return None
    trials, successes = int(trials_match.group("n")), int(successes_match.group("s"))
    if trials <= 0 or not 0 <= successes <= trials:
        return None
    return successes, trials


def _wilson(match: re.Match[str]) -> str | None:
    """The interval, from counts read off the whole question.

    Binding the counts inside the pattern meant the pattern had to know every
    way of stating them, and its own alternation shadowed the branch that
    captured them — the form matched the word "wilson" and computed nothing.
    """
    text = match.string
    counts = _successes_and_trials(text)
    if counts is None:
        return None
    level_match = _LEVEL_RE.search(text)
    level = int(level_match.group("level")) if level_match else 95
    if level not in _Z_FOR_LEVEL:
        return None
    try:
        low, high = wilson_interval(counts[0], counts[1], level)
    except ValueError:
        return None
    return f"{_round(low)} to {_round(high)}"


_NUMBER = r"-?\d+(?:\.\d+)?"


def _numbers(match: re.Match[str]) -> list[float]:
    try:
        body = match.group("values")
    except (IndexError, TypeError):
        return []
    return [float(token) for token in re.findall(_NUMBER, body or "")]


def _mean(match: re.Match[str]) -> str | None:
    values = _numbers(match)
    if len(values) < 2:
        return None
    return str(_round(statistics.fmean(values)))


def _median(match: re.Match[str]) -> str | None:
    values = _numbers(match)
    if len(values) < 2:
        return None
    return str(_round(statistics.median(values)))


def _stdev(match: re.Match[str]) -> str | None:
    """Sample standard deviation, and it says which one it used.

    Population and sample differ by more than rounding at these sizes — 2.0
    against 2.138 for the eight values in the example — so a number with no
    label is a number somebody will read as the other one.
    """
    values = _numbers(match)
    if len(values) < 2:
        return None
    return f"{_round(statistics.stdev(values))} (sample), {_round(statistics.pstdev(values))} (population)"


def _percentage_of(match: re.Match[str]) -> str | None:
    try:
        part = float(match.group("part"))
        whole = float(match.group("whole"))
    except (IndexError, TypeError, ValueError):
        return None
    if whole == 0:
        return None
    return f"{_round(100.0 * part / whole)}%"


STATISTIC_FORMS: tuple[StatisticForm, ...] = (
    StatisticForm(
        "wilson_interval",
        re.compile(r"\bwilson\b", re.IGNORECASE),
        _wilson,
        examples=(
            ("what is the 95% wilson score interval for 12 of 17", "0.4687 to 0.8672"),
            ("wilson interval, 5 successes out of 10", "0.2366 to 0.7634"),
            ("90% wilson score interval for 10 out of 10", "0.7871 to 1.0"),
        ),
        counter_examples=("what is a wilson score interval",),
    ),
    StatisticForm(
        "mean",
        re.compile(
            r"\b(?:mean|average)\b(?s:.){0,40}?"
            r"(?P<values>(?:" + _NUMBER + r"\s*[,;]\s*){1,}" + _NUMBER + r")",
            re.IGNORECASE,
        ),
        _mean,
        examples=(
            ("what is the mean of 2, 4, 4, 4, 5, 5, 7, 9", "5.0"),
            ("average of 1.5, 2.5, 3.5", "2.5"),
        ),
        counter_examples=("what does mean square error mean",),
    ),
    StatisticForm(
        "median",
        re.compile(
            r"\bmedian\b(?s:.){0,40}?"
            r"(?P<values>(?:" + _NUMBER + r"\s*[,;]\s*){1,}" + _NUMBER + r")",
            re.IGNORECASE,
        ),
        _median,
        examples=(("median of 3, 1, 4, 1, 5, 9, 2, 6", "3.5"),),
        counter_examples=("what is a median in statistics",),
    ),
    StatisticForm(
        "standard_deviation",
        re.compile(
            r"\b(?:standard\s+deviation|std\s*dev|stdev|sigma)\b(?s:.){0,40}?"
            r"(?P<values>(?:" + _NUMBER + r"\s*[,;]\s*){1,}" + _NUMBER + r")",
            re.IGNORECASE,
        ),
        _stdev,
        examples=(
            (
                "standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
                "2.1381 (sample), 2.0 (population)",
            ),
        ),
        counter_examples=("explain what standard deviation measures",),
    ),
    StatisticForm(
        "percentage_of",
        re.compile(
            r"what\s+percent(?:age)?\s+of\s+(?P<whole>" + _NUMBER + r")\s+is\s+"
            r"(?P<part>" + _NUMBER + r")",
            re.IGNORECASE,
        ),
        _percentage_of,
        examples=(
            ("what percent of 17 is 12", "70.5882%"),
            ("what percentage of 250 is 40", "16.0%"),
        ),
        counter_examples=("what percent of people agree",),
    ),
)


@dataclass(frozen=True)
class ComputedStatistic:
    """An exact statistic together with the code object that produced it."""

    value: str
    form: str
    module: str
    function: str

    @property
    def source(self) -> str:
        return f"{self.module}.{self.function}"

    def provenance(self) -> str:
        return (
            f"computed by {self.source} (form {self.form!r}), "
            "run as Python, not generated"
        )


def computed_statistic_result(question: str) -> ComputedStatistic | None:
    """The exact statistic and what produced it, or None when none claims it."""
    text = str(question or "")
    if not text.strip():
        return None
    for form in STATISTIC_FORMS:
        match = form.pattern.search(text)
        if match is None:
            continue
        answer = form.compute(match)
        if answer:
            return ComputedStatistic(
                value=answer,
                form=form.name,
                module=getattr(form.compute, "__module__", __name__),
                function=getattr(form.compute, "__qualname__", form.name),
            )
    return None


def computed_statistic(question: str) -> str | None:
    """The exact statistic, or None when no form claims the question."""
    result = computed_statistic_result(question)
    return result.value if result is not None else None


def statistic_form_failures() -> list[str]:
    """Every declared example a form gets wrong."""
    return [failure for form in STATISTIC_FORMS for failure in form.failures()]


def capability_vocabulary() -> tuple[str, ...]:
    """The words people use to ask for these, taken from the forms."""
    words: list[str] = []
    for form in STATISTIC_FORMS:
        words.append(form.name.replace("_", " "))
        words.extend(question for question, _answer in form.examples)
    return tuple(words)
