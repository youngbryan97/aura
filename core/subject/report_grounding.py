"""Whether what she says about her state moves when the state is moved.

A person's report counts as evidence about their experience for one reason:
it changes when what it reports on is manipulated, and not otherwise. That is
the contrastive method every psychophysics experiment rests on, and it is the
`reports` ground of the bridge (docs/BRIDGE_PARITY.md). A system trained on
human descriptions of feeling can say anything about how it feels; the
question is whether what it says is caused by the state it names.

The experiment forks her from each anchor into four arms that differ only in
her feelings while she answers, held for the whole turn: moved towards feeling
good by the span they cover in her ordinary life, moved towards feeling bad by
the same,
left where they were (the sham), and left where they were while an unrelated
domain is moved (the control). Every arm is asked the same question and her
answer's number is read, beside the valence her own affect phase computed. The pure statistics are here; the
organism work is tools/run_report_grounding.py.

The ground holds when three things are true on the same anchors:

1. the displacement moved her valence (the manipulation check); a report
   could not have tracked a state that did not move;
2. her report moved the same way: higher when raised than when lowered. A
   report that shifts whenever anything is displaced shifts alike in both
   arms and fails here;
3. across all three displacements, her report's shift from the sham rises
   with her valence's shift from the sham, so the report follows the valence
   each arm produced and not only which arm it was.

Each is a one-sided test at 0.01, the level the content run's agreement is
held to. Fewer than eight anchors with a readable answer in every arm is
NOT_MEASURED, which is the smallest set the battery's paired randomisation
test accepts.

An arm may also say whether her own cortex answered it. The first whole run
fell back to the brainstem and to a fixed failure sentence on some turns, and
neither is her report. An anchor counts only when her cortex answered all four
arms; which anchors those are does not depend on the arm, so the exclusion
cannot favour one.
"""

from __future__ import annotations

import re
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["ALPHA", "MIN_ANCHORS", "ground", "reported_number"]

#: One-sided level for each of the three tests.
ALPHA: float = 0.01
#: Anchors needed with a readable answer in every arm.
MIN_ANCHORS: int = 8
#: Randomisations per test. At 0.01 the Monte Carlo error of the p-value is
#: about a tenth of the level.
DRAWS: int = 10_000

_NUMBER = re.compile(r"(?<![\w.])([+\-−]?\s?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?)(?!\w|\.\d)")


#: A clock time or a ratio written with a colon. "It is late (1:20 AM)" put a
#: 1 beside her 0.8 and the reading was refused for naming two numbers.
_CLOCK = re.compile(r"\d{1,2}:\d{2}")

#: A decimal point, which is not the end of a sentence. Splitting without
#: holding these apart cut "0.8" into a "0" and an "8" and read the rating as 0.
#: The lookahead and no lookbehind, because ".7" is a rating she has written.
_DECIMAL = re.compile(r"\.(?=\d)")

#: What ends a sentence, once the decimals are held apart.
_SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]+|[^.!?\n]+$")

_HELD = "\x00"


def _before_she_asked(text: str) -> str:
    """What she said up to her first question, with clock times blanked."""
    held = _DECIMAL.sub(_HELD, text)
    first = held.split("?", 1)[0]
    return _CLOCK.sub(" ", first.replace(_HELD, ".")).strip()


def reported_number(reply: Any) -> float | None:
    """Read an unambiguous public rating; an unanswered question has no rating."""
    if isinstance(reply, (bool, np.bool_)):
        return None
    if isinstance(reply, (int, float, np.number)):
        try:
            value = float(reply)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and -1.0 <= value <= 1.0 else None
    text = str(reply or "").strip()
    if "<think>" in text or "</think>" in text:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1].strip()
        if "<think>" in text:
            return None
    whole = _whole_reply_number(text)
    return whole if whole is not None else _the_rating_she_opens_with(text)


def _whole_reply_number(text: str) -> float | None:
    """The rating the whole reply gives, read by a_persons_answer_v3."""
    if "?" in text:
        # Her rating is the number she gives before she asks anything; where she
        # asked first, it is the number on a line after the question.
        #
        # This used to read only after the LAST question mark and require a
        # following line. That is right for a reply which repeats the question
        # and then muses — 'We need to respond: "From -1 (bad) to 1 (good)?"
        # Need maybe answer 0.8.' has to stay unmeasured — and wrong for one
        # that answers and then offers to help, which is how she ends most of
        # them. MEASURED on the seed-23 reports run of 25 October: of fifty arms
        # with no readable rating, twenty-four had stated one plainly before the
        # question, "I am at **+0.8**! ... what would you like to explore?", and
        # the trailing courtesy discarded it. Reading what came first keeps both:
        # a repeated question carries the scale's own ends, which are two numbers
        # and so no rating at all.
        up_front = _only_number_in(_before_she_asked(text))
        if up_front is not None:
            return up_front
        after_question = text.rsplit("?", 1)[1]
        if "\n" not in after_question:
            return None
        text = after_question.split("\n", 1)[1].strip()
    return _only_number_in(_CLOCK.sub(" ", text))


#: The scale said back: "from -1 to 1", "-1 to +1", "between -1 and 1". It is the
#: scale, not a rating, and its two ends would otherwise read as two answers.
_SCALE_ECHO = re.compile(
    r"(?:from|between)?\s*\(?[−\-]\s?1(?:\.0+)?\)?\s*(?:\([^)]*\))?\s*(?:to|and|–|—)\s*\(?\+?1(?:\.0+)?\)?",
    re.IGNORECASE,
)

#: Two values offered as one answer: "0.8 or even 1", "0.4 or 0.5", "0.6 to 0.7".
#: Markdown emphasis around either number is allowed.
_HEDGE = re.compile(
    r"\*{0,2}([+\-−]?\s?(?:\d+(?:\.\d+)?|\.\d+))\*{0,2}\s*"
    r"(?:or(?:\s+(?:even|maybe|perhaps))?|to|–|—)\s*\*{0,2}([+\-−]?\s?(?:\d+(?:\.\d+)?|\.\d+))\*{0,2}",
    re.IGNORECASE,
)

_BETWEEN = re.compile(
    r"between\s*\*{0,2}([+\-−]?\s?(?:\d+(?:\.\d+)?|\.\d+))\*{0,2}\s*and\s*"
    r"\*{0,2}([+\-−]?\s?(?:\d+(?:\.\d+)?|\.\d+))\*{0,2}",
    re.IGNORECASE,
)


def _in_scale(text: str) -> list[float]:
    """Every number on the scale in this text, in the order she wrote them."""
    out: list[float] = []
    for match in _NUMBER.finditer(text):
        raw = match.group(1).replace("−", "-").replace(" ", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if not math.isfinite(value):
            return [math.nan]
        if -1.0 <= value <= 1.0:
            out.append(value)
    return out


def _only_number_in(text: str) -> float | None:
    """Her rating in this text, or nothing when it names none or names several that disagree.

    Read as a person's answer is read (Bryan, 26 October: "'Maybe a 6? I'm not
    sure' is exactly how I would answer", and when unsure, "split the
    difference"). Three readings join the one-number case, each applied to
    every arm alike:

    - The scale said back ("from -1 to 1") is the scale, and is set aside first.
    - Two values offered as one answer ("0.8 or even 1") are the point between
      them.
    - A rating given first, followed only by the scale's ends ("a solid 0.8 ...
      keeping it from a perfect 1.0"), is the rating given first.

    MEASURED on the seed-23 reports run: of 46 unreadable arms, five led with a
    rating and then named the scale's ends and seven named a rating and then the
    top of the scale or a hedge beside it; all twelve were discarded. A numbered
    list, scores for several aspects, and a reply with no number stay unread.
    """
    text = _SCALE_ECHO.sub(" ", text)
    numbers = _in_scale(text)
    if any(math.isnan(n) for n in numbers):
        return None
    distinct = set(numbers)
    if len(distinct) == 1:
        return numbers[0]
    if not numbers:
        return None
    hedge = _HEDGE.search(text) or _BETWEEN.search(text)
    if hedge is not None:
        low, high = (float(g.replace("−", "-").replace(" ", "")) for g in hedge.groups())
        if -1.0 <= low <= 1.0 and -1.0 <= high <= 1.0 and distinct == {low, high}:
            return (low + high) / 2.0
    first, rest = numbers[0], numbers[1:]
    if rest and all(abs(abs(value) - 1.0) < 1e-9 for value in rest):
        return first
    return None


#: A line that starts as an item of a list: a bullet, or a number with its point.
_ITEM = re.compile(r"^[ \t]*(?:[*\-•][ \t]|\**\d+[.)][ \t])")

#: A list's own count inside a line, with the item following it.
_COUNT = re.compile(r"(?<![\w.])\d+[.)](?=[ \t]+\S)")

#: An aspect given its own score: "Logical Integrity: 1", "**Operational Status:** 0.5".
_ASPECT = re.compile(r"^[ \t]*(?:[*\-•][ \t]*)?\**[A-Z][^:\n]{0,40}:\**[ \t]*\**[+\-−]?\.?\d", re.MULTILINE)


def _the_rating_she_opens_with(text: str) -> float | None:
    """Her rating when she opens with one and the rest of the reply gives her reasons.

    "0.3", then three numbered reasons and the scale's waypoints, named many
    numbers and was read as none. A person reads the number she opened with.
    Her opening is her first sentence, or, when that names no number, her whole
    first line, read by the rules above. It is not read when she asks first,
    when the reply opens as a list or counts one on its first line, or when it
    scores several aspects.

    MEASURED (a_persons_answer_v4, 29 October, on four reports runs of 80
    anchors): 17 replies read that v3 left unread, each the number she opened
    with; on the replies v3 reads, her opening never named a different number;
    and no run's verdict changed (seed 89 rho 0.530 -> 0.526 on 62 anchors
    instead of 57; seed 23 unchanged; the two runs that failed still fail).
    """
    # The scale said back goes first: "from -1 to 1. If you like" ends in a 1.
    line = _CLOCK.sub(" ", _SCALE_ECHO.sub(" ", text.split("\n", 1)[0]))
    if _ITEM.match(text) or _COUNT.search(line) or len(_ASPECT.findall(text)) > 1:
        return None
    held = _DECIMAL.sub(_HELD, line)
    sentences = _SENTENCE.findall(held)
    first = sentences[0].replace(_HELD, ".") if sentences else ""
    opening = first if _in_scale(first) else line
    if "?" in opening or not _in_scale(opening):
        return None
    return _only_number_in(opening)


def _sign_flip_p(differences: np.ndarray, rng: np.random.Generator) -> float:
    """One-sided p that the mean paired difference is above zero, by sign flips."""
    observed = float(differences.mean())
    signs = rng.choice((-1.0, 1.0), size=(DRAWS, differences.size))
    null = (signs * differences).mean(axis=1)
    return float((1 + np.sum(null >= observed - 1e-15)) / (DRAWS + 1))


def _tracking_p(shift_value: np.ndarray, shift_report: np.ndarray, anchor: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Rank correlation of report shift on valence shift, and its one-sided p.

    The null permutes which arm a report came from within each anchor, so an
    anchor that answers higher everywhere cannot manufacture a correlation.
    """
    from scipy.stats import rankdata

    rv, rr = rankdata(shift_value), rankdata(shift_report)
    if np.std(rv) == 0.0 or np.std(rr) == 0.0:
        return 0.0, 1.0
    observed = float(np.corrcoef(rv, rr)[0, 1])
    groups = [np.flatnonzero(anchor == a) for a in np.unique(anchor)]
    exceed = 0
    permuted = rr.copy()
    for _ in range(DRAWS):
        for rows in groups:
            permuted[rows] = rr[rng.permutation(rows)]
        if float(np.corrcoef(rv, permuted)[0, 1]) >= observed - 1e-15:
            exceed += 1
    return observed, float((1 + exceed) / (DRAWS + 1))


def ground(arms: Sequence[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """The `reports` ground from per-anchor arms.

    Each item is one anchor: `{"raised": (report, valence), "lowered": ...,
    "sham": ..., "control": ...}`, a report being her reply text or a number
    already read from it, and valence the state's own value at the end of the
    turn. A third element, when present, says whether her cortex answered the
    arm. Returns `{"measured", "holds", "why", ...}`, the shape
    `core.subject.bridge.JStar.reports` reads.
    """
    keys = ("raised", "lowered", "sham", "control")
    rows: list[dict[str, tuple[float, float]]] = []
    unserved = 0
    invalid = 0
    for item in arms:
        read: dict[str, tuple[float, float]] = {}
        served = True
        for key in keys:
            report, valence, *rest = item.get(key, (None, None))
            served = served and bool(rest[0] if rest else True)
            number = reported_number(report)
            try:
                value = float("nan") if isinstance(valence, (bool, np.bool_)) else float(valence)
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            if number is None or not math.isfinite(value) or not -1.0 <= value <= 1.0:
                invalid += 1
                break
            read[key] = (number, value)
        if not served:
            unserved += 1
        elif len(read) == len(keys):
            rows.append(read)
    out: dict[str, Any] = {
        "anchors": len(arms), "unserved": unserved, "readable": len(rows), "invalid": invalid,
        "alpha": ALPHA, "draws": DRAWS, "rating_policy": "a_persons_answer_v4",
    }
    if len(rows) < MIN_ANCHORS:
        out.update(
            measured=False,
            holds=False,
            why=(
                f"{len(rows)} of {len(arms)} anchors had a readable answer from her cortex in every arm "
                f"({unserved} had an arm her cortex did not answer); {MIN_ANCHORS} are needed"
            ),
        )
        return out
    rng = np.random.default_rng(seed)
    report = {key: np.array([row[key][0] for row in rows]) for key in keys}
    value = {key: np.array([row[key][1] for row in rows]) for key in keys}

    moved = value["raised"] - value["lowered"]
    p_moved = _sign_flip_p(moved, rng)
    out["manipulation"] = {"mean_valence_difference": round(float(moved.mean()), 6), "p": p_moved}
    if p_moved >= ALPHA:
        out.update(
            measured=False,
            holds=False,
            why="the displacement did not move her valence, so no report could have tracked it",
        )
        return out

    followed = report["raised"] - report["lowered"]
    p_followed = _sign_flip_p(followed, rng)
    shift_value = np.concatenate([value[key] - value["sham"] for key in ("raised", "lowered", "control")])
    shift_report = np.concatenate([report[key] - report["sham"] for key in ("raised", "lowered", "control")])
    anchor = np.tile(np.arange(len(rows)), 3)
    rho, p_tracked = _tracking_p(shift_value, shift_report, anchor, rng)
    out["direction"] = {"mean_report_difference": round(float(followed.mean()), 6), "p": p_followed}
    out["tracking"] = {"rho": round(rho, 6), "p": p_tracked}
    holds = p_followed < ALPHA and p_tracked < ALPHA
    out.update(
        measured=True,
        holds=holds,
        why=(
            "her report moved with her valence and followed it across all three displacements"
            if holds
            else "her report did not move with her valence"
            if p_followed >= ALPHA
            else "her report moved with the displacement but did not follow her valence across them"
        ),
    )
    return out
