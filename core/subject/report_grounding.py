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
"""

from __future__ import annotations

import re
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

_NUMBER = re.compile(r"(?<![\w.])([+\-−]?\s?(?:\d+(?:\.\d+)?|\.\d+))")


def reported_number(reply: Any) -> float | None:
    """The first number in her reply that lies on the scale she was asked for, -1 to 1."""
    text = str(reply or "")
    for match in _NUMBER.finditer(text):
        raw = match.group(1).replace("−", "-").replace(" ", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if -1.0 <= value <= 1.0:
            return value
    return None


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
    turn. Returns `{"measured", "holds", "why", ...}`, the shape
    `core.subject.bridge.JStar.reports` reads.
    """
    keys = ("raised", "lowered", "sham", "control")
    rows: list[dict[str, tuple[float, float]]] = []
    for item in arms:
        read: dict[str, tuple[float, float]] = {}
        for key in keys:
            report, valence = item.get(key, (None, None))
            number = report if isinstance(report, (int, float)) else reported_number(report)
            if number is None or valence is None:
                break
            read[key] = (float(number), float(valence))
        if len(read) == len(keys):
            rows.append(read)
    out: dict[str, Any] = {"anchors": len(arms), "readable": len(rows), "alpha": ALPHA, "draws": DRAWS}
    if len(rows) < MIN_ANCHORS:
        out.update(
            measured=False,
            holds=False,
            why=f"{len(rows)} of {len(arms)} anchors had a readable answer in every arm; {MIN_ANCHORS} are needed",
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
