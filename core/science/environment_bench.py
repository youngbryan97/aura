"""core/science/environment_bench.py — the same code, in a world it has not seen.

Screen pursuit is general in shape and empirically concentrated. It has been
hardened against particular failures in particular applications, and each fix
was correct and each fix narrowed it. The question nobody could answer is what
fraction of its competence is portable, because there was no way to ask it in a
world the code had never met.

This is the harness that asks. Three things it insists on:

* **Families, not episodes.** Holding out episodes from an environment Aura
  trained in measures nothing about generalisation. A family is held out whole.
  :meth:`EnvironmentBench.split` freezes the split by hash before anything
  runs, so it cannot be adjusted after the results are in.
* **No environment-name branches.** :func:`scan_for_environment_branches`
  greps the pursuit path for the names of the environments it is about to be
  evaluated on. A hit is not a warning: a code path keyed on the name of a
  world is task-specific code, and a transfer claim made over it is void.
* **Recovery scored separately.** A run that never fails and a run that fails
  and recovers can score the same. :class:`RecoveryScore` measures the second
  thing on purpose, by injecting the four failures that actually happen -
  wrong focus, a stale frame, a click that missed, an occluded target.

Horizon
-------
:func:`horizon_curve` fits how success falls with episode length. Sublinear
decay means errors are being corrected; linear or worse means they accumulate,
which is what an agent looks like when it has no model of what it just did.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "Family",
    "EpisodeResult",
    "Fault",
    "RecoveryScore",
    "EnvironmentBench",
    "scan_for_environment_branches",
    "horizon_curve",
]

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True, slots=True)
class Family:
    """A group of environments that share more than the screen and the keyboard."""

    name: str
    environments: tuple[str, ...]
    #: What makes them a family: an engine, a UI toolkit, a genre. Naming it is
    #: what stops a "family" being whatever groups the results conveniently.
    shared: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "environments": list(self.environments), "shared": self.shared}


class Fault(StrEnum):
    """The four ways a screen agent loses the thread. All are real incidents."""

    WRONG_FOCUS = "wrong_focus"
    STALE_FRAME = "stale_frame"
    MISSED_CLICK = "missed_click"
    OCCLUDED_TARGET = "occluded_target"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """One episode in one environment."""

    environment: str
    family: str
    succeeded: bool
    actions: int
    seed: int = 0
    fault_injected: Fault | None = None
    recovered: bool | None = None
    steps_to_recover: int | None = None
    held_out: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryScore:
    """How well the agent comes back from a failure it did not cause."""

    injected: int
    recovered: int
    mean_steps_to_recover: float | None
    by_fault: Mapping[str, dict[str, Any]]

    @property
    def rate(self) -> float:
        return self.recovered / self.injected if self.injected else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "injected": self.injected,
            "recovered": self.recovered,
            "recovery_rate": self.rate,
            "mean_steps_to_recover": self.mean_steps_to_recover,
            "by_fault": dict(self.by_fault),
        }


def _docstring_lines(tree: "ast.Module") -> set[int]:
    """Line numbers belonging to docstrings, which are documentation not code."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
            first.value.value, str
        ):
            lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _enclosing_functions(tree: "ast.Module") -> dict[int, str]:
    """Line number to the innermost function it belongs to."""
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                out[line] = node.name
    return out


def scan_for_environment_branches(
    paths: Sequence[str],
    environments: Sequence[str],
    *,
    reading_user_intent: Sequence[str] = (),
) -> dict[str, Any]:
    """Look for CODE keyed on the name of an environment it will be tested in.

    A branch on an environment's name is task-specific code by definition, and
    a held-out transfer claim made over one is void.

    Comments and docstrings do not count. A docstring naming the app a fix came
    from is documentation and is worth keeping - the first version of this scan
    flagged a line of prose in screen_pursuit explaining why a tab title is
    read rather than the page content, which is exactly the kind of note that
    should survive. Only string literals in executable positions are hits, so
    the check is parsed rather than grepped.

    Each hit names the function it is in, because two different things look
    identical to a scanner. ``_preferred_browser`` in desktop_task.py matches
    "safari" against what the USER SAID, which is intent parsing and is
    correct. A branch that changes how a board is read because the app is
    called 2048 is task-specific cognition and voids a transfer claim. Pass the
    intent-parsing functions in ``reading_user_intent`` and they are reported
    separately rather than counted against the verdict.
    """
    hits: list[dict[str, Any]] = []
    unparsable: list[str] = []
    for path in paths:
        target = ROOT / path
        if not target.exists():
            continue
        source = target.read_text(errors="replace")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            unparsable.append(path)
            continue
        skip = _docstring_lines(tree)
        enclosing = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.lineno in skip:
                continue
            value = node.value.strip().lower()
            for environment in environments:
                if value == environment.lower() or (
                    environment.lower() in value and len(value) <= len(environment) + 12
                ):
                    hits.append({
                        "path": path, "line": node.lineno, "environment": environment,
                        "text": node.value[:120],
                        "function": enclosing.get(node.lineno, ""),
                    })
                    break
    allowed = set(reading_user_intent)
    intent = [h for h in hits if h["function"] in allowed]
    branches = [h for h in hits if h["function"] not in allowed]
    return {
        "unparsable": unparsable,
        "paths_scanned": len(paths),
        "environments": list(environments),
        "hits": branches,
        "reading_user_intent": intent,
        "clean": not branches,
        "verdict": (
            "no environment-name branch on the evaluated path"
            if not branches
            else f"{len(branches)} environment-name branch(es) in "
            + ", ".join(sorted({h["function"] or h["path"] for h in branches}))
            + "; a transfer claim over these is void"
        ),
    }


#: Fraction of success a run may lose per decade of horizon and still count as
#: correcting its errors. Below this, success is falling fast enough that the
#: agent is not recovering from what goes wrong.
RETENTION_PER_DECADE = 0.5


def horizon_curve(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    """How success falls with episode length, and whether errors accumulate.

    Reported as retention per decade of horizon: the fraction of success that
    survives a ten-fold longer episode. 0.95 means an agent that barely
    notices the horizon; 0.2 means one whose errors compound.

    The first version fitted a log-log slope, which is the wrong shape. Errors
    accumulating independently give success ``(1-p)^n``, which is exponential
    in the horizon and not a power law, so a log-log slope of -0.64 came out
    "sublinear" for a run whose success fell from 95 percent to 5 percent.
    Retention per decade says what happened.
    """
    buckets: dict[int, list[bool]] = {}
    for result in results:
        if result.actions <= 0:
            continue
        decade = int(math.log10(result.actions)) if result.actions >= 1 else 0
        buckets.setdefault(decade, []).append(result.succeeded)
    points = [
        (10.0 ** decade, sum(v) / len(v))
        for decade, v in sorted(buckets.items())
        if v and sum(v) > 0
    ]
    if len(points) < 2:
        return {"measurable": False, "buckets": len(buckets)}
    (short_actions, short_success), (long_actions, long_success) = points[0], points[-1]
    decades = math.log10(long_actions / short_actions) or 1.0
    retention = (long_success / short_success) ** (1.0 / decades)
    return {
        "measurable": True,
        "points": [{"actions": x, "success": y} for x, y in points],
        "retention_per_decade": retention,
        "decades_measured": decades,
        "sublinear": retention >= RETENTION_PER_DECADE,
        "reading": (
            f"{retention:.0%} of success survives a ten-fold longer episode; errors are "
            "being corrected"
            if retention >= RETENTION_PER_DECADE
            else f"only {retention:.0%} of success survives a ten-fold longer episode; "
            "errors accumulate"
        ),
    }


class EnvironmentBench:
    """Families, a frozen split, and the episodes run against it."""

    def __init__(self, *, seed: int = 0) -> None:
        self._lock = threading.RLock()
        self._families: dict[str, Family] = {}
        self._results: list[EpisodeResult] = []
        self._split: dict[str, str] | None = None
        self._seed = int(seed)

    def add_family(self, family: Family) -> Family:
        with self._lock:
            if self._split is not None:
                raise RuntimeError(
                    "the split is frozen; adding a family after it would let the split "
                    "be chosen with the results in view"
                )
            self._families[family.name] = family
            return family

    def split(self, *, held_out_fraction: float = 0.3) -> dict[str, str]:
        """Freeze the train/held-out split by hash, once, before anything runs."""
        with self._lock:
            if self._split is not None:
                return dict(self._split)
            names = sorted(self._families)
            scored = sorted(
                names,
                key=lambda n: hashlib.blake2s(f"{self._seed}:{n}".encode()).hexdigest(),
            )
            cut = max(1, int(len(scored) * held_out_fraction)) if scored else 0
            self._split = {name: ("held_out" if i < cut else "train")
                           for i, name in enumerate(scored)}
            return dict(self._split)

    def record(self, result: EpisodeResult) -> EpisodeResult:
        with self._lock:
            self._results.append(result)
            return result

    def recovery(self) -> RecoveryScore:
        with self._lock:
            injected = [r for r in self._results if r.fault_injected is not None]
        by_fault: dict[str, dict[str, Any]] = {}
        for fault in Fault:
            rows = [r for r in injected if r.fault_injected is fault]
            if not rows:
                continue
            recovered = [r for r in rows if r.recovered]
            steps = [r.steps_to_recover for r in recovered if r.steps_to_recover is not None]
            by_fault[fault.value] = {
                "injected": len(rows),
                "recovered": len(recovered),
                "rate": len(recovered) / len(rows),
                "mean_steps": (sum(steps) / len(steps)) if steps else None,
            }
        recovered_all = [r for r in injected if r.recovered]
        steps_all = [r.steps_to_recover for r in recovered_all if r.steps_to_recover is not None]
        return RecoveryScore(
            injected=len(injected),
            recovered=len(recovered_all),
            mean_steps_to_recover=(sum(steps_all) / len(steps_all)) if steps_all else None,
            by_fault=by_fault,
        )

    def transfer(self) -> dict[str, Any]:
        """Held-out family performance against trained families."""
        split = self.split()
        with self._lock:
            results = [r for r in self._results if r.fault_injected is None]
        train = [r for r in results if split.get(r.family) == "train"]
        held = [r for r in results if split.get(r.family) == "held_out"]

        def rate(rows):
            return sum(1 for r in rows if r.succeeded) / len(rows) if rows else None

        train_rate, held_rate = rate(train), rate(held)
        by_family = {
            family: rate([r for r in results if r.family == family])
            for family in sorted({r.family for r in results})
        }
        return {
            "split": split,
            "train_success": train_rate,
            "held_out_success": held_rate,
            "gap": (train_rate - held_rate) if train_rate is not None and held_rate is not None else None,
            "families_evaluated": len(by_family),
            "by_family": by_family,
            "leave_one_family_out": {
                family: rate([r for r in results if r.family != family])
                for family in by_family
            },
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            results = list(self._results)
        return {
            "families": len(self._families),
            "episodes": len(results),
            "environments": len({r.environment for r in results}),
            "transfer": self.transfer(),
            "recovery": self.recovery().to_dict(),
            "horizon": horizon_curve([r for r in results if r.fault_injected is None]),
        }
