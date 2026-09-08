"""Every arm the battery ran, written down next to the verdict that used it.

A report that gives a number for an edge and throws away the trials behind it
asks to be believed. The whole point of an interventional claim is that someone
else can take the arms apart: recompute the sign-flip test with more draws,
apply a different multiple-comparison correction, look at the pair that missed
the bar by one condition, or notice that a domain's floor is larger than its
effect. None of that is possible from a summary.

So each run directory holds the raw arms as well as the summary. One line per
trial, with the displaced arm, the sham arm, the second sham that measures the
floor, and the per-lag traces both of them produced. One row per tested pair,
kept or not, with its effect, its p, its q, and which conditions carried it.
And the null draws and the lesion arms in the same shape, because a null the
reader cannot inspect is a null they have to take on trust.

The files are written through the write gateway like everything else. They are
plain text on purpose: a reader with nothing but `cut` and `sort` should be
able to check the edge table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["write_json", "write_text", "save_arms", "save_edge_table"]

_SOURCE = "subject_core.archive"


def write_text(directory: Path, name: str, body: str) -> Path:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    target = directory / name
    with local_internal_governed_scope(_SOURCE):
        get_file_write_gateway().write_text(target, body, source=_SOURCE)
    return target


def write_json(directory: Path, name: str, payload: Any) -> Path:
    return write_text(directory, name, json.dumps(payload, indent=2, default=str))


def save_arms(directory: Path, results: Any, *, name: str = "intervention_arms.jsonl") -> Path:
    """One line per trial: what was displaced, what moved, and what moved anyway.

    ``effect`` is the displaced arm against its sham; ``floor`` is one sham
    against another, which is the same measurement with nothing done to it. A
    reader comparing those two columns is looking at the only comparison the
    edge rule makes.
    """
    lines = []
    for trial in getattr(results, "trials", []):
        lines.append(
            json.dumps(
                {
                    "source": trial.source,
                    "condition": trial.condition,
                    "trial": trial.index,
                    "took": bool(trial.took),
                    "self_effect": round(float(trial.self_effect), 6),
                    "injected_at": int(trial.injected_at),
                    "effect": {k: round(float(v), 6) for k, v in sorted(trial.effect.items())},
                    "floor": {k: round(float(v), 6) for k, v in sorted(trial.floor.items())},
                    "trace": {
                        k: [round(float(x), 6) for x in v]
                        for k, v in sorted(trial.trace.items())
                    },
                    "floor_trace": {
                        k: [round(float(x), 6) for x in v]
                        for k, v in sorted(trial.floor_trace.items())
                    },
                },
                separators=(",", ":"),
            )
        )
    header = json.dumps(
        {
            "_meta": {
                "delta": getattr(results, "delta", None),
                "lags": getattr(results, "lags", None),
                "unwritable": list(getattr(results, "unwritable", ())),
                "seconds": round(float(getattr(results, "seconds", 0.0)), 1),
                "trials": len(lines),
            }
        },
        separators=(",", ":"),
    )
    return write_text(directory, name, "\n".join([header, *lines]) + "\n")


def save_edge_table(directory: Path, tested: list[dict[str, Any]], *, name: str = "edges.csv") -> Path:
    """Every pair the battery tested, in the order it tested them."""
    rows = ["source,target,effect,p,q,replication,trials,kept,conditions"]
    for record in tested:
        rows.append(
            ",".join(
                [
                    str(record.get("source", "")),
                    str(record.get("target", "")),
                    f"{float(record.get('effect', 0.0)):.4f}",
                    f"{float(record.get('p', 1.0)):.6f}",
                    f"{float(record.get('q', 1.0)):.6f}",
                    str(record.get("replication", 0)),
                    str(record.get("trials", 0)),
                    "1" if record.get("kept") else "0",
                    " ".join(record.get("conditions", ())),
                ]
            )
        )
    return write_text(directory, name, "\n".join(rows) + "\n")
