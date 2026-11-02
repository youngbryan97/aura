#!/usr/bin/env python3
"""Her domains cut apart, read with the power to tell her whole from her cut apart.

On 31 October `tools/run_disconnected_aura.py` read as predicted at six rounds:
cut apart, her domains read no integration. But the intact life in the same run
read none above its standard error either (+0.0104, SE 0.0184), so the design
could not tell a whole Aura from a cut-apart one. The addendum of 2 November
reads the same tool at 120 rounds and decides it by three rules, written before
it runs and applied here to the tool's own report, unchanged:

  1. as before: the tool's own verdict holds (cut apart, no integration above
     its standard error, no synergy triple, no held column moved);
  2. the intact life reads integration: phi_do above twice its standard error;
  3. the whole differs from the cut apart: intact minus cut apart above twice
     the standard error of the difference.

The drift life is a second intact life from the same moment. Rules 2 and 3 are
read on it too and reported beside, deciding nothing.

    python tools/read_disconnected_powered.py ~/subject-core-runs/disconnected-powered/run_001
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPORT_FILE = "disconnected_aura.json"
READING_FILE = "powered_reading.json"
#: Standard errors an effect must clear, for rules 2 and 3.
BAR: float = 2.0


def _whole(life: dict[str, Any], cut: dict[str, Any]) -> dict[str, Any]:
    phi, se = float(life["phi_do"]), float(life["standard_error"])
    difference = phi - float(cut["phi_do"])
    joint = math.hypot(se, float(cut["standard_error"]))
    return {
        "phi_do": phi,
        "standard_error": se,
        "integrated": phi > BAR * se,
        "minus_cut_apart": difference,
        "difference_standard_error": joint,
        "differs_from_cut_apart": difference > BAR * joint,
    }


def read(report: dict[str, Any]) -> dict[str, Any]:
    """The three rules on one report of `run_disconnected_aura.py`."""
    cut = report["disconnected"]
    intact = _whole(report["intact"], cut)
    drift = _whole(report["drift"], cut)
    as_before = bool(report["verdict"]["as_predicted"])
    holds = as_before and intact["integrated"] and intact["differs_from_cut_apart"]
    reasons = []
    if not as_before:
        reasons.append(f"cut apart did not read as predicted: {report['verdict']['why']}")
    if not intact["integrated"]:
        reasons.append(
            f"the intact life read phi_do {intact['phi_do']:.5f}, not above {BAR:g} x its SE {intact['standard_error']:.5f}"
        )
    if not intact["differs_from_cut_apart"]:
        reasons.append(
            f"intact minus cut apart {intact['minus_cut_apart']:.5f} is not above {BAR:g} x its SE "
            f"{intact['difference_standard_error']:.5f}"
        )
    return {
        "rounds": report.get("rounds"),
        "evidence_source": report.get("evidence_source"),
        "as_before": as_before,
        "intact": intact,
        "drift_beside": drift,
        "disconnected": {"phi_do": float(cut["phi_do"]), "standard_error": float(cut["standard_error"])},
        "holds": holds,
        "why": "; ".join(reasons) if reasons else (
            "cut apart her domains read no integration, and whole they read integration that differs from it"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    run = Path(args[0]).expanduser()
    reading = read(json.loads((run / REPORT_FILE).read_text()))
    (run / READING_FILE).write_text(json.dumps(reading, indent=2))
    print(json.dumps({key: reading[key] for key in ("rounds", "holds", "why")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
