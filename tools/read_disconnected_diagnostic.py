#!/usr/bin/env python3
"""Why her domains cut apart read as integrated: the leak, and the shared order of conditions.

On 3 November, at 120 rounds, her domains cut apart read integration (+0.0107, SE 0.0036) about as large as her
whole, and one held column moved (`S.values_foretell_choice`, in D's life). The addendum of 3 November (afternoon)
runs the same design with the clamp holding her portrait's reading, and beside it each domain alone in its own
order of conditions, and reads it by four rules, applied here to the tool's report unchanged:

  0. the fix holds: no held column moves, in any life;
  1. in the shared order, cut apart still reads integration: phi_do above its standard error;
  2. each in its own order, cut apart reads none: phi_do not above its standard error;
  3. the shared order adds integration: shared minus own order above twice the standard error of the difference.

Whole against cut apart is reported beside, deciding nothing.

    python tools/read_disconnected_diagnostic.py ~/subject-core-runs/disconnected-diagnostic/run_001
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPORT_FILE = "disconnected_aura.json"
READING_FILE = "diagnostic_reading.json"
BAR: float = 2.0


def _moved(leaks: dict[str, dict[str, float]] | None) -> dict[str, dict[str, float]]:
    return {domain: moved for domain, moved in (leaks or {}).items() if moved}


def read(report: dict[str, Any]) -> dict[str, Any]:
    """The four rules on one report of `run_disconnected_aura.py --also-shuffled`."""
    shared, own = report["disconnected"], report.get("disconnected_shuffled")
    if own is None:
        raise ValueError("the report has no life in its own order (--also-shuffled)")
    leaks = {"shared": _moved(report["held_columns_that_moved"]),
             "own": _moved(report.get("held_columns_that_moved_shuffled"))}
    fixed = not leaks["shared"] and not leaks["own"]
    shared_reads = shared["phi_do"] > max(shared["standard_error"], 0.0)
    own_reads = own["phi_do"] > max(own["standard_error"], 0.0)
    difference = shared["phi_do"] - own["phi_do"]
    joint = math.hypot(shared["standard_error"], own["standard_error"])
    adds = difference > BAR * joint
    rules = {"0_the_fix_holds": fixed, "1_shared_order_reads_integration": shared_reads,
             "2_own_order_reads_none": not own_reads, "3_shared_order_adds_integration": adds}
    whole = {label: {"phi_do": report[label]["phi_do"], "standard_error": report[label]["standard_error"],
                     "minus_cut_apart": report[label]["phi_do"] - shared["phi_do"],
                     "difference_standard_error": math.hypot(report[label]["standard_error"], shared["standard_error"])}
             for label in ("intact", "drift")}
    return {
        "rounds": report.get("rounds"),
        "evidence_source": report.get("evidence_source"),
        "held_columns_that_moved": leaks,
        "shared_order": {"phi_do": shared["phi_do"], "standard_error": shared["standard_error"]},
        "own_order": {"phi_do": own["phi_do"], "standard_error": own["standard_error"]},
        "shared_minus_own": difference,
        "difference_standard_error": joint,
        "rules": rules,
        "as_predicted": all(rules.values()),
        "whole_beside": whole,
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    run = Path(args[0]).expanduser()
    reading = read(json.loads((run / REPORT_FILE).read_text()))
    (run / READING_FILE).write_text(json.dumps(reading, indent=2))
    print(json.dumps({key: reading[key] for key in ("rounds", "rules", "as_predicted")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
