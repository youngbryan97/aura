#!/usr/bin/env python3
"""One place to ask what the runtime is: topology, owners, phases, counters.

Soar publishes uniform introspection commands, and the closure asked for the
same: a single API over topology, state owners, active phases, counters,
queues, resources and degradations, with machine-readable output.

Aura had all of those and they were in eight places. This is the one place,
and it is deliberately thin — every section is read from the module that owns
it, so nothing here can drift from what the runtime actually reports. A
section that raises is reported as an error in that section rather than
taking the whole answer down: an introspection tool that fails entirely when
one subsystem is unhappy is useless exactly when it is needed.

    python tools/inspect_runtime.py                  # everything, as JSON
    python tools/inspect_runtime.py --section phases # one section
    python tools/inspect_runtime.py --list           # what can be asked
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _topology() -> dict[str, Any]:
    from core.state.what_kind_of_state_is_this import how_the_state_is_organised

    return how_the_state_is_organised()


def _owners() -> dict[str, Any]:
    from core.state.who_owns_each_field import what_it_stood_at_last_time

    return what_it_stood_at_last_time()


def _phases() -> dict[str, Any]:
    from core.runtime.the_shape_of_one_turn import (
        THE_MODES,
        as_a_drawing,
        compile_the_cognition,
    )

    said: dict[str, Any] = {}
    for mode in THE_MODES:
        plan = compile_the_cognition(mode)
        said[mode] = {
            "seal": plan.seal,
            "holds": plan.holds,
            "runs": list(plan.runs),
            "refusals": list(plan.refusals),
            "drawing": as_a_drawing(plan),
        }
    return said


def _counters() -> dict[str, Any]:
    from core.runtime.what_answered_this_turn import (
        how_the_routes_have_gone,
        routes_that_have_never_answered,
    )
    from core.state.what_a_phase_changed import how_the_boundaries_have_gone

    return {
        "answer_routes": how_the_routes_have_gone(),
        "routes_that_never_answered": routes_that_have_never_answered(),
        "phase_boundaries": how_the_boundaries_have_gone(),
    }


def _resources() -> dict[str, Any]:
    from core.runtime.who_gets_it_next import (
        how_it_has_gone,
        who_holds_what,
        who_is_waiting,
    )

    return {
        "held": who_holds_what(),
        "waiting": who_is_waiting(),
        "record": how_it_has_gone(),
    }


def _the_turn() -> dict[str, Any]:
    from core.runtime.whose_turn_it_is import the_turn

    return the_turn().report()


def _threads() -> dict[str, Any]:
    from core.runtime.which_thread_may_do_this import how_it_has_gone

    return how_it_has_gone()


def _degradations() -> dict[str, Any]:
    from core.runtime.health_contract import runtime_health_report

    report = runtime_health_report()
    integrity = report.get("integrity") or {}
    return {
        "status": report.get("status"),
        "healthy": report.get("healthy"),
        "concerns": integrity.get("concerns") or [],
        "advisory": integrity.get("advisory") or [],
        "taint": integrity.get("taint"),
    }


def _organs() -> dict[str, Any]:
    from core.verify.what_each_organ_says import the_baseline

    return the_baseline()


def _measured() -> dict[str, Any]:
    from core.verify.what_has_a_measured_effect import what_it_stood_at_last_time

    return what_it_stood_at_last_time()


def _outside() -> dict[str, Any]:
    from core.verify.what_was_measured_outside import how_it_stands

    return how_it_stands()


#: Every question this answers, and where the answer comes from.
THE_SECTIONS: dict[str, Callable[[], Any]] = {
    "topology": _topology,
    "owners": _owners,
    "phases": _phases,
    "counters": _counters,
    "resources": _resources,
    "turn": _the_turn,
    "threads": _threads,
    "degradations": _degradations,
    "organs": _organs,
    "measured_effect": _measured,
    "measured_outside": _outside,
}


def inspect(section: str = "") -> dict[str, Any]:
    """Everything, or one section. A section that raises reports its error."""
    wanted = [section] if section else list(THE_SECTIONS)
    said: dict[str, Any] = {}
    for name in wanted:
        read = THE_SECTIONS.get(name)
        if read is None:
            said[name] = {"error": f"no such section; try {', '.join(THE_SECTIONS)}"}
            continue
        try:
            said[name] = read()
        except Exception as exc:  # noqa: BLE001 — one unhappy subsystem is not all of them
            said[name] = {"error": repr(exc)}
    return said


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default="", help="just this one")
    parser.add_argument("--list", action="store_true", help="what can be asked")
    args = parser.parse_args()

    if args.list:
        for name in THE_SECTIONS:
            print(name)
        return 0
    print(json.dumps(inspect(args.section), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
