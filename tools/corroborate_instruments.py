#!/usr/bin/env python3
"""tools/corroborate_instruments.py — do her two instruments agree about her?

Operationally: reads what the connectome measured between phase stations and
what the subject battery measured between state domains, and reports every
ordered pair they both looked at.

They share no code, no recording and no null. One watches the cells fire while
a turn runs; the other perturbs a state domain and watches the others move.
Where they agree, that is corroboration from two directions. Where they
disagree, one of them is measuring something other than what it is named after,
and which one is a question worth having.

Usage:

    tools/corroborate_instruments.py
    tools/corroborate_instruments.py --stations <json> --domains <csv> --json <out>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_corroboration_logs")

DEFAULT_STATIONS = "artifacts/connectome/turn_user/all_station_pairs.json"
DEFAULT_DOMAINS = "artifacts/subject_core/run_009/edges.csv"


def _newest_domain_edges(root: Path) -> Path | None:
    """The latest battery run that recorded its edges."""
    runs = sorted((root / "artifacts" / "subject_core").glob("run_*/edges.csv"))
    return runs[-1] if runs else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", default="")
    parser.add_argument("--domains", default="")
    parser.add_argument("--json", default="")
    arguments = parser.parse_args()

    from core.connectome.corroboration import (
        CORRESPONDENCE,
        corroborate_from_disk,
    )

    stations = Path(arguments.stations or REPO / DEFAULT_STATIONS)
    domains = Path(arguments.domains) if arguments.domains else (
        _newest_domain_edges(REPO) or REPO / DEFAULT_DOMAINS
    )
    if not stations.exists() or not domains.exists():
        print(f"missing a recording: {stations} / {domains}", file=sys.stderr)
        return 2

    print(f"connectome: {stations.relative_to(REPO)}")
    print(f"battery:    {domains.relative_to(REPO)}\n")

    report = corroborate_from_disk(stations, domains)
    payload = report.as_json()
    print(payload["verdict"])
    print(
        f"  both say influence: {payload['both_say_influence']}, "
        f"neither: {payload['neither_says_influence']}, "
        f"agreement {payload['rate']:.1%}"
    )
    named = [domain for domain, station in CORRESPONDENCE.items() if station]
    print(
        f"  {len(named)} of {len(CORRESPONDENCE)} domains have a station; "
        f"{', '.join(payload['unmatched_domains'])} do not, and "
        f"{', '.join(payload['unmatched_stations']) or 'no station'} has no domain\n"
    )

    if payload["disagreed"]:
        print("Where they disagree:")
        for row in payload["disagreed"]:
            print(
                f"  {row['domains']:8s} {row['stations']:34s} "
                f"connectome {row['connectome_gain']:>7.4f} carries, "
                f"battery {row['battery_effect']:>8.4f} at q={row['battery_q']:g}"
            )

    if arguments.json:
        Path(arguments.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {arguments.json}")
    return 0 if report.holds else 1


if __name__ == "__main__":
    raise SystemExit(main())
