from __future__ import annotations

import argparse
from pathlib import Path

from .profiles import get_profile
from .registry import build_registry
from .report import write_report_bundle
from .simulate import run_forecast


def main() -> int:
    parser = argparse.ArgumentParser(description="Forecast Aura's long-run behavior and maintenance cliffs.")
    parser.add_argument("--profile", default="stress_load", help="Simulation profile name")
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["24h", "48h", "72h", "7d", "14d", "31d"],
        help="Forecast checkpoints such as 24h 48h 7d 31d",
    )
    parser.add_argument(
        "--output",
        default=str(Path("data") / "forecasts" / "long_run_model"),
        help="Directory for Markdown and JSON outputs",
    )
    args = parser.parse_args()

    profile = get_profile(args.profile)
    registry = build_registry()
    summary = run_forecast(profile=profile, horizons=args.horizons, registry=registry)
    outputs = write_report_bundle(summary, Path(args.output))

    print("Aura long-run forecast complete.")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
