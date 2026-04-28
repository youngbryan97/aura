"""SLO comparator — load baseline, run measurements, fail on regression.

Exit code 0 iff every measurement is within ``tolerance_pct`` of its
baseline AND below any applicable hard limit.

Hard limits cap the absolute value (not just the regression delta), so
a never-before-seen environment that happens to be slow still gets
caught.

Brier-style "lower-is-better correctness" SLOs are detected by their
``unit == 'score'`` and treated as hard ceilings rather than relative
regressions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from slo.measure import run_all


def compare(
    baseline: Dict[str, Any],
    measured: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    slos = baseline.get("slos", {})
    hard_limits = baseline.get("hard_limits", {}) or {}

    results: List[Dict[str, Any]] = []
    overall_ok = True

    for name, slo in slos.items():
        baseline_value = float(slo["value"])
        tolerance_pct = float(slo.get("tolerance_pct", 50))
        unit = slo.get("unit", "ms")
        m = measured.get(name)
        if m is None:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "reason": "measurement missing",
                    "baseline": baseline_value,
                    "measured": None,
                    "unit": unit,
                }
            )
            overall_ok = False
            continue

        actual = float(m["value"])
        soft_ceiling = baseline_value * (1.0 + tolerance_pct / 100.0)

        # Lower-is-better correctness: hard limit is the ceiling, no soft tol.
        if unit == "score":
            hard = float(hard_limits.get(name, soft_ceiling))
            ok = actual <= hard
            reason = "" if ok else (
                f"correctness regressed: {actual:.6f} > hard_limit {hard:.6f}"
            )
        else:
            hard = float(hard_limits.get(name, float("inf")))
            ok_soft = actual <= soft_ceiling
            ok_hard = actual <= hard
            ok = ok_soft and ok_hard
            reason_parts: List[str] = []
            if not ok_soft:
                reason_parts.append(
                    f"soft regression: {actual:.4f} > {soft_ceiling:.4f} "
                    f"(baseline {baseline_value:.4f}, tolerance {tolerance_pct}%)"
                )
            if not ok_hard:
                reason_parts.append(
                    f"hard limit exceeded: {actual:.4f} > {hard:.4f}"
                )
            reason = "; ".join(reason_parts)

        results.append(
            {
                "name": name,
                "ok": ok,
                "reason": reason,
                "baseline": baseline_value,
                "measured": actual,
                "unit": unit,
                "soft_ceiling": soft_ceiling,
                "hard_limit": hard_limits.get(name),
            }
        )
        overall_ok = overall_ok and ok

    return {"ok": overall_ok, "results": results}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="slo/baseline.json")
    parser.add_argument(
        "--report",
        default=None,
        help="Write JSON report to this file in addition to stdout.",
    )
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"baseline missing: {baseline_path}", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    measured = run_all()
    report = compare(baseline, measured)

    print(f"SLO gate: {'PASS' if report['ok'] else 'FAIL'}")
    for r in report["results"]:
        status = "ok" if r["ok"] else "FAIL"
        print(
            f"  [{status}] {r['name']:48s} measured={r['measured']:.4f} "
            f"baseline={r['baseline']:.4f} {r['unit']}"
        )
        if not r["ok"]:
            print(f"           reason: {r['reason']}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
