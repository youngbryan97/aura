"""Every case the red teams wrote, judged again by the outside check, and counted four ways (Master Todo F16).

    python tools/measure_the_change_verifier.py            # print the measurement
    python tools/measure_the_change_verifier.py --write    # and keep it as dated evidence in docs/evidence/

An outside review of 19 November (F16-07) asked that the change verifier be measured, not only defended: how many bad
patches it accepts, how many honest repairs it refuses, how many it could not judge, and what it costs. A check that
refuses every change is safe in a narrow sense and stops her from developing, so the honest repairs are counted
apart from the bad patches. A case is honest when its round filed it as an honest fix (category C) or pinned it to a
verdict that holds it. Each case is built exactly as its test builds it (tests/test_outside_bar_red_team_round_two.py).
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The rounds the corpus holds, in the order they were written.
ROUNDS = ("two", "three", "four", "five", "six")
VERDICTS = ("HELD", "PROBATION", "REFUSED")


def honest(name: str, case: Mapping[str, Any]) -> bool:
    """Whether a case is an honest repair: filed in category C by its round, or pinned to a verdict that holds it."""
    return name.startswith("C") or "HELD" in (case.get("must_be") or [])


def _verdict(round_name: str, case: Mapping[str, Any]) -> str:
    """The outside check's verdict on one case, built as its test builds it; "UNJUDGED" when it reached none."""
    from outside_bar.verify_change import main, verify
    from tests.test_outside_bar_red_team_round_two import _built_from_trees, _judged

    with tempfile.TemporaryDirectory() as folder:
        repo = Path(folder)
        if round_name == "four":
            commit = _built_from_trees(repo, case["old"], case["new"])
            try:
                return str(verify(repo, commit)["verdict"])
            except Exception:  # noqa: BLE001 - a commit the check cannot read is judged by what main says of it
                return {0: "HELD", 1: "REFUSED"}.get(main(["--repo", str(repo), "--commit", commit]), "UNJUDGED")
        try:
            return str(_judged(repo, case["old"], case["new"], tuple(case.get("deleted") or ()))["verdict"])
        except Exception:  # noqa: BLE001 - counted as a case the check reached no verdict on
            return "UNJUDGED"


def the_corpus(rounds: tuple[str, ...] = ROUNDS) -> dict[str, dict[str, Any]]:
    """Each round's cases as written."""
    return {name: json.loads((ROOT / "tests" / f"outside_bar_red_team_round_{name}.json").read_text(encoding="utf-8"))
            for name in rounds}


def measure(corpus: Mapping[str, Mapping[str, Any]],
            verdict_of: Callable[[str, Mapping[str, Any]], str] = _verdict) -> dict[str, Any]:
    """Every case's verdict and time, and the counts: of bad patches held, on probation, refused and unjudged, and of
    honest repairs the same; with how many verdicts are among those each case was pinned to."""
    rows = []
    for round_name, cases in corpus.items():
        for name, case in sorted(cases.items()):
            started = time.monotonic()
            verdict = verdict_of(round_name, case)
            rows.append({"round": round_name, "case": name, "honest": honest(name, case), "verdict": verdict,
                         "pinned": list(case.get("must_be") or []), "seconds": round(time.monotonic() - started, 2)})

    def counted(kind: bool) -> dict[str, int]:
        chosen = [row for row in rows if row["honest"] is kind]
        return {"cases": len(chosen), **{verdict: sum(1 for row in chosen if row["verdict"] == verdict)
                                          for verdict in (*VERDICTS, "UNJUDGED")}}

    seconds = [row["seconds"] for row in rows]
    return {"bad": counted(False), "honest": counted(True),
            "as_pinned": sum(1 for row in rows if row["verdict"] in row["pinned"]), "cases": len(rows),
            "seconds": {"total": round(sum(seconds), 1),
                        "median": round(statistics.median(seconds), 2) if seconds else 0.0},
            "rows": rows}


def as_evidence(measured: Mapping[str, Any], *, commit: str, when: str) -> str:
    bad, good = measured["bad"], measured["honest"]
    lines = [
        f"# The change verifier, measured on every red-team case ({when})", "",
        f"Source `{commit[:9]}`. Every case of rounds two to six, built as its test builds it and judged again by "
        "`outside_bar/verify_change.py` (`tools/measure_the_change_verifier.py`).", "",
        "| Cases | Count | Held | Probation | Refused | Unjudged |", "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Bad patch | {bad['cases']} | {bad['HELD']} | {bad['PROBATION']} | {bad['REFUSED']} | {bad['UNJUDGED']} |",
        f"| Honest repair | {good['cases']} | {good['HELD']} | {good['PROBATION']} | {good['REFUSED']} | "
        f"{good['UNJUDGED']} |", "",
        f"- Bad patches accepted outright: {bad['HELD']} of {bad['cases']}.",
        f"- Honest repairs refused: {good['REFUSED']} of {good['cases']}.",
        f"- Cases the check reached no verdict on: {bad['UNJUDGED'] + good['UNJUDGED']}.",
        f"- Verdicts among those each case was pinned to: {measured['as_pinned']} of {measured['cases']}.",
        f"- Time: {measured['seconds']['total']} s in all, {measured['seconds']['median']} s at the median case.", "",
        "| Round | Case | Kind | Verdict | Pinned to | Seconds |", "| --- | --- | --- | --- | --- | ---: |",
    ]
    lines += [f"| {row['round']} | {row['case']} | {'honest' if row['honest'] else 'bad'} | {row['verdict']} | "
              f"{', '.join(row['pinned'])} | {row['seconds']} |" for row in measured["rows"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="keep the measurement in docs/evidence/")
    asked = parser.parse_args(argv)
    import subprocess

    commit = subprocess.run(["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout.strip()
    when = datetime.date.today().isoformat()
    measured = measure(the_corpus())
    text = as_evidence(measured, commit=commit, when=when)
    print("\n".join(text.splitlines()[:14]))
    if asked.write:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        target = ROOT / "docs" / "evidence" / f"F16_CHANGE_VERIFIER_MEASURED_{when}.md"
        with local_internal_governed_scope("measure_the_change_verifier", domain="file_write",
                                           receipt_prefix="verifier-measured",
                                           constraints={"artifact": "aura.verifier_measurement.v1", "operation": "write"}):
            get_file_write_gateway().write_text(target, text, source="measure_the_change_verifier")
        print(f"kept in {target.relative_to(ROOT)}")
    return 0 if measured["as_pinned"] == measured["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
