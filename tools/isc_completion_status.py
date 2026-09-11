#!/usr/bin/env python3
"""Read the ISC completion list's status off the repository and the run artifacts.

The list has 778 items. A tracker that records what someone remembers doing
goes stale between one session and the next, so nothing here is asserted: each
item that claims to be done names checks, every check is executed, and an item
whose checks do not all pass is reported open however confident the note beside
it sounds.

    .venv/bin/python tools/isc_completion_status.py            # rewrite the doc
    .venv/bin/python tools/isc_completion_status.py --check    # fail on a lie

Check kinds:

    path        a file or directory exists
    grep        a regular expression matches somewhere in a file
    absent      a regular expression matches nowhere in a file
    criterion   the newest battery report passes a named criterion
    report      a Python expression over the newest report is true
    scorecard   a criterion holds on every run the scorecard read
    command     a command exits zero (`{python}` is this interpreter)
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ITEMS = REPO / "config" / "isc_completion_items.json"
EVIDENCE = REPO / "config" / "isc_completion_evidence.json"
DOC = REPO / "docs" / "ISC_COMPLETION_TODO.md"
RUNS = REPO / "artifacts" / "subject_core"


def newest_report() -> tuple[Path | None, dict[str, Any]]:
    """The newest run directory that actually holds a report, and its report."""
    best: tuple[Path, dict[str, Any]] | None = None
    if RUNS.is_dir():
        for directory in sorted(RUNS.glob("run_*")):
            report = directory / "subject_core_report.json"
            if not report.is_file():
                continue
            try:
                loaded = json.loads(report.read_text())
            except (OSError, ValueError):
                continue
            best = (directory, loaded)
    return (best[0], best[1]) if best else (None, {})


def scorecard() -> dict[str, Any]:
    path = RUNS / "scorecard.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def criteria_of(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verdict = report.get("verdict") or {}
    return {entry["criterion"]: entry for entry in verdict.get("criteria", [])}


class Checker:
    def __init__(self) -> None:
        self.run_dir, self.report = newest_report()
        self.criteria = criteria_of(self.report)
        self.card = scorecard()

    def run(self, check: dict[str, Any]) -> tuple[bool, str]:
        kind = check.get("kind")
        handler = getattr(self, f"_check_{kind}", None)
        if handler is None:
            return False, f"unknown check kind {kind!r}"
        try:
            return handler(check)
        except Exception as exc:  # a check that explodes has not passed
            return False, f"{kind} raised {exc!r}"

    def _check_path(self, check: dict[str, Any]) -> tuple[bool, str]:
        target = REPO / check["path"]
        return target.exists(), f"{check['path']} {'exists' if target.exists() else 'missing'}"

    def _check_grep(self, check: dict[str, Any]) -> tuple[bool, str]:
        target = REPO / check["path"]
        if not target.is_file():
            return False, f"{check['path']} missing"
        hit = re.search(check["pattern"], target.read_text(errors="replace"), re.MULTILINE)
        return bool(hit), f"{check['pattern']} {'found' if hit else 'not found'} in {check['path']}"

    def _check_absent(self, check: dict[str, Any]) -> tuple[bool, str]:
        target = REPO / check["path"]
        if not target.is_file():
            return False, f"{check['path']} missing"
        hit = re.search(check["pattern"], target.read_text(errors="replace"), re.MULTILINE)
        return not hit, f"{check['pattern']} {'still present' if hit else 'absent'} in {check['path']}"

    def _check_criterion(self, check: dict[str, Any]) -> tuple[bool, str]:
        name = check["criterion"]
        entry = self.criteria.get(name)
        if entry is None:
            return False, f"no run reports {name}"
        return bool(entry.get("passed")), f"{name} {'passed' if entry.get('passed') else 'failed'} in {self.run_dir.name if self.run_dir else '?'}"

    def _check_report(self, check: dict[str, Any]) -> tuple[bool, str]:
        if not self.report:
            return False, "no report to read"
        value = eval(check["expr"], {"__builtins__": {}}, {"report": self.report, "len": len, "any": any, "all": all, "sum": sum, "float": float, "int": int, "str": str, "abs": abs})
        return bool(value), f"{check['expr']} -> {value!r}"

    def _check_scorecard(self, check: dict[str, Any]) -> tuple[bool, str]:
        if not self.card:
            return False, "no scorecard"
        rows = {row["criterion"]: row for row in self.card.get("criteria", [])}
        row = rows.get(check["criterion"])
        if row is None:
            return False, f"scorecard has no {check['criterion']}"
        want = check.get("standing", "always")
        return row.get("standing") == want, f"{check['criterion']} standing {row.get('standing')!r}"

    def _check_command(self, check: dict[str, Any]) -> tuple[bool, str]:
        # `{python}` is whatever interpreter is running this, because a
        # worktree has no `.venv` of its own and a hard-coded `.venv/bin/python`
        # turns every command check into a silent failure there — which reads
        # as an item that is not done rather than as a check that could not run.
        command = check["command"].format(python=shlex.quote(sys.executable))
        done = subprocess.run(command, cwd=REPO, shell=True, capture_output=True, text=True, timeout=check.get("timeout", 600))
        return done.returncode == 0, f"exit {done.returncode}: {command}"


def evaluate() -> tuple[list[dict[str, Any]], Checker]:
    items = json.loads(ITEMS.read_text())["items"]
    evidence = json.loads(EVIDENCE.read_text()) if EVIDENCE.is_file() else {}
    checker = Checker()
    rows: list[dict[str, Any]] = []
    for item in items:
        claim = evidence.get(item["id"])
        row = dict(item)
        row["note"] = ""
        row["status"] = "open"
        row["why"] = []
        if claim:
            row["note"] = claim.get("note", "")
            checks = claim.get("checks", [])
            results = [checker.run(check) for check in checks]
            row["why"] = [message for _, message in results]
            if claim.get("status") == "not_applicable":
                row["status"] = "n/a" if all(ok for ok, _ in results) else "open"
            elif checks and all(ok for ok, _ in results):
                row["status"] = "done"
            elif claim.get("status") == "blocked":
                row["status"] = "blocked"
        rows.append(row)
    return rows, checker


MARK = {"done": "x", "open": " ", "blocked": "!", "n/a": "-"}


def render(rows: list[dict[str, Any]], checker: Checker) -> str:
    done = sum(1 for row in rows if row["status"] == "done")
    na = sum(1 for row in rows if row["status"] == "n/a")
    blocked = sum(1 for row in rows if row["status"] == "blocked")
    out: list[str] = []
    out.append("# The ISC completion list")
    out.append("")
    out.append("Every checkbox in `Aura_ISC_Completion_Master_List.pdf`, one row each, with")
    out.append("its status read off this repository rather than remembered. An item counts as")
    out.append("done only when the checks recorded beside it in")
    out.append("`config/isc_completion_evidence.json` all pass, so a note that has gone stale")
    out.append("reverts to open by itself.")
    out.append("")
    out.append("```bash")
    out.append(".venv/bin/python tools/isc_completion_status.py          # rewrite this file")
    out.append(".venv/bin/python tools/isc_completion_status.py --check  # fail if it is out of date")
    out.append("```")
    out.append("")
    out.append(f"**{done} done, {blocked} blocked, {na} not applicable, {len(rows) - done - blocked - na} open, of {len(rows)}.**")
    if checker.run_dir is not None:
        verdict = checker.report.get("verdict", {})
        out.append("")
        out.append(f"Newest run with a report: `{checker.run_dir.name}` — {verdict.get('passed', '?')}/{verdict.get('total', '?')} criteria, "
                   f"commit `{(checker.report.get('campaign') or {}).get('commit', '?')[:12]}`.")
    out.append("")
    phase = None
    section = None
    for row in rows:
        if row["phase"] != phase:
            phase = row["phase"]
            section = None
            out.append("")
            out.append(f"## Phase {phase} — {row['phase_title']}")
            out.append("")
        if row["section"] and row["section"] != section:
            section = row["section"]
            out.append("")
            out.append(f"**{section}**")
            out.append("")
        line = f"- [{MARK[row['status']]}] `{row['id']}` {row['item']}"
        if row["note"]:
            line += f" — {row['note']}"
        out.append(line)
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the doc is not what the checks say")
    parser.add_argument("--json", type=Path, help="also write the rows here")
    args = parser.parse_args()

    rows, checker = evaluate()
    text = render(rows, checker)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    if args.check:
        current = DOC.read_text() if DOC.is_file() else ""
        if current != text:
            print("docs/ISC_COMPLETION_TODO.md is out of date; run tools/isc_completion_status.py", file=sys.stderr)
            return 1
        print(f"{sum(1 for r in rows if r['status'] == 'done')}/{len(rows)} done")
        return 0
    DOC.write_text(text)
    done = sum(1 for row in rows if row["status"] == "done")
    print(f"{DOC.relative_to(REPO)}: {done}/{len(rows)} done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
