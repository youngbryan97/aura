"""What the outside benchmarks last said about her, kept on the record of each change she lands (Master Todo F16).

The cheapest way for a self-changing system to look healthier is to prune the hard thinking its own metrics never
test, so her changes are read against benchmarks somebody else wrote, measured by a harness she cannot edit
(``outside_bar/``). This module only reads that harness's ledger. Bryan, 18 November 2026: she goes on editing and
pushing live while this is watched over time, so what it says is written down with each landing and stops none of them.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["LEDGER", "verdicts_on", "verdicts_since", "what_the_outside_bar_says"]

LEDGER = Path("~/.aura/outside-bar/ledger.jsonl").expanduser()


def _entries(ledger: Path) -> list[dict[str, Any]] | None:
    """The ledger's lines, or None when one of them is not what the line after it recorded."""
    lines = ledger.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    before = ""
    for line in lines:
        entry = json.loads(line)
        if entry.get("previous") != before:
            return None
        before = hashlib.sha256(line.encode("utf-8")).hexdigest()
        entries.append(entry)
    return entries


def what_the_outside_bar_says(ledger: Path = LEDGER) -> str:
    """What stands against her changes on the outside benchmarks, in words for her record; empty when nothing does."""
    try:
        entries = _entries(ledger)
    except FileNotFoundError:
        return "the outside benchmarks have no baseline yet"
    except (OSError, ValueError, AttributeError):
        return "the outside benchmarks' ledger cannot be read"
    if entries is None:
        return "the outside benchmarks' ledger has been altered"
    if not any(entry.get("kind") == "baseline" for entry in entries):
        return "the outside benchmarks have no baseline yet"
    decided = [entry for entry in entries if entry.get("kind") == "compare" and entry.get("verdict") in ("HELD", "REGRESSED")]
    if decided and decided[-1]["verdict"] == "REGRESSED":
        return f"the outside benchmarks regressed in run {decided[-1].get('candidate')}"
    return ""


def _run_commit(home: Path, run: str) -> str:
    """The source commit a run of the outside bar measured, from its own meta.json; "" when that cannot be read."""
    try:
        return str(json.loads((home / "runs" / run / "meta.json").read_text(encoding="utf-8")).get("commit") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def _contains(root: Path, commit: str, measured: str) -> bool | None:
    """Whether the source ``measured`` holds ``commit``; None when git cannot say (a commit it does not have)."""
    import subprocess

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    try:
        done = get_subprocess_gateway().run(["git", "-c", "core.fsmonitor=false", "merge-base", "--is-ancestor", commit,
                                             measured], cwd=root, timeout=30, read_only=True,
                                            source="the_outside_bar.contains", accelerator_capability="none")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None
    return {0: True, 1: False}.get(done.returncode)


def verdicts_on(commit: str, root: Path, ledger: Path = LEDGER) -> list[str]:
    """The outside benchmarks' decided verdicts on ``commit``, oldest first: comparisons whose candidate run measured
    source that holds it and whose baseline run measured source that does not.

    A verdict recorded after a change landed is not a verdict on it. A run started before it landed measured the source
    without it, and a comparison of two runs that both hold it says nothing about it."""
    if not commit:
        return []
    try:
        entries = _entries(ledger) or []
    except (OSError, ValueError, AttributeError):
        return []
    home = ledger.parent
    found = []
    for entry in entries:
        if entry.get("kind") != "compare" or entry.get("verdict") not in ("HELD", "REGRESSED"):
            continue
        measured, before = _run_commit(home, str(entry.get("candidate") or "")), _run_commit(home, str(entry.get("baseline") or ""))
        if measured and before and _contains(root, commit, measured) is True and _contains(root, commit, before) is False:
            found.append(str(entry["verdict"]))
    return found


def other_changes_measured_with(commit: str, root: Path, ledger: Path = LEDGER) -> list[str]:
    """The other commits to what she runs that the latest decided comparison on ``commit`` measured with it: those its
    candidate's source holds and its baseline's does not, ``commit`` aside, that change a file other than documentation
    or a test. [] when there are none, or when they cannot be read.

    A comparison measures her source as it stands when it runs, so a change on probation is measured beside whatever
    landed since the baseline. LIVE 19 November 2026 a commit of mine took MATH-500 from 9 of 12 to 2 between two
    baselines; a change of hers on probation then would have been taken back for it, and its record would not have
    said that anything else was measured with it (F16-08)."""
    if not commit:
        return []
    try:
        entries = _entries(ledger) or []
    except (OSError, ValueError, AttributeError):
        return []
    home = ledger.parent
    for entry in reversed(entries):
        if entry.get("kind") != "compare" or entry.get("verdict") not in ("HELD", "REGRESSED"):
            continue
        measured, before = _run_commit(home, str(entry.get("candidate") or "")), _run_commit(home, str(entry.get("baseline") or ""))
        if measured and before and _contains(root, commit, measured) is True and _contains(root, commit, before) is False:
            return _code_commits_between(root, before, measured, leaving_out=commit)
    return []


def _code_commits_between(root: Path, before: str, measured: str, *, leaving_out: str) -> list[str]:
    import subprocess

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    try:
        done = get_subprocess_gateway().run(["git", "-c", "core.fsmonitor=false", "log", "--format=@%H", "--name-only",
                                             f"{before}..{measured}"], cwd=root, timeout=60, read_only=True,
                                            source="the_outside_bar.measured_with", accelerator_capability="none")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    found: list[str] = []
    current, runs_code = "", False
    for line in [*str(done.stdout or "").splitlines(), "@"]:
        if line.startswith("@"):
            if current and runs_code and not leaving_out.startswith(current) and not current.startswith(leaving_out):
                found.append(current)
            current, runs_code = line[1:].strip(), False
        elif line.strip() and not (line.startswith(("docs/", "tests/")) or line.endswith(".md")):
            runs_code = True
    return found


def verdicts_since(moment: float, ledger: Path = LEDGER) -> list[str]:
    """The outside benchmarks' decided verdicts recorded after ``moment``, oldest first."""
    try:
        entries = _entries(ledger) or []
    except (OSError, ValueError, AttributeError):
        return []
    return [str(entry["verdict"]) for entry in entries if entry.get("kind") == "compare" and entry.get("verdict") in ("HELD", "REGRESSED")
            and datetime.datetime.strptime(str(entry.get("at")), "%Y-%m-%dT%H:%M:%S%z").timestamp() > moment]
