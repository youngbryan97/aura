"""A progress report that cannot read its own rows overstates what is left.

The CP126 ledger silently excluded any row whose status it did not
recognise. Twenty-four rows were excluded that way:

* eleven written without a status field at all, each carrying a commit and
  a passing test — genuinely fixed findings, reported as open;
* thirteen written with ``verified_already_remediated`` or
  ``analyzed_scope_corrected``, meaningful closure claims the reader's
  status enum never listed.

None of it surfaced, because the rows the report could not read were the
same rows it could not mention. That is the recurring shape in this
codebase: the absence of a check reported as a passed check.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "closeout" / "semantic_remediation_ledger.py"


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


@pytest.fixture
def ledger_world(tmp_path):
    """An inventory of three findings and a ledger we control."""
    source_path = "core/conversation/claimed_effect.py"
    source_sha = hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest()
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "file": source_path,
                "file_sha256": source_sha,
                "findings": [
                    {
                        "finding_id": f"semantic-{n:016x}",
                        "severity": "critical",
                        "title": f"finding {n}",
                    }
                    for n in (1, 2, 3)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path, inventory, tmp_path / "LEDGER.jsonl"


def test_record_rejects_a_nonexistent_commit_without_writing(ledger_world):
    tmp_path, inventory, ledger = ledger_world
    result = _run(
        "record",
        "--finding", "semantic-0000000000000001",
        "--status", "remediated",
        "--commit", "5abfe5816dfb3311060342b35849e8228f805f92",
        "--ledger", str(ledger),
        "--inventory", str(inventory),
        cwd=ROOT,
    )

    assert result.returncode != 0
    assert "does not resolve to a git commit" in result.stderr
    assert not ledger.exists()


def test_record_canonicalizes_commit_to_full_sha(ledger_world):
    tmp_path, inventory, ledger = ledger_world
    result = _run(
        "record",
        "--finding", "semantic-0000000000000001",
        "--status", "remediated",
        "--commit", "HEAD",
        "--ledger", str(ledger),
        "--inventory", str(inventory),
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr
    entry = json.loads(ledger.read_text(encoding="utf-8"))
    assert len(entry["commit"]) == 40
    assert entry["commit"] == subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _status(world, *extra: str) -> str:
    tmp_path, inventory, ledger = world
    result = _run(
        "status", "--inventory", str(inventory), "--ledger", str(ledger), *extra,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# ------------------------------------------------------ the reported statuses


@pytest.mark.parametrize(
    "status",
    ["remediated", "verified_already_remediated", "analyzed_scope_corrected"],
)
def test_a_closure_status_actually_closes_the_finding(ledger_world, status):
    """Every status the tool ACCEPTS must also be one it COUNTS.

    ``verified_already_remediated`` was accepted by nothing and written by
    something, then dropped by the reader. A status that can be written and
    cannot be counted is a hole in the report.
    """
    tmp_path, inventory, ledger = ledger_world
    written = _run(
        "record",
        "--finding", "semantic-0000000000000001",
        "--status", status,
        "--note", "checked against current source; behaviour already correct",
        "--ledger", str(ledger),
        "--inventory", str(inventory),
        cwd=ROOT,
    )
    assert written.returncode == 0, written.stderr

    out = _status(ledger_world)
    assert "recorded closed       : 1" in out, (
        f"a finding closed as {status} was not counted as closed:\n{out}"
    )


def test_an_unreadable_row_is_named_not_dropped(ledger_world):
    """The whole defect in one assertion."""
    tmp_path, inventory, ledger = ledger_world
    ledger.write_text(
        json.dumps(
            {
                "finding_id": "semantic-0000000000000002",
                "file": "core/example.py",
                "severity": "critical",
                "commit": "abc1234",
                "evidence": "tests/test_example.py passed",
                # no status — exactly the eleven rows that vanished
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = _status(ledger_world)

    assert "semantic-0000000000000002" in out, (
        "a row the reader could not classify was excluded from the counts "
        "without being mentioned; the report looked complete because the "
        "rows it could not read were invisible"
    )
    assert "NOT counted as closed" in out


def test_a_clean_ledger_reports_no_warning(ledger_world):
    """The warning must mean something, so it must not always fire."""
    tmp_path, inventory, ledger = ledger_world
    _run(
        "record",
        "--finding", "semantic-0000000000000001",
        "--status", "remediated",
        "--note", "fixed",
        "--ledger", str(ledger),
        "--inventory", str(inventory),
        cwd=ROOT,
    )

    assert "NOT counted as closed" not in _status(ledger_world)


def test_custom_campaign_status_is_not_mislabeled_cp126(ledger_world):
    assert "CP126" not in _status(ledger_world)


def test_status_does_not_crash_on_a_row_with_no_status(ledger_world):
    """It raised TypeError comparing None to str, taking the whole report down."""
    tmp_path, inventory, ledger = ledger_world
    ledger.write_text(
        json.dumps({"finding_id": "semantic-0000000000000003", "status": None}) + "\n",
        encoding="utf-8",
    )

    result = _run(
        "status", "--inventory", str(inventory), "--ledger", str(ledger), cwd=ROOT
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------- the claim must be explained


@pytest.mark.parametrize(
    "status", ["verified_already_remediated", "analyzed_scope_corrected"]
)
def test_a_no_change_closure_must_say_why(ledger_world, status):
    """An unexplained "already fine" is indistinguishable from a skipped finding."""
    tmp_path, inventory, ledger = ledger_world
    result = _run(
        "record",
        "--finding", "semantic-0000000000000001",
        "--status", status,
        "--ledger", str(ledger),
        "--inventory", str(inventory),
        cwd=ROOT,
    )
    assert result.returncode != 0, (
        f"{status} was accepted with no explanation; it asserts no code "
        "changed, which is exactly the claim that needs a reason"
    )


# --------------------------------------------- the real ledger reads cleanly


def test_the_live_ledger_has_no_unreadable_rows():
    """Regression pin on the artifact itself, not a fixture."""
    result = _run("status", cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "NOT counted as closed" not in result.stdout, (
        "the CP126 ledger has rows the reader cannot classify again; they are "
        f"listed in this output and are being reported as open:\n{result.stdout}"
    )
