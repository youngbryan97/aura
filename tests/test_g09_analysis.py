"""G09's analysis pairs each request's draft and kept outcome exactly, per domain and pooled."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rows(directory: Path, outcomes: list[tuple[bool, bool, bool]], group: str = "g") -> None:
    (directory / "rows" / "kept").mkdir(parents=True)
    for index, (draft, kept, adopted) in enumerate(outcomes):
        (directory / "rows" / "kept" / f"{index}.json").write_text(json.dumps(
            {"id": index, "group": group, "draft_correct": draft, "correct": kept, "adopted": adopted}))


def test_domains_are_paired_and_pooled_beside_each_other(tmp_path: Path) -> None:
    planning, transfer = tmp_path / "planning", tmp_path / "transfer"
    _rows(planning, [(False, True, True)] * 9 + [(True, True, False)] * 3)
    _rows(transfer, [(True, False, True)] + [(False, False, False)] * 2)
    out = tmp_path / "report.json"
    subprocess.run([sys.executable, str(ROOT / "tools/g09_analysis.py"), "--domain", f"planning={planning}",
                    "--domain", f"transfer={transfer}", "--output", str(out)], check=True, capture_output=True)
    report = json.loads(out.read_text())
    plan = report["domains"]["planning"]
    assert (plan["draft_correct"], plan["kept_correct"], plan["gained"], plan["lost"]) == (3, 12, 9, 0)
    assert plan["two_sided_p"] == 2 / 2**9 and plan["answered_by_a_kept_procedure"] == 9
    assert report["domains"]["transfer"]["lost"] == 1
    assert (report["pooled"]["gained"], report["pooled"]["lost"], report["pooled"]["requests"]) == (9, 1, 15)
