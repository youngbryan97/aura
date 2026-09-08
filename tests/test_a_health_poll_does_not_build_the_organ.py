"""Asking an organ how it is must not be what brings it into existence.

LIVE, 2026-09-08: every boot tainted the runtime with a lock-order violation —
``fsync attempted while holding
['core.runtime.health_contract._INTEGRITY_COLLECTION_LOCK']``. The integrity
collection called ``ontogeny_health_report()``, which called ``get_ontogeny()``,
which built ``OntogenyCore``, which built ``ExperienceSpine``, whose constructor
creates its store directory and fsyncs the parent. A blocking disk write, under
a process-wide lock, on the health path, because health asked a question.

The lock is deliberately held across the scan — it is the singleflight that
stops two threads scanning at once — so the fix belongs at the other end: a
health poll observes. These tests run in a fresh interpreter because the shape
only appears cold; inside a warm test process the organ is already built and
the constructor never runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_A_COLD_BOOT_IS_SLOW_S = 240


def _in_a_fresh_interpreter(body: str, tmp_path: Path) -> dict:
    """Run `body` cold and read the JSON dict it prints on its last line."""
    proc = subprocess.run(
        [sys.executable, "-c", body],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=_A_COLD_BOOT_IS_SLOW_S,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(tmp_path),
            "AURA_LOG_DIR": str(tmp_path / "logs"),
            "AURA_TESTING": "1",
        },
    )
    assert proc.returncode == 0, f"cold run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError(f"no verdict line in cold run output:\n{proc.stdout[-4000:]}")


def test_the_integrity_collection_holds_its_lock_over_no_blocking_write(tmp_path):
    """The whole class, not one subsystem: no fsync under the collection lock."""
    verdict = _in_a_fresh_interpreter(
        """
import json
from core.runtime import health_contract as hc
from core.runtime.lockdep import lockdep_report

hc._collect_integrity_snapshot()

splats = lockdep_report().get("splats") or []
offending = [
    dict(s)
    for s in splats
    if s.get("kind") == "blocking_op_under_lock"
    and any("health_contract" in str(name) for name in (s.get("held") or []))
]
print(json.dumps({"offending": offending[:4], "count": len(offending)}))
""",
        tmp_path,
    )
    assert verdict["count"] == 0, (
        "collecting the integrity block performed a blocking disk operation while "
        "holding a health lock. A health poll observes; it does not build, open or "
        f"write. Offenders: {verdict['offending']}"
    )


def test_a_cold_ontogeny_poll_reports_not_built_instead_of_building_one(tmp_path):
    verdict = _in_a_fresh_interpreter(
        """
import json
from core.ontogeny.service import ontogeny_built, ontogeny_health_report

before = ontogeny_built()
report = ontogeny_health_report()
after = ontogeny_built()
print(json.dumps({
    "built_before": before,
    "built_after": after,
    "available": report.get("available"),
    "built_field": report.get("built"),
}))
""",
        tmp_path,
    )
    assert verdict["built_before"] is False
    assert verdict["built_after"] is False, (
        "ontogeny_health_report() built the organ it was asked to describe"
    )
    assert verdict["available"] is False
    assert verdict["built_field"] is False
