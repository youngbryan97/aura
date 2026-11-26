"""An outside comparison, started on its own while a change of hers is on probation (Master Todo F16, part 9).

A change her checks cannot settle lands on probation, and stays once the outside benchmarks hold
(``her_goal_at_work.settle_probation``). Nothing started a comparison, so probation waited on a person to start one.
This starts one when every condition holds:

- a change of hers is on probation, the benchmarks have not spoken since it landed, and none was started for it;
- a baseline was recorded under the seal the harness has now, since a comparison across seals is refused;
- nothing is measuring her already;
- the machine is on its charger: on 14 November an overnight run emptied the battery;
- nobody has talked with her for as long as the baseline took, so the hours of questions fall where nobody waits on her.

The run is the harness's own: its panel and its cold restart, as many of each as the baseline asked, then its
comparison. The cold restart quits her app, and her shutdown ends the processes she started, so the run is handed to
the system by a shell that exits at once; its log is beside its script.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.agency.the_outside_bar import LEDGER, _entries, verdicts_on

logger = logging.getLogger("Aura.Agency.MeasuringOnProbation")

__all__ = ["comparison_due", "start_a_comparison_when_due"]

HOME = LEDGER.parent


def _latest_baseline(ledger: Path | None = None) -> dict[str, Any]:
    try:
        entries = _entries(ledger or LEDGER) or []
    except (OSError, ValueError, AttributeError):
        return {}
    baselines = [entry for entry in entries if entry.get("kind") == "baseline" and entry.get("run")]
    return baselines[-1] if baselines else {}


def _seal_now(root: Path) -> str:
    """The harness's seal as its own code names it: the digest of SEAL.json (outside_bar/run.py, ``seal_id``)."""
    try:
        return hashlib.sha256((root / "outside_bar" / "SEAL.json").read_bytes()).hexdigest()
    except OSError:
        return ""


def _how_long_a_run_takes(baseline: dict[str, Any]) -> float:
    """The seconds the baseline run took: from the start its meta.json names to the hour the ledger recorded it.

    Not from file times: scoring writes the meta and every row again at the end, so on 19 November all 60 files of the
    baseline carried 04:12:39, and their spread was zero. 0 when either time cannot be read."""
    try:
        meta = json.loads((HOME / "runs" / str(baseline["run"]) / "meta.json").read_text(encoding="utf-8"))
        started = datetime.strptime(str(meta["started"]), "%Y-%m-%dT%H:%M:%S%z").timestamp()
        recorded = datetime.strptime(str(baseline["at"]), "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0
    return max(recorded - started, 0.0)


def _read(argv: list[str], what: str) -> subprocess.CompletedProcess:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    return get_subprocess_gateway().run(argv, timeout=10, read_only=True, source=f"measuring_her_on_probation.{what}",
                                        accelerator_capability="none")


def _on_the_charger() -> bool:
    try:
        shown = _read(["pmset", "-g", "batt"], "charger")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return False
    return "AC Power" in shown.stdout.splitlines()[0] if shown.stdout else False


def _measuring() -> str:
    from core.agency.her_goal_at_work import MEASURING_HOLD

    if MEASURING_HOLD.exists():
        return "a run is measuring her"
    try:
        shown = _read(["pgrep", "-f", "[o]utside_bar"], "measuring")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return "whether a run of the outside bar is going could not be read"
    return "a run of the outside bar is going" if shown.stdout.strip() else ""


def _how_long_nobody_has_talked_with_her(seconds: float) -> str:
    """"" when nobody has talked with her for ``seconds``, by the policy her background work stands aside by.

    Somebody must have talked with her since she started: a fresh start with nobody yet is the morning Bryan opens her,
    not hours of quiet."""
    from core.runtime.background_policy import BackgroundPolicyProfile, background_activity_reason
    from core.runtime.service_access import resolve_orchestrator

    return background_activity_reason(resolve_orchestrator(), profile=BackgroundPolicyProfile(min_idle_seconds=seconds),
                                      needs_the_model=False)


def _rows_of(run: str) -> int:
    try:
        return sum(1 for _ in (HOME / "runs" / run / "rows").iterdir())
    except OSError:
        return 0


def _her_process() -> str:
    try:
        shown = _read(["pgrep", "-f", "[a]ura_main"], "her_process")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return ""
    return ",".join(sorted(shown.stdout.split()))


def _source_head(root: Path) -> str:
    try:
        shown = _read(["git", "-C", str(root), "-c", "core.fsmonitor=false", "rev-parse", "HEAD"], "source_head")
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return ""
    return shown.stdout.strip()


def _started_run_can_go_on(started: dict[str, Any], root: Path) -> str:
    """"" when a comparison started before, which ended without a verdict, may be resumed; otherwise why not.

    Resumed when its last attempt added rows, or when something that could change how it ends has changed since: her
    source, or her running process. A run that added nothing, with nothing changed, would end the same way again."""
    run = str(started.get("run") or "")
    if not run:
        return "the comparison started for it names no run"
    if _rows_of(run) > int(started.get("rows") or 0):
        return ""
    if _source_head(root) != str(started.get("source_head") or "") or _her_process() != str(started.get("her_process") or ""):
        return ""
    return f"the comparison run {run} added nothing when it last ran, and nothing it depends on has changed since"


def comparison_due(standing: Any, *, root: Path, ledger: Path | None = None) -> str:
    """Why a comparison is not started or resumed for ``standing`` now; "" when it is due."""
    ledger = ledger or LEDGER
    if standing is None:
        return "no change of hers is on probation"
    if verdicts_on(str(standing.offer.get("commit") or ""), root, ledger):
        return "the outside benchmarks have spoken on it"
    started = standing.offer.get("comparison_started")
    if started:
        measuring = _measuring()
        if measuring:
            return measuring
        stuck = _started_run_can_go_on(started, root)
        if stuck:
            return stuck
    baseline = _latest_baseline(ledger)
    if not baseline:
        return "the outside benchmarks have no baseline"
    if baseline.get("seal") != _seal_now(root):
        return "the baseline was made under another seal, and a comparison across seals is refused"
    measuring = _measuring()
    if measuring:
        return measuring
    if not _on_the_charger():
        return "she is not on the charger"
    takes = _how_long_a_run_takes(baseline)
    if takes <= 0.0:
        return "how long a run takes could not be read from the baseline"
    busy = _how_long_nobody_has_talked_with_her(takes)
    return f"somebody may want her: {busy}" if busy else ""


def the_chain(*, root: Path, name: str, baseline: dict[str, Any], python: str = sys.executable) -> str:
    """The run as a shell script: the panel and the cold restart as large as the baseline's, then the comparison."""
    meta = json.loads((HOME / "runs" / str(baseline["run"]) / "meta.json").read_text(encoding="utf-8"))
    log = HOME / f"compare-{name}.log"
    step = f"{shlex.quote(python)} -W ignore -m"
    return "\n".join([
        "#!/bin/zsh",
        "# Started by her own loop while a change of hers was on probation (core/agency/measuring_her_on_probation.py).",
        f"cd {shlex.quote(str(root))} || exit 1",
        "caffeinate -dimsu -w $$ &",
        f"{step} outside_bar.run ask --run {shlex.quote(name)} --each {int(meta['each'])} >> {shlex.quote(str(log))} 2>&1 "
        "|| exit $?",
        f"{step} outside_bar.cold_restart --run {shlex.quote(name)} --facts {int(meta['facts'])} >> "
        f"{shlex.quote(str(log))} 2>&1 || exit $?",
        f"{step} outside_bar.run compare --baseline {shlex.quote(str(baseline['run']))} --candidate {shlex.quote(name)} >> "
        f"{shlex.quote(str(log))} 2>&1",
        "",
    ])


def start_a_comparison_when_due() -> dict[str, Any]:
    """Start the outside comparison for her change on probation when it is due; what was done, or why not."""
    from core.agency import her_goal_at_work as goals
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    standing, root = goals.change_on_probation(), goals.source_root()
    why_not = comparison_due(standing, root=root)
    if why_not:
        return {"started": False, "why": why_not}
    started = dict(standing.offer.get("comparison_started") or {})
    baseline = _latest_baseline()
    if started.get("run"):
        # The run that ended without a verdict is resumed under its own name: each step skips what it already has.
        name, against = str(started["run"]), str(started.get("against") or baseline["run"])
        baseline = {**baseline, "run": against}
    else:
        name = f"probation-{time.strftime('%Y%m%d-%H%M')}-{standing.slug[:40]}"
    script = HOME / f"compare-{name}.sh"
    with local_internal_governed_scope("measuring_her_on_probation.start", domain="tool_execution",
                                       receipt_prefix="outside-comparison",
                                       constraints={"artifact": "aura.outside_comparison.v1", "operation": "start"}):
        get_file_write_gateway().write_text(script, the_chain(root=root, name=name, baseline=baseline),
                                            source="measuring_her_on_probation")
        # Handed to the system: the shell exits at once, so the run is no child of hers for her shutdown to end.
        get_subprocess_gateway().spawn(["/bin/zsh", "-c", f"nohup /bin/zsh {shlex.quote(str(script))} >/dev/null 2>&1 &"],
                                       cwd=str(root), source="measuring_her_on_probation",
                                       accelerator_capability="none")
    attempt = int(started.get("attempt") or 0) + 1
    standing.offer = {**standing.offer, "comparison_started": {
        "run": name, "at": time.time(), "against": baseline["run"], "attempt": attempt, "rows": _rows_of(name),
        "source_head": _source_head(root), "her_process": _her_process()}}
    goals._write_now(standing)
    logger.info("an outside comparison %s for her change on probation '%s': run %s against %s",
                "started" if attempt == 1 else f"resumed (attempt {attempt})", standing.goal[:60], name, baseline["run"])
    return {"started": True, "run": name, "against": baseline["run"], "attempt": attempt}
