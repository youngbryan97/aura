"""Many forcings in one test session (Master Todo F16, semantic repair).

Semantic repair forces a place and runs the failing tests to see whether they pass (semantic_repair.py). Each run was a
pytest process of its own: on her repository about 1.4 s to start and about 4 s in her session's fixtures, for a test
that itself takes milliseconds. LIVE 17 November 2026 the search on f5da1f5da had 88 runs in its ten minutes and spent
them before reaching the candidate that was the fix.

This is a pytest plugin. In one session it runs the session's tests once for each plan in a list it is given, the
probe's plan replaced between them (semantic_probe.py reads it), and writes each plan's verdicts as it goes, so a plan
that hangs or ends the process costs only itself: the caller starts again after it. Session fixtures are set up once.
State a test leaves in a module can carry from one plan to the next, so a plan that mends here is confirmed by a run
of its own before it is used.

It is loaded into the test process of whatever copy is being repaired, an old commit whose ``core`` may not hold this
module, so it is put on the path under a name of its own (``pytest_invocation``), as the probe is loaded by file.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The module name the plugin is loaded under in a test process.
PLUGIN = "aura_forcing_session"
#: The environment variable naming the file that holds the plans.
SPEC = "AURA_FORCINGS"
_PROBE_FILE = Path(__file__).with_name("semantic_probe.py")


def pytest_invocation(spec: Path) -> tuple[list[str], dict[str, str]]:
    """The pytest arguments and environment that load this plugin for the plans in ``spec``."""
    shim = Path(tempfile.mkdtemp(prefix="aura-forcing-"))
    (shim / f"{PLUGIN}.py").write_text(
        "import importlib.util as _u\n"
        f"_s = _u.spec_from_file_location({PLUGIN!r}, {str(Path(__file__))!r})\n"
        "_m = _u.module_from_spec(_s)\n"
        "_s.loader.exec_module(_m)\n"
        "pytest_runtestloop = _m.pytest_runtestloop\n", encoding="utf-8")
    path = os.pathsep.join(p for p in (str(shim), os.environ.get("PYTHONPATH", "")) if p)
    return ["-p", PLUGIN], {SPEC: str(spec), "PYTHONPATH": path}


def write_spec(spec: Path, *, plan: Path, steps: Sequence[Mapping[str, Any]], results: Path,
               deadline: float) -> None:
    """The plans for one session: the probe's plan file, each plan in order, where verdicts go, and when to stop.

    A raw write, not her write gateway, for the reason semantic_probe.py gives for its record: a scratch file of the
    search's own, read by a test process where her governance is not running."""
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(json.dumps({"plan": str(plan), "steps": list(steps), "results": str(results),
                                "deadline": float(deadline)}), encoding="utf-8")


def read_results(results: Path) -> list[dict[str, bool]]:
    """Each plan's verdicts, in order, as far as the session got."""
    if not results.exists():
        return []
    out = []
    for line in results.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            break
        out.append({str(k): bool(v) for k, v in dict(row.get("verdicts") or {}).items()})
    return out


def _probe() -> Any:
    probe = sys.modules.get("aura_semantic_probe")
    if probe is None:
        spec = importlib.util.spec_from_file_location("aura_semantic_probe", _PROBE_FILE)
        probe = importlib.util.module_from_spec(spec)
        sys.modules["aura_semantic_probe"] = probe
        spec.loader.exec_module(probe)
    return probe


def _passed(reports: Sequence[Any]) -> bool:
    return any(r.when == "call" and r.passed for r in reports) and not any(r.failed for r in reports)


def pytest_runtestloop(session: Any) -> Any:
    """Run the session's tests once for each plan, when a list of plans is given; otherwise pytest's own loop."""
    given = os.environ.get(SPEC, "")
    if not given or session.config.option.collectonly:
        return None
    from _pytest.runner import runtestprotocol

    spec = json.loads(Path(given).read_text(encoding="utf-8"))
    probe = _probe()
    items = list(session.items)
    steps = list(spec.get("steps") or ())
    with open(spec["results"], "a", encoding="utf-8") as results:
        for index, step in enumerate(steps):
            if time.time() >= float(spec.get("deadline") or 0.0):
                break
            probe._plans[spec["plan"]] = dict(step)
            probe._counts.clear()
            probe.FORCING = index
            verdicts = {}
            for at, item in enumerate(items):
                last = index == len(steps) - 1 and at == len(items) - 1
                if hasattr(item, "_initrequest"):
                    item._initrequest()  # its fixtures asked for afresh, as a test run again is (pytest-rerunfailures)
                reports = runtestprotocol(item, log=False, nextitem=None if last else items[(at + 1) % len(items)])
                verdicts[item.nodeid] = _passed(reports)
            results.write(json.dumps({"step": index, "verdicts": verdicts}) + "\n")
            results.flush()
    return True


__all__ = ["PLUGIN", "SPEC", "pytest_invocation", "pytest_runtestloop", "read_results", "write_spec"]
