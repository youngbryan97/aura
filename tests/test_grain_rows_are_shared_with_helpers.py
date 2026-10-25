"""Grain helpers claim grain rows beside the workers that score cuts.

On 24 October the carrier's coordinator made its grain rows alone at twelve
minutes a row while the workers scored their cuts, and said 1,956 minutes were
left. A helper starts as a shard worker does, takes the coordinator's anchors,
scores no cuts, and claims rows until none is left; a row is used only if it was
made from that anchor's state and doses, whoever made it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_launcher_starts_helpers_that_score_no_cuts():
    launcher = _tool("run_subject_core_v25_sharded")
    args = argparse.Namespace(workers=3, hours=1.0, grain_claims=True, share_anchors=True, grain_helpers=2)
    plan = dict(launcher.commands(args, Path("/tmp/launch")))
    helpers = {name: argv for name, argv in plan.items() if name.startswith("helper_")}
    assert sorted(helpers) == ["helper_0", "helper_1"]
    for argv in helpers.values():
        assert "--grain-helper" in argv and "--grain-claims" in argv and "--share-anchors" in argv
    workers = [argv for name, argv in plan.items() if name.startswith("worker_")]
    assert len(workers) == 3 and not any("--grain-helper" in argv for argv in workers)


def test_without_helpers_the_plan_is_unchanged():
    launcher = _tool("run_subject_core_v25_sharded")
    args = argparse.Namespace(workers=2, hours=1.0, grain_claims=True, share_anchors=True)
    assert [name for name, _argv in launcher.commands(args, Path("/tmp/launch"))] == [
        "worker_0", "worker_1", "coordinator"]


def test_the_runner_takes_the_helper_flag():
    source = (REPO / "tools" / "run_subject_core_v25.py").read_text(encoding="utf-8")
    assert '"--grain-helper"' in source
    assert "if shard is not None and not args.grain_helper:" in source
