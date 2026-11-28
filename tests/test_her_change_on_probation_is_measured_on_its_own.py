"""While a change of hers is on probation, the outside benchmarks are asked again on their own (Master Todo F16).

A change her checks cannot settle stays on probation until the outside benchmarks speak, and nothing started them, so
probation waited on a person. A comparison now starts itself, only when nobody waits on her and the run can count.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency import measuring_her_on_probation as measuring


def _ledger(path: Path, *entries: dict) -> Path:
    lines, before = [], ""
    for entry in entries:
        line = json.dumps({**entry, "previous": before}, sort_keys=True)
        before = hashlib.sha256(line.encode("utf-8")).hexdigest()
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def ready(tmp_path, monkeypatch):
    """Every condition holding: a baseline under today's seal, nothing measuring, on the charger, hours of quiet."""
    root = tmp_path / "her-source"
    (root / "outside_bar").mkdir(parents=True)
    (root / "outside_bar" / "SEAL.json").write_text('{"files": {}}')
    seal = hashlib.sha256((root / "outside_bar" / "SEAL.json").read_bytes()).hexdigest()
    home = tmp_path / "outside-bar"
    run = home / "runs" / "baseline-1"
    (run / "rows").mkdir(parents=True)
    # The baseline of 19 November: started 23:43:52, recorded 04:12:39, and scoring wrote every file again at the end.
    (run / "meta.json").write_text(json.dumps({"each": 12, "facts": 12, "seal": seal,
                                               "started": "2026-11-18T23:43:52-0800"}))
    (run / "rows" / "a.json").write_text("{}")
    ledger = _ledger(home / "ledger.jsonl", {"kind": "baseline", "run": "baseline-1", "seal": seal,
                                              "at": "2026-11-19T04:12:39-0800"})
    monkeypatch.setattr(measuring, "HOME", home)
    monkeypatch.setattr(measuring, "LEDGER", ledger)
    monkeypatch.setattr(measuring, "_measuring", lambda: "")
    monkeypatch.setattr(measuring, "_on_the_charger", lambda: True)
    monkeypatch.setattr(measuring, "verdicts_on", lambda commit, root, ledger=None: [])
    monkeypatch.setattr(measuring, "_source_head", lambda root: "head-1")
    monkeypatch.setattr(measuring, "_her_process", lambda: "4242")
    quiet_for: list[float] = []
    monkeypatch.setattr(measuring, "_how_long_nobody_has_talked_with_her",
                        lambda seconds: (quiet_for.append(seconds), "")[1])
    standing = SimpleNamespace(slug="mend-x", goal="Mend x",
                               offer={"merged_at": time.time() - 60, "on_probation": True, "commit": "c0ffee"})
    return SimpleNamespace(root=root, ledger=ledger, home=home, seal=seal, standing=standing, quiet_for=quiet_for)


TOOK = 4 * 3600 + 28 * 60 + 47


def test_it_is_due_when_every_condition_holds_and_quiet_is_as_long_as_the_baseline_took(ready):
    assert measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger) == ""
    assert ready.quiet_for == [TOOK], "as long as the baseline run took, 4 h 28 min 47 s"


def test_how_long_the_run_took_is_read_from_its_record_and_not_from_file_times(ready):
    """Scoring writes every file of a run again when it ends, so their times say the run took no time at all."""
    run = ready.home / "runs" / "baseline-1"
    for path in [run / "meta.json", *(run / "rows").iterdir()]:
        os.utime(path, (1_763_554_359, 1_763_554_359))
    baseline = measuring._latest_baseline(ready.ledger)
    assert measuring._how_long_a_run_takes(baseline) == TOOK
    assert measuring._how_long_a_run_takes({**baseline, "at": "not a time"}) == 0.0


@pytest.mark.parametrize(("change", "why"), [
    (lambda ready, monkeypatch: setattr(ready, "standing", None), "no change of hers is on probation"),
    (lambda ready, monkeypatch: ready.standing.offer.update(comparison_started={"run": "r", "rows": 0,
                                                                               "source_head": "head-1",
                                                                               "her_process": "4242"}),
     "added nothing when it last ran"),
    (lambda ready, monkeypatch: (ready.root / "outside_bar" / "SEAL.json").write_text('{"files": {"x": "y"}}'),
     "another seal"),
    (lambda ready, monkeypatch: monkeypatch.setattr(measuring, "_measuring", lambda: "a run is measuring her"),
     "measuring her"),
    (lambda ready, monkeypatch: monkeypatch.setattr(measuring, "_on_the_charger", lambda: False), "charger"),
    (lambda ready, monkeypatch: monkeypatch.setattr(measuring, "_how_long_nobody_has_talked_with_her",
                                                    lambda seconds: "recent_user_40"), "somebody may want her"),
])
def test_each_condition_alone_keeps_it_from_starting(ready, monkeypatch, change, why):
    change(ready, monkeypatch)
    assert why in measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger)


def test_a_comparison_the_benchmarks_already_made_on_it_is_not_made_again(ready, monkeypatch):
    monkeypatch.setattr(measuring, "verdicts_on", lambda commit, root, ledger=None: ["HELD"] if commit == "c0ffee" else [])
    assert "spoken on it" in measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger)


def test_a_started_run_that_ended_without_a_verdict_is_resumed_when_it_made_progress(ready, monkeypatch):
    """F16, 19 November: a comparison that started was taken for one that would end, and nothing started another, so a
    run that died kept the change on probation for good."""
    (ready.home / "runs" / "probation-1" / "rows").mkdir(parents=True)
    (ready.home / "runs" / "probation-1" / "rows" / "a.json").write_text("{}")
    ready.standing.offer["comparison_started"] = {"run": "probation-1", "rows": 0, "source_head": "head-1",
                                                  "her_process": "4242", "against": "baseline-1", "attempt": 1}
    assert measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger) == ""


def test_a_started_run_is_resumed_when_her_source_or_her_process_changed_since(ready, monkeypatch):
    ready.standing.offer["comparison_started"] = {"run": "probation-1", "rows": 0, "source_head": "head-0",
                                                  "her_process": "4242", "against": "baseline-1"}
    assert measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger) == ""
    ready.standing.offer["comparison_started"].update(source_head="head-1", her_process="17")
    assert measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger) == ""


def test_a_started_run_still_going_is_left_to_go_on(ready, monkeypatch):
    monkeypatch.setattr(measuring, "_measuring", lambda: "a run of the outside bar is going")
    ready.standing.offer["comparison_started"] = {"run": "probation-1"}
    assert "going" in measuring.comparison_due(ready.standing, root=ready.root, ledger=ready.ledger)


def test_the_run_is_as_large_as_the_baseline_and_ends_in_the_comparison(ready, monkeypatch):
    chain = measuring.the_chain(root=ready.root, name="probation-1", baseline={"run": "baseline-1"}, python="/py")
    lines = [line for line in chain.splitlines() if line.startswith("/py")]
    assert "outside_bar.run ask --run probation-1 --each 12" in lines[0]
    assert "outside_bar.cold_restart --run probation-1 --facts 12" in lines[1]
    assert "outside_bar.run compare --baseline baseline-1 --candidate probation-1" in lines[2]
    assert all("|| exit $?" in line for line in lines[:2]), "a step that fails stops the run"


def test_starting_it_hands_the_run_to_the_system_and_says_so_on_her_record(ready, monkeypatch):
    from core.agency import her_goal_at_work as goals

    monkeypatch.setattr(goals, "change_on_probation", lambda: ready.standing)
    monkeypatch.setattr(goals, "source_root", lambda: ready.root)
    written: list = []
    monkeypatch.setattr(goals, "_write_now", lambda record: written.append(dict(record.offer)))
    spawned: list = []

    class _Gateway:
        def write_text(self, path, text, *, source):
            Path(path).write_text(text)

        def spawn(self, argv, **kwargs):
            spawned.append(argv)

    monkeypatch.setattr("core.runtime.file_write_gateway.get_file_write_gateway", lambda: _Gateway())
    monkeypatch.setattr("core.runtime.subprocess_gateway.get_subprocess_gateway", lambda: _Gateway())
    started = measuring.start_a_comparison_when_due()
    assert started["started"] and started["against"] == "baseline-1"
    assert spawned[0][:2] == ["/bin/zsh", "-c"] and "nohup" in spawned[0][2] and spawned[0][2].endswith("&")
    assert written[-1]["comparison_started"]["run"] == started["run"]
    assert (ready.home / f"compare-{started['run']}.sh").exists()


def test_resuming_runs_the_same_run_again_and_counts_the_attempt(ready, monkeypatch):
    from core.agency import her_goal_at_work as goals

    monkeypatch.setattr(goals, "change_on_probation", lambda: ready.standing)
    monkeypatch.setattr(goals, "source_root", lambda: ready.root)
    written: list = []
    monkeypatch.setattr(goals, "_write_now", lambda record: written.append(dict(record.offer)))

    class _Gateway:
        def write_text(self, path, text, *, source):
            Path(path).write_text(text)

        def spawn(self, argv, **kwargs):
            pass

    monkeypatch.setattr("core.runtime.file_write_gateway.get_file_write_gateway", lambda: _Gateway())
    monkeypatch.setattr("core.runtime.subprocess_gateway.get_subprocess_gateway", lambda: _Gateway())
    (ready.home / "runs" / "probation-1" / "rows").mkdir(parents=True)
    (ready.home / "runs" / "probation-1" / "rows" / "a.json").write_text("{}")
    ready.standing.offer["comparison_started"] = {"run": "probation-1", "rows": 0, "source_head": "head-1",
                                                  "her_process": "4242", "against": "baseline-1", "attempt": 1}
    resumed = measuring.start_a_comparison_when_due()
    assert resumed == {"started": True, "run": "probation-1", "against": "baseline-1", "attempt": 2}
    assert written[-1]["comparison_started"]["rows"] == 1, "the next resume asks for progress past this"


def test_a_drafts_outside_check_is_not_a_run_of_the_outside_bar(monkeypatch, tmp_path):
    """LIVE 26 to 27 November 2026 every outside check of a draft, outside_bar/verify_change.py, read as a run of the
    outside bar, so for 34 hours her change on probation was never compared and never settled."""
    import re
    import subprocess as sp

    from core.agency import her_goal_at_work, measuring_her_on_probation as measuring

    monkeypatch.setattr(her_goal_at_work, "MEASURING_HOLD", tmp_path / "measuring.hold")
    running: list[str] = []

    def pgrep(argv, why):
        pattern = argv[-1]
        hits = [line for line in running if re.search(pattern, line)]
        return sp.CompletedProcess(argv, 0, stdout="\n".join(hits), stderr="")

    monkeypatch.setattr(measuring, "_read", pgrep)
    running[:] = ["python outside_bar/verify_change.py --repo /her_goals/x --commit abc"]
    assert measuring._measuring() == ""
    running[:] = ["python -W ignore -m outside_bar.run ask --run probation-1"]
    assert measuring._measuring() == "a run of the outside bar is going"
    running[:] = ["python -W ignore -m outside_bar.cold_restart --run probation-1 --facts 8"]
    assert measuring._measuring() == "a run of the outside bar is going"
