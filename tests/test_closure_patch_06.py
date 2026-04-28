from pathlib import Path
import importlib.util
import json


def test_flagship_doctor_runs_on_minimal_repo(tmp_path: Path):
    from core.runtime.flagship_doctor import run_doctor

    (tmp_path / "core" / "runtime").mkdir(parents=True)
    report = run_doctor(tmp_path, include_gates=False)

    assert report.root == str(tmp_path.resolve())
    assert report.overall in {"pass", "warn", "fail"}
    assert report.findings


def test_morphogenesis_report_analyzes_registry(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "morph_report",
        Path("scripts/aura_morphogenesis_longitudinal_report.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    data = {
        "payload": {
            "cells": {
                "cell1": {
                    "state": {"lifecycle": "active"},
                    "manifest": {"role": "sensor"},
                },
                "cell2": {
                    "state": {"lifecycle": "quarantined"},
                    "manifest": {"role": "repair"},
                },
            },
            "organs": {"organ1": {}},
        }
    }

    analysis = mod.analyze_registry(data)
    assert analysis["cell_count"] == 2
    assert analysis["organ_count"] == 1
    assert analysis["by_lifecycle"]["active"] == 1
    assert analysis["by_lifecycle"]["quarantined"] == 1


def test_morphogenesis_report_writes_files(tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "morph_report",
        Path("scripts/aura_morphogenesis_longitudinal_report.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    root = tmp_path / "repo"
    state_dir = root / "data" / "morphogenesis"
    get_task_tracker().create_task(get_storage_gateway().create_dir(state_dir, cause='test_morphogenesis_report_writes_files'))
    (state_dir / "morphogenesis_state.json").write_text(
        json.dumps({"payload": {"cells": {}, "organs": {}}}),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    report = mod.build_report(root, out)

    assert (out / "morphogenesis_longitudinal_report.json").exists()
    assert (out / "morphogenesis_longitudinal_report.md").exists()
    assert report["state_paths"]
