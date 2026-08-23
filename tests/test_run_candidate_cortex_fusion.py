from __future__ import annotations

import json
from pathlib import Path

from tools import run_candidate_cortex_fusion as controller


def _plan(tmp_path: Path) -> tuple[dict, Path]:
    aura_root = tmp_path / ".aura"
    fusion_root = aura_root / "fusion"
    output_root = aura_root / "models"
    fusion_root.mkdir(parents=True)
    output_root.mkdir()
    key = tmp_path / "journal.key"
    key.write_bytes(b"k" * 64)
    plan = {
        "fusion_plan_sha256": "a" * 64,
        "python": "/usr/bin/python3",
        "output": {
            "generation_id": "generation",
            "fusion_root": str(fusion_root),
            "root": str(output_root),
        },
    }
    return plan, key


def test_launch_is_detached_resumable_and_binds_both_output_roots(
    tmp_path: Path, monkeypatch
) -> None:
    plan, key = _plan(tmp_path)
    observed: list[str] = []
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr(
        controller.detached,
        "main",
        lambda args: observed.extend(args) or 0,
    )
    monkeypatch.setattr(
        controller.detached,
        "_status",
        lambda path: {"state": "running", "run_dir": str(path)},
    )

    result = controller._launch(plan, key, resume=True)

    assert result["state"] == "running"
    assert "--resume" in observed
    assert observed[observed.index("--resume-contract") + 1] == "target_checkpoint"
    roots = [
        observed[index + 1]
        for index, value in enumerate(observed)
        if value == "--execution-output-root"
    ]
    assert roots == [plan["output"]["fusion_root"], plan["output"]["root"]]
    verifier = json.loads(observed[observed.index("--resume-verifier-json") + 1])
    assert verifier[1].endswith("verify_candidate_cortex_fusion_resume.py")
    target = observed[observed.index("--") + 1 :]
    assert target[1].endswith("run_candidate_cortex_fusion_target.py")


def test_status_does_not_require_a_model_or_plan(tmp_path: Path) -> None:
    root = tmp_path / "not-launched"
    root.mkdir()

    assert controller._status_root(root) == {
        "state": "not_launched",
        "terminal": False,
    }
