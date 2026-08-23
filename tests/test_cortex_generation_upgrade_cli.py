"""Operator-CLI contracts for model-scale cortex comparisons."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_model_lane_lease_covers_load_and_every_use(monkeypatch):
    import mlx_lm

    import core.runtime.model_lane_control as lane_control
    import tools.cortex_generation_upgrade as cli

    events = []

    @contextmanager
    def lane(**kwargs):
        events.append(("enter", kwargs["model_path"]))
        try:
            yield
        finally:
            events.append(("exit", kwargs["model_path"]))

    def load(path):
        events.append(("load", path))
        return object(), object()

    monkeypatch.setattr(lane_control, "standalone_model_lane", lane)
    monkeypatch.setattr(mlx_lm, "load", load)

    with cli._model_session("/models/candidate") as pair:
        assert len(pair) == 2
        events.append(("use", "/models/candidate"))
        assert [event[0] for event in events] == ["enter", "load", "use"]

    assert [event[0] for event in events] == ["enter", "load", "use", "exit"]


def test_compare_reuses_frozen_batteries_without_loading_a_model(tmp_path, monkeypatch):
    import mlx_lm

    import core.learning.cortex_generation_upgrade as upgrade
    import tools.cortex_generation_upgrade as cli

    current = {
        "schema": upgrade.EVALUATION_SCHEMA,
        "label": "current",
        "breadth_accuracy": 1.0,
        "reasoning_accuracy": 0.4,
        "identity_digests": ["old"],
    }
    candidate = {
        "schema": upgrade.EVALUATION_SCHEMA,
        "label": "candidate",
        "breadth_accuracy": 1.0,
        "reasoning_accuracy": 1.0,
        "identity_digests": ["new"],
    }
    descriptor = {"descriptor_sha256": "d" * 64}
    for name, value in (
        ("current.json", current),
        ("candidate.json", candidate),
        ("descriptor.json", descriptor),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(
        mlx_lm,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compare must not load a model")
        ),
    )
    result = cli.cmd_compare(
        SimpleNamespace(
            current_battery=str(tmp_path / "current.json"),
            candidate_battery=str(tmp_path / "candidate.json"),
            descriptor=str(tmp_path / "descriptor.json"),
            critical_gates="",
            out=str(tmp_path / "out"),
        )
    )

    comparison = json.loads((tmp_path / "out" / "comparison.json").read_text())
    assert result == 0
    assert comparison["verdict"] == "PASS"
    assert comparison["promotion_eligible"] is False


def test_serving_command_parses_one_resumable_session() -> None:
    import tools.cortex_generation_upgrade as cli

    args = cli.build_parser().parse_args(
        [
            "qualify-serving",
            "--candidate",
            "/models/candidate",
            "--descriptor",
            "/evidence/descriptor.json",
            "--context-windows",
            "32768,8192,8192",
            "--prefill-chunk",
            "2048",
        ]
    )

    assert args.func is cli.cmd_qualify_serving
    assert args.context_windows == (8192, 32768)
    assert args.prefill_chunk == 2048


def test_artifact_seal_changes_when_a_model_file_moves(tmp_path: Path) -> None:
    import tools.cortex_generation_upgrade as cli

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    before = cli._artifact_stat_seal(str(tmp_path))
    (tmp_path / "config.json").write_text('{"changed":true}', encoding="utf-8")
    after = cli._artifact_stat_seal(str(tmp_path))

    assert before["seal_sha256"] != after["seal_sha256"]


def test_serving_progress_is_authenticated_and_rejects_an_edited_pass(tmp_path: Path) -> None:
    import tools.cortex_generation_upgrade as cli

    key = b"k" * 32
    kwargs = {
        "model_path": str(tmp_path / "candidate"),
        "descriptor_sha256": "d" * 64,
        "context_windows": (8192, 32768),
        "prefill_chunk_tokens": 2048,
        "artifact_seal_sha256": "a" * 64,
        "auth_key": key,
    }
    journal, record, _persist, validator, _binding, run_dir = (
        cli._serving_progress_recorder(tmp_path, **kwargs)
    )
    record(
        {
            "schema": "aura.cortex_upgrade.serving_progress.v2",
            "cell_id": "template",
            "completed": 1,
            "total": 6,
            "row": {"passed": True, "row_sha256": "r" * 64},
            "updated_at": 1.0,
        }
    )
    assert validator(journal["events"][0]) is False
    reopened, *_rest = cli._serving_progress_recorder(tmp_path, **kwargs)
    reopened_validator = _rest[2]
    assert reopened_validator(reopened["events"][0]) is True

    progress_path = run_dir / "progress_serving.json"
    attacked = json.loads(progress_path.read_text(encoding="utf-8"))
    attacked["events"][0]["row"]["passed"] = False
    progress_path.write_text(json.dumps(attacked), encoding="utf-8")
    with pytest.raises(RuntimeError, match="authentication_failed"):
        cli._serving_progress_recorder(tmp_path, **kwargs)
