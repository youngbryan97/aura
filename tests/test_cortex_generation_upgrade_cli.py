"""Operator-CLI contracts for model-scale cortex comparisons."""
from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace


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
