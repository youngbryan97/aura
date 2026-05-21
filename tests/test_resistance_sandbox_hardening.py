from __future__ import annotations

from core.embodiment import resistance_sandbox as sandbox_module
from core.embodiment.resistance_sandbox import ResistanceSandbox


def test_resistance_sandbox_blocks_path_escape(tmp_path):
    root = tmp_path / "sandbox"
    outside = tmp_path / "escape.txt"
    sandbox = ResistanceSandbox(root)

    action = sandbox.execute_with_prediction("create", "../escape.txt", "created")

    assert action.actual_outcome == "permission_denied"
    assert action.prediction_correct is False
    assert sandbox.get_resource_pressure() > 0
    assert not outside.exists()


def test_resistance_sandbox_corrupt_state_records_receipt(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / ".sandbox_state.json").write_text("{not-json", encoding="utf-8")

    monkeypatch.setattr(
        sandbox_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    sandbox = ResistanceSandbox(root)

    assert sandbox.get_prediction_accuracy() == 0.5
    assert recorded
    assert recorded[0][0] == "resistance_sandbox"
    assert recorded[0][2]["receipt_required"] is True
    assert "persisted state load failed" in str(recorded[0][2]["action"])


def test_resistance_sandbox_rejects_async_action_function(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    async def async_action():
        return "done"

    monkeypatch.setattr(
        sandbox_module,
        "record_degradation",
        lambda module, exc, **kwargs: recorded.append((module, type(exc).__name__, kwargs)),
    )

    sandbox = ResistanceSandbox(tmp_path / "sandbox")
    action = sandbox.execute_with_prediction("create", "file.txt", "created", async_action)

    assert action.actual_outcome == "unexpected:TypeError"
    assert action.error_magnitude == 0.9
    assert recorded
    assert recorded[0][1] == "TypeError"
    assert recorded[0][2]["receipt_required"] is True
