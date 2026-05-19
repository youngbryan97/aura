import json

import pytest

from interface.routes import dashboard


def test_dashboard_safe_records_runtime_failure(monkeypatch):
    recorded = []
    calls = []

    def unavailable_source():
        calls.append("called")
        raise RuntimeError("source offline")

    monkeypatch.setattr(
        dashboard,
        "record_degradation",
        lambda subsystem, error: recorded.append((subsystem, str(error))),
    )

    assert dashboard._safe(unavailable_source, default={"status": "offline"}) == {"status": "offline"}
    assert calls == ["called"]
    assert recorded == [("dashboard", "source offline")]


@pytest.mark.asyncio
async def test_conscience_recent_keeps_valid_rows_when_log_has_bad_lines(tmp_path, monkeypatch):
    log_path = tmp_path / ".aura" / "data" / "conscience" / "violations.jsonl"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "first", "severity": "low"}),
                "{broken",
                json.dumps({"id": "second", "severity": "high"}),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard.pathlib.Path, "home", lambda: tmp_path)

    response = await dashboard.conscience_recent(limit=2, _=None)
    payload = json.loads(response.body)

    assert payload == {
        "violations": [
            {"id": "first", "severity": "low"},
            {"id": "second", "severity": "high"},
        ]
    }
