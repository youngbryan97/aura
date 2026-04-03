from pathlib import Path
from types import SimpleNamespace

import pytest

from core import demo_support


def test_extract_background_diagnostic_target_accepts_realistic_async_phrasing():
    target = demo_support.extract_background_diagnostic_target(
        "Aura, inspect interface/static/shell/src/App.jsx in the background and post the result here when you're done."
    )

    assert target == "interface/static/shell/src/App.jsx"


@pytest.mark.asyncio
async def test_surface_activity_marks_user_requested_delivery_as_authorized():
    captured = {}

    class _FakeOutputGate:
        async def emit(self, content, origin="system", target="primary", metadata=None, timeout=5.0):
            captured["content"] = content
            captured["origin"] = origin
            captured["target"] = target
            captured["metadata"] = dict(metadata or {})

    orch = SimpleNamespace(output_gate=_FakeOutputGate())

    await demo_support._surface_activity(orch, "Finished the background check.")

    assert captured["content"] == "Finished the background check."
    assert captured["origin"] == "assistant"
    assert captured["target"] == "primary"
    assert captured["metadata"]["requested_by_user"] is True
    assert captured["metadata"]["executive_authority"] is True


@pytest.mark.asyncio
async def test_run_background_file_diagnostic_records_failures_honestly(monkeypatch):
    recorded = {}

    async def _fake_record_recent_activity(_orch, payload):
        recorded["payload"] = payload

    async def _fake_surface_activity(_orch, summary):
        recorded["summary"] = summary

    monkeypatch.setattr(demo_support, "_record_recent_activity", _fake_record_recent_activity)
    monkeypatch.setattr(demo_support, "_surface_activity", _fake_surface_activity)
    monkeypatch.setattr(demo_support, "_resolve_target_path", lambda *_args, **_kwargs: Path("/tmp/example.py"))
    monkeypatch.setattr(demo_support, "_summarize_target", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))

    await demo_support.run_background_file_diagnostic("example.py", SimpleNamespace())

    assert recorded["payload"]["ok"] is False
    assert "RuntimeError" in recorded["summary"]


@pytest.mark.asyncio
async def test_recent_activity_reply_ignores_stale_persisted_state(monkeypatch):
    monkeypatch.setattr(
        demo_support,
        "_load_last_activity",
        lambda: {
            "target_name": "old_demo.py",
            "summary": "I finished the background diagnostic on `old_demo.py`. It did something useful.",
            "completed_at": 1.0,
        },
    )

    reply = await demo_support.maybe_build_recent_activity_reply(
        "What were you doing right before this session started?",
        SimpleNamespace(),
    )

    assert reply is None


@pytest.mark.asyncio
async def test_recent_activity_reply_ignores_stale_live_state(monkeypatch):
    monkeypatch.setattr(demo_support, "_load_last_activity", lambda: None)

    orch = SimpleNamespace(
        _demo_last_background_activity={
            "target_name": "old_demo.py",
            "summary": "I finished the background diagnostic on `old_demo.py`. It did something useful.",
            "completed_at": 1.0,
        }
    )

    reply = await demo_support.maybe_build_recent_activity_reply(
        "What were you doing right before this session started?",
        orch,
    )

    assert reply is None


def test_python_summary_reports_real_parse_failures():
    summary = demo_support._python_summary(
        Path("broken.py"),
        "def nope(:\n    pass\n",
    )

    assert "doesn't currently parse as Python" in summary
    assert "line 1" in summary
