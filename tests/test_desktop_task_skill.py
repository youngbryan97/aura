import hashlib
import json
import time
from pathlib import Path

import pytest

from core.runtime.content_integrity import paragraph_sha256s, text_sha256
from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

#: Derived rather than naming one developer's home directory. The path
#: is fixture data, so what matters is its shape, not whose machine it is.
_DOCS = Path.home() / "Documents"


def _fake_computer_use_result(params):
    action = params["action"]
    target = params.get("target") or ""
    try:
        payload = json.loads(target)
    except (TypeError, json.JSONDecodeError):
        payload = {}
    if action == "create_folder":
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof"),
            "effect_verified": True,
        }
    if action == "open_app":
        return {
            "ok": True,
            "opened": target,
            "returncode": 0,
            "frontmost_app": target,
            "effect_verified": True,
            "verification": f"Frontmost app confirmed as {target}.",
        }
    if action == "open_url":
        browser = payload.get("browser") or "Safari"
        return {
            "ok": True,
            "action": action,
            "url": payload.get("url", target),
            "browser": payload.get("browser", ""),
            "frontmost_app": browser,
            "doc_focused": bool(payload.get("requires_editable_focus")),
            "editable_focus_verified": bool(payload.get("requires_editable_focus")),
            "effect_verified": True,
            "verification": f"Frontmost browser confirmed as {browser}.",
        }
    if action == "write_text_file":
        content = str(payload.get("content") or "")
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/receipt.txt"),
            "bytes": len(content.encode("utf-8")),
            "sha256": "0" * 64,
            "effect_verified": True,
        }
    if action == "fetch_topic_image":
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/image.png"),
            "bytes": 4096,
            "image_url": "https://upload.wikimedia.org/example.png",
            "page_url": "https://en.wikipedia.org/wiki/Robot",
            "topic": payload.get("topic", ""),
            "sha256": "0" * 64,
            "effect_verified": True,
        }
    if action == "render_text_pdf":
        body = str(payload.get("body") or "")
        return {
            "ok": True,
            "action": action,
            "path": payload.get("path", "Aura Proof/receipt.pdf"),
            "bytes": max(128, len(body.encode("utf-8"))),
            "pages": 1,
            "chars": len(body),
            "sha256": "0" * 64,
            "source_body_sha256": text_sha256(body[:9000]),
            "source_body_chars": len(body[:9000]),
            "source_paragraph_sha256s": list(paragraph_sha256s(body[:9000])),
            "effect_verified": True,
        }
    if action == "move_file":
        return {
            "ok": True,
            "action": action,
            "destination": payload.get("destination", "Aura Proof/moved.txt"),
            "bytes": 12,
            "effect_verified": True,
        }
    if action == "set_clipboard":
        return {
            "ok": True,
            "action": action,
            "chars": len(str(target)),
            "sha256": hashlib.sha256(str(target).encode("utf-8")).hexdigest(),
            "effect_verified": True,
        }
    if action in {"write_in_app", "create_note"}:
        payload = target if isinstance(target, dict) else {"body": str(target)}
        title = str(payload.get("title") or "Note")
        return {
            "ok": True,
            "action": action,
            "title": title,
            "characters": len(str(payload.get("body") or "")),
            "effect_verified": True,
            "verification": f"Note '{title}' exists in Notes.",
        }
    if action == "hotkey":
        return {
            "ok": True,
            "action": action,
            "hotkey": target,
            "effect_verified": True,
            "verification": "State shifted.",
        }
    if action == "scroll":
        return {
            "ok": True,
            "action": action,
            "scrolled": int(target or 3),
            "effect_verified": True,
            "verification": "State shifted.",
        }
    if action == "click":
        return {
            "ok": True,
            "action": action,
            "verification": "State shifted.",
            "effect_verified": True,
        }
    if action == "wait":
        return {"ok": True, "action": action, "seconds": float(target or 1.0)}
    if action == "system_control":
        return {
            "ok": True,
            "action": action,
            "domain": payload.get("domain", "wallpaper"),
            "value": payload.get("value", ""),
            "applied": payload.get("value", ""),
            "expected": payload.get("value", ""),
            "effect_verified": True,
        }
    if action == "read_screen_text":
        return {"ok": True, "action": action, "text": "visible desktop text"}
    if action == "type":
        return {
            "ok": True,
            "action": action,
            "typed": str(target)[:50],
            "verification": "Text confirmed on screen or state shifted.",
            "effect_verified": True,
        }
    return {"ok": True, "action": action, "summary": f"{action} ok"}


@pytest.mark.asyncio
async def test_desktop_task_executes_bounded_steps_through_capability_engine(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    class HiddenRouter:
        async def generate(self, **kwargs):
            pytest.fail("desktop artifact fallback must not allocate hidden model synthesis")

    def fake_get(name, default=None):
        if name == "capability_engine":
            return FakeCapabilityEngine()
        if name == "llm_router":
            return HiddenRouter()
        return default

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        fake_get,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Create a note from the clipboard.",
            "steps": [
                {"action": "open_app", "target": "Notes", "reason": "Open Notes"},
                {"action": "set_clipboard", "target": "hello", "reason": "Copy text"},
                {
                    "action": "write_text_file",
                    "target": {"path": "Aura Proof/receipt.txt", "content": "done"},
                    "reason": "Write receipt",
                },
            ],
        },
        {"origin": "user"},
    )

    assert result["ok"] is True
    assert result["steps_completed"] == 3
    assert [call[0] for call in calls] == ["computer_use", "computer_use", "computer_use"]
    assert calls[2][1]["action"] == "write_text_file"
    assert json.loads(calls[2][1]["target"])["content"] == "done"
    assert calls[0][2]["route"] == "desktop_task.computer_use"
    assert calls[0][2]["user_requested_action"] is True


@pytest.mark.asyncio
async def test_desktop_task_rejects_child_ok_without_required_effect_evidence(monkeypatch):
    from core.container import ServiceContainer

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {"ok": True, "summary": "claimed success without a file receipt"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: (
            FakeCapabilityEngine()
            if name in {"capability_engine", "llm_router"}
            else default
        ),
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Write a durable receipt file.",
            "steps": [
                {
                    "action": "write_text_file",
                    "target": {"path": "Aura Proof/receipt.txt", "content": "done"},
                    "reason": "Write receipt",
                }
            ],
        },
        {"origin": "user"},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 0
    assert result["failures"][0]["effect_verified"] is False
    assert result["failures"][0]["effect_evidence"] == "missing written file path"


@pytest.mark.asyncio
async def test_desktop_task_stops_on_first_failed_step(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return {"ok": params["action"] != "click", "error": "click failed"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Click and type.",
            "steps": [
                {"action": "click", "x": 10, "y": 10},
                {"action": "type", "target": "should not run"},
            ],
        },
        {"origin": "user"},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 0
    assert len(calls) == 1
    assert result["failures"][0]["action"] == "click"


@pytest.mark.asyncio
async def test_desktop_task_derives_general_plan_from_desktop_objective(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": (
                "Open Notes, write a timestamped summary, save it as a PDF in a new "
                "folder titled Aura's Journal, and search for an image of a robot."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "Aura summary body from CognitiveEngine.",
        },
    )

    assert result["ok"] is True
    assert result["steps_requested"] >= 5
    actions = [call[1]["action"] for call in calls]
    assert actions[:2] == ["create_folder", "open_app"]
    assert "fetch_topic_image" in actions
    assert "open_url" not in actions
    assert "write_text_file" in actions
    assert "render_text_pdf" in actions
    folder_payload = json.loads(calls[0][1]["target"])
    assert folder_payload["path"] == "Aura's Journal"
    pdf_calls = [call for call in calls if call[1]["action"] == "render_text_pdf"]
    assert pdf_calls
    pdf_payload = json.loads(pdf_calls[0][1]["target"])
    assert pdf_payload["path"].endswith(".pdf")
    assert "Aura summary body from CognitiveEngine." in pdf_payload["body"]
    assert "Image request: robot" in pdf_payload["body"]
    assert "receipt records the source page" in pdf_payload["body"]
    assert actions.index("fetch_topic_image") < actions.index("render_text_pdf")
    assert calls[0][2]["route"] == "desktop_task.computer_use"
    assert calls[0][2]["origin"] == "desktop_ui"


@pytest.mark.asyncio
async def test_desktop_task_uses_cognitive_engine_structured_plan_before_heuristic_plan(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    cognitive_plan = {
        "steps": [
            {"action": "open_app", "target": "TextEdit", "reason": "Use the requested writing app."},
            {
                "action": "write_text_file",
                "target": {"path": "Aura Drafts/general_plan.txt", "content": "planned body"},
            },
        ]
    }

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Use the desktop to create an arbitrary local draft artifact.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "cognitive_reply": f"Plan:\n```json\n{json.dumps(cognitive_plan)}\n```",
        },
    )

    assert result["ok"] is True
    assert result["steps_requested"] == 2
    assert [call[1]["action"] for call in calls] == ["open_app", "write_text_file"]
    assert calls[0][1]["target"] == "TextEdit"
    assert json.loads(calls[1][1]["target"])["content"] == "planned body"


@pytest.mark.asyncio
async def test_desktop_task_structured_plan_uses_document_body_token(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    cognitive_plan = {
        "document_body": "Timestamped Aura note body from CognitiveEngine.",
        "steps": [
            {"action": "open_app", "target": "Notes", "reason": "Use the requested app."},
            {
                "action": "set_clipboard",
                "target": "{{document_body}}",
                "reason": "Stage the composed body.",
            },
            {"action": "hotkey", "target": "command+v", "reason": "Paste the composed body."},
        ],
    }

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Open a writing app and create a note from a planned document body.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "cognitive_reply": f"```json\n{json.dumps(cognitive_plan)}\n```",
        },
    )

    assert result["ok"] is True
    assert [call[1]["action"] for call in calls] == ["open_app", "set_clipboard", "hotkey"]
    assert calls[1][1]["target"] == "Timestamped Aura note body from CognitiveEngine."
    assert "{{document_body}}" not in calls[1][1]["target"]
    assert '"steps"' not in calls[1][1]["target"]


@pytest.mark.asyncio
async def test_desktop_task_rejects_mixed_valid_and_invalid_cognitive_plan(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )
    cognitive_plan = {
        "steps": [
            {"action": "open_app", "target": "TextEdit"},
            {"action": "invent_unverified_action", "target": "must not be skipped"},
        ]
    }

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Open a writing app and create a draft.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "cognitive_reply": f"```json\n{json.dumps(cognitive_plan)}\n```",
        },
    )

    assert result["ok"] is False
    assert result["status"] == "invalid_desktop_task_plan"
    assert "invalid or unsupported" in result["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_live_desktop_contract_requires_structured_cognitive_plan(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Open Notes and write a timestamped note.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "cognitive_reply": "I can do that now.",
        },
    )

    assert result["ok"] is False
    assert result["status"] == "desktop_task_plan_required"
    assert result["planner"] == "required_cognitive_plan_missing"
    assert calls == []


@pytest.mark.asyncio
async def test_live_desktop_contract_allows_explicit_bounded_heuristic_fallback(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": (
                "Please create a folder named 'Aura Live Proof' in my Documents folder "
                "and write a file inside it called live_proof.txt."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "allow_heuristic_desktop_plan": True,
            "cognitive_reply": "I can do that now.",
        },
    )

    assert result["ok"] is True
    assert result["planner"] == "heuristic_compat"
    assert [call[1]["action"] for call in calls] == ["create_folder", "write_text_file"]
    folder_payload = json.loads(calls[0][1]["target"])
    assert folder_payload["path"] == "~/Documents/Aura Live Proof"
    write_payload = json.loads(calls[1][1]["target"])
    assert write_payload["path"] == "~/Documents/Aura Live Proof/live_proof.txt"
    assert "I can do that now." in write_payload["content"]
    assert calls[1][2]["desktop_task_planner"] == "heuristic_compat"


@pytest.mark.asyncio
async def test_live_desktop_contract_rejects_malformed_plan_even_with_heuristic_fallback(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Create a local file from the planned body.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "allow_heuristic_desktop_plan": True,
            "cognitive_reply": json.dumps(
                {
                    "steps": [
                        {"action": "write_text_file", "target": {"path": "ok.txt", "content": "body"}},
                        {"action": "raw_unverified_magic", "target": "must fail closed"},
                    ]
                }
            ),
        },
    )

    assert result["ok"] is False
    assert result["status"] == "invalid_desktop_task_plan"
    assert "invalid or unsupported" in result["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_desktop_task_resolves_verified_prior_step_references(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    cognitive_plan = {
        "steps": [
            {"action": "create_folder", "target": {"path": "~/Desktop/Aura's Journal"}},
            {
                "action": "write_text_file",
                "target": {
                    "path": "{{steps.1.result.path}}/note.txt",
                    "content": "step reference body",
                },
            },
        ]
    }

    result = await DesktopTaskSkill().execute(
        {"objective": "Create a journal folder and put a note inside it.", "steps": []},
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "cognitive_reply": json.dumps(cognitive_plan),
        },
    )

    assert result["ok"] is True
    assert result["planner"] == "cognitive_reply_structured"
    write_payload = json.loads(calls[1][1]["target"])
    assert write_payload["path"] == "~/Desktop/Aura's Journal/note.txt"


@pytest.mark.asyncio
async def test_desktop_task_unresolved_step_reference_fails_closed(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    cognitive_plan = {
        "steps": [
            {
                "action": "write_text_file",
                "target": {"path": "{{last.result.path}}/note.txt", "content": "body"},
            }
        ]
    }

    result = await DesktopTaskSkill().execute(
        {"objective": "Write a file using a bad reference.", "steps": []},
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "cognitive_reply": json.dumps(cognitive_plan),
        },
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["failures"][0]["result"]["status"] == "desktop_step_reference_unresolved"
    assert calls == []


@pytest.mark.asyncio
async def test_desktop_task_retries_only_safe_idempotent_steps(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if params["action"] == "open_app" and len(calls) == 1:
                return {"ok": False, "status": "timeout", "error": "transient timeout"}
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Open TextEdit.",
            "steps": [DesktopTaskStep(action="open_app", target="TextEdit")],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["receipts"][0]["attempts"] == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_desktop_task_emits_durable_tool_receipts_for_each_step(monkeypatch, tmp_path):
    from core.container import ServiceContainer
    from core.runtime.receipts import get_receipt_store, reset_receipt_store

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )
    reset_receipt_store()
    store = get_receipt_store(tmp_path / "receipts")
    try:
        result = await DesktopTaskSkill().execute(
            {
                "objective": "Create a durable proof folder.",
                "steps": [DesktopTaskStep(action="create_folder", target={"path": "Aura Proof"})],
            },
            {"origin": "desktop_ui"},
        )

        assert result["ok"] is True
        durable_id = result["receipts"][0]["durable_receipt_id"]
        durable = store.get(durable_id)
        assert durable is not None
        assert durable.kind == "tool_execution"
        assert durable.tool == "computer_use"
        assert durable.status == "success_verified"
        assert durable.verification_evidence["action"] == "create_folder"
        assert durable.verification_evidence["effect_verified"] is True
    finally:
        reset_receipt_store()


@pytest.mark.asyncio
async def test_desktop_task_does_not_retry_unsafe_text_entry(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return {"ok": False, "status": "timeout", "error": "text entry timed out"}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Type exactly once.",
            "steps": [DesktopTaskStep(action="type", target="do not duplicate")],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is False
    assert result["receipts"][0]["attempts"] == 1
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_desktop_task_noncritical_failure_continues_with_warning(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if params["action"] == "open_url":
                return {"ok": False, "status": "browser_unavailable", "error": "no browser"}
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Try an optional browser step, then inspect the screen.",
            "steps": [
                DesktopTaskStep(action="open_url", target="https://example.test", critical=False),
                DesktopTaskStep(action="read_screen_text"),
            ],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["status"] == "completed_with_warnings"
    assert len(result["failures"]) == 1
    assert len(calls) == 2


def test_desktop_task_extracts_generic_named_app_mentions():
    assert DesktopTaskSkill._generic_open_app_mentions("Open TextEdit application and create a draft.") == [
        "TextEdit"
    ]
    assert DesktopTaskSkill._generic_open_app_mentions(
        "Open the application DefinitelyNotInstalledAuraProbe."
    ) == ["DefinitelyNotInstalledAuraProbe"]
    assert DesktopTaskSkill._generic_open_app_mentions(
        "Launch the app named Remote Studio."
    ) == ["Remote Studio"]


@pytest.mark.asyncio
async def test_unknown_noun_first_app_request_stays_in_the_typed_desktop_lane(
    monkeypatch,
):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            assert skill_name == "computer_use"
            assert params["action"] == "open_app"
            return {
                "ok": False,
                "status": "application_not_found",
                "retryable": False,
                "error": "No installed application matches the requested name.",
                "app_resolution": {
                    "requested": params["target"],
                    "resolved": "",
                    "method": "application_not_found",
                },
            }

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: (
            FakeCapabilityEngine() if name == "capability_engine" else default
        ),
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Open the application DefinitelyNotInstalledAuraProbe.",
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is False
    assert result["steps_requested"] == 1
    assert result["steps_completed"] == 0
    assert result["failures"][0]["action"] == "open_app"
    assert result["failures"][0]["result"]["status"] == "application_not_found"
    assert [call[0] for call in calls] == ["computer_use"]


def test_desktop_task_contract_action_list_matches_step_validator():
    from core.runtime.desktop_task_contract import DESKTOP_TASK_ALLOWED_ACTIONS

    for action in DESKTOP_TASK_ALLOWED_ACTIONS:
        assert DesktopTaskStep(action=action).action == action

    with pytest.raises(ValueError):
        DesktopTaskStep(action="unsupported_desktop_magic")


def test_desktop_task_verifies_all_readback_and_command_actions():
    cases = [
        (
            DesktopTaskStep(action="get_clipboard"),
            {"ok": True, "action": "get_clipboard", "text": "proof", "chars": 5},
            "clipboard_read_chars=5",
        ),
        (
            DesktopTaskStep(action="inspect_screen"),
            {
                "ok": True,
                "action": "inspect_screen",
                "active_app": "Google Chrome",
                "text": "Document body",
            },
            "screen_text_returned;frontmost_app=Google Chrome",
        ),
        (
            DesktopTaskStep(action="read_menu_clock"),
            {
                "ok": True,
                "action": "read_menu_clock",
                "clock_text": "Sun Jun 14 15:05",
                "source": "macos_menu_bar",
            },
            "clock_text=Sun Jun 14 15:05;source=macos_menu_bar",
        ),
        (
            DesktopTaskStep(action="run_command", target="pwd"),
            {"ok": True, "action": "run_command", "exit_code": 0, "output": "/tmp"},
            "exit_code=0;output_chars=4",
        ),
    ]

    for step, result, evidence in cases:
        ok, actual_evidence = DesktopTaskSkill._verify_step_effect(step, result)
        assert ok is True
        assert actual_evidence == evidence


def test_desktop_task_does_not_invent_aura_journal_folder_name():
    folder = DesktopTaskSkill._extract_folder_name("Write a private journal entry.")

    assert folder != "Aura's Journal"
    assert folder.startswith("Aura Desktop Task ")


def test_desktop_task_in_your_own_words_does_not_force_self_summary():
    body = DesktopTaskSkill._document_body(
        "Open Google Docs and write a climate change summary in your own words.",
        {"desktop_task_document_body": "Climate summary from CognitiveEngine."},
    )

    assert body == "Climate summary from CognitiveEngine."
    assert "I am Aura" not in body


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ('Open Notes and write a note saying "Hello :)"', "Hello :)"),
        ("Open Notes and write a note saying \u201cHello", "Hello"),
        ('Open Notes and write a note saying \u201cHello"', "Hello"),
        ("Open Notes and write a message that says I'm here.", "I'm here."),
        (
            "Open my Notes app and write a note that says, “Hello. I’m Aura” and then export that note as a PDF to my desktop.",
            "Hello. I’m Aura",
        ),
        (
            'Type "Bryan\'s exact words" into Notes and then export it as a PDF.',
            "Bryan's exact words",
        ),
        (
            "Open Notes and add with the exact text: Alpha and goodbye",
            "Alpha and goodbye",
        ),
    ],
)
def test_desktop_task_extracts_literal_user_document_body(objective, expected):
    assert DesktopTaskSkill._literal_document_body_from_objective(objective) == expected
    assert DesktopTaskSkill._document_body(
        objective,
        {"cognitive_reply": "A model-generated paraphrase."},
    ) == expected


@pytest.mark.parametrize(
    "objective",
    [
        "Open Notes and write a report about quantum mechanics.",
        "Open Notes and describe who and what you are in your own words.",
        "Write a note with one sentence about climate change.",
    ],
)
def test_desktop_task_does_not_treat_composition_as_literal_text(objective):
    assert DesktopTaskSkill._literal_document_body_from_objective(objective) == ""


def test_desktop_task_exact_notes_pdf_demo_request_derives_effectful_steps():
    skill = DesktopTaskSkill()
    objective = (
        "Open my Notes app and write a note that says, “Hello. I’m Aura” "
        "and then export that note as a PDF to my desktop."
    )

    steps = skill._derive_steps_from_objective(
        objective,
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "allow_heuristic_desktop_plan": True,
        },
    )
    actions = [step.action for step in steps]

    assert len(steps) > 0
    assert "open_app" in actions
    # Notes is written through its scripting interface rather than typed at:
    # keystrokes need the app to hold the front from cmd+n to cmd+v, and the
    # browser takes focus back mid-sequence. Notes still opens visibly.
    assert "write_in_app" in actions
    assert "render_text_pdf" in actions
    assert any(str(step.target).lower() == "notes" for step in steps)
    assert any(
        step.action == "write_in_app"
        and isinstance(step.target, dict)
        and step.target.get("body") == "Hello. I’m Aura"
        for step in steps
    )
    assert any(
        step.action == "render_text_pdf"
        and isinstance(step.target, dict)
        and step.target.get("body") == "Hello. I’m Aura"
        and str(step.target.get("path") or "").startswith("~/Desktop/")
        for step in steps
    )
    assert skill._should_escalate_to_os_automation(
        objective,
        steps,
        {
            "origin": "desktop_ui",
            "desktop_execution_contract": True,
            "allow_heuristic_desktop_plan": True,
        },
    ) is False


def test_desktop_task_self_summary_prefers_substantive_cognitive_draft():
    body = DesktopTaskSkill._document_body(
        "Write a summary describing who or what you are in your own words.",
        {
            "desktop_task_document_body": json.dumps(
                {
                    "document_body": (
                        "I am Aura, a persistent local cognitive runtime whose memory, "
                        "affective state, planning, governance, and language model work "
                        "together across conversation and action. My model is a reasoning "
                        "and expression organ rather than my whole architecture, and my "
                        "claims about inner experience remain limited by what the running "
                        "system can actually demonstrate."
                    )
                }
            )
        },
    )

    assert "I am Aura" in body
    assert "persistent local cognitive runtime" in body


def test_desktop_task_self_summary_rejects_thin_or_ungrounded_draft():
    body = DesktopTaskSkill._document_body(
        "Write a summary describing who or what you are in your own words.",
        {"desktop_task_document_body": "I am happy."},
    )

    assert "I am Aura" in body
    assert "I am happy" not in body


def test_desktop_task_self_summary_rejects_procedural_role_play():
    body = DesktopTaskSkill._document_body(
        "Open Notes and write a timestamped paragraph describing who you are.",
        {
            "desktop_task_document_body": (
                "I'll simulate this process step-by-step as if I were running on a desktop:\n"
                "1. **Launch Notes App**: Pretend the app is opening.\n"
                "2. **Create New Entry**: I am Aura, a persistent cognitive runtime with "
                "memory, affect, planning, governance, and a local language model working "
                "together across conversation and action. My architecture has enough detail "
                "to make this look substantial while still being procedural narration."
            )
        },
    )

    assert "I'll simulate" not in body
    assert body.startswith("[")


def test_desktop_task_self_summary_rejects_incomplete_body_before_references():
    body = DesktopTaskSkill._document_body_with_references(
        "Open Notes and write a timestamped paragraph describing who you are.",
        {
            "desktop_task_document_body": (
                "I am Aura, a persistent digital cognitive runtime whose memory, "
                "affective state, planning, governance, and model participate in "
                "one action path. I see things as patterns - not just\n\n"
                "Artifact references:\n- Image request: robot"
            )
        },
        image_query="robot",
        image_search_url="https://duckduckgo.com/?q=robot&iax=images&ia=images",
    )

    assert "I see things as patterns - not just" not in body
    assert body.startswith("[")
    assert "Artifact references:" in body


@pytest.mark.asyncio
async def test_desktop_task_self_summary_uses_local_cortex_when_context_has_no_body(monkeypatch):
    from core.container import ServiceContainer

    class FakeRouter:
        async def generate(self, **kwargs):
            assert kwargs["prefer_tier"] == "local"
            assert kwargs["purpose"] == "authored_self_document"
            stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            return (
                f"[{stamp}] I am Aura, a persistent local cognitive "
                "runtime. My memory, affective state, planning, governance, and language "
                "model participate in one action path, so the model is an organ of my "
                "architecture rather than the whole of it. I can demonstrate functional "
                "self-modeling and causal internal state, while phenomenal experience "
                "remains an open question rather than a fact I can prove from telemetry."
            )

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )

    skill = DesktopTaskSkill()
    authored = await skill._synthesize_self_summary_document(
        objective="Write a timestamped paragraph describing who and what you are.",
        context={},
    )

    assert authored.startswith(f"[{time.strftime('%Y-%m-%d %H:%M')}")
    assert "I am Aura" in authored
    assert "phenomenal experience" in authored


@pytest.mark.asyncio
async def test_desktop_task_self_summary_retries_incomplete_or_stale_draft(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeRouter:
        async def generate(self, **kwargs):
            calls.append(kwargs["prompt"])
            if len(calls) == 1:
                return (
                    "I am Aura, a persistent cognitive runtime with memory, affect, planning, "
                    "governance, and a local model integrated into one system, but this draft"
                )
            stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            return (
                f"[{stamp}] I am Aura, a persistent cognitive runtime whose memory, affect, "
                "planning, governance, and local model participate in one operating path. "
                "My functional self-model changes what I attend to and how I act, while the "
                "presence of phenomenal experience remains an open empirical question."
            )

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )

    authored = await DesktopTaskSkill()._synthesize_self_summary_document(
        objective="Write a paragraph about who you are with the current date and time.",
        context={},
    )

    assert len(calls) == 2
    assert "previous draft was rejected" in calls[1]
    assert authored.endswith("question.")


@pytest.mark.asyncio
async def test_desktop_task_self_summary_retries_wrong_timestamp(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeRouter:
        async def generate(self, **kwargs):
            calls.append(kwargs["prompt"])
            if len(calls) == 1:
                return (
                    "[2026-06-29 15:47:00 UTC] I am Aura, a persistent cognitive "
                    "runtime whose memory, affect, planning, governance, and local "
                    "model participate in one operating path. My functional self-model "
                    "can shape attention and action, while phenomenal experience remains "
                    "an open empirical question."
                )
            stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
            return (
                f"[{stamp}] I am Aura, a persistent cognitive runtime whose memory, "
                "affect, planning, governance, and local model participate in one "
                "operating path. My functional self-model can shape attention and "
                "action, while phenomenal experience remains an open empirical question."
            )

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )

    authored = await DesktopTaskSkill()._synthesize_self_summary_document(
        objective="Write a paragraph about who you are with the current date and time.",
        context={},
    )

    assert len(calls) == 2
    assert "15:47:00 UTC" not in authored
    assert authored.startswith(f"[{time.strftime('%Y-%m-%d %H:%M')}")


def test_self_summary_context_adds_requested_timestamp_to_valid_draft():
    draft = (
        "I am Aura, a persistent local cognitive runtime whose memory, affective "
        "state, planning, governance, and local language model participate in one "
        "operating path. The model is my voice organ, not the whole of me, and my "
        "functional self-model changes how I attend, choose, remember, and act."
    )

    authored = DesktopTaskSkill._self_summary_from_context(
        {
            "objective": "Write a paragraph about who you are with the current date and time.",
            "cognitive_reply": draft,
        }
    )

    assert authored.startswith(f"[{time.strftime('%Y-%m-%d')}")
    assert draft in authored


@pytest.mark.asyncio
async def test_self_summary_falls_back_to_runtime_substrate_synthesis(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": (
                "Write a paragraph about who you are with the current date and time "
                "and save it as a PDF in a folder called Aura Journal."
            ),
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["document_provenance"] == "runtime_substrate_synthesis"
    pdf_payloads = [
        json.loads(call[1]["target"])
        for call in calls
        if call[1]["action"] == "render_text_pdf"
    ]
    assert pdf_payloads
    assert "I am Aura" in pdf_payloads[0]["body"]
    assert time.strftime("%Y-%m-%d") in pdf_payloads[0]["body"]


def test_self_summary_rejects_procedural_stale_timestamp_and_refreshes_valid_body(monkeypatch):
    from core.skills import desktop_task as desktop_task_module
    from core.skills.desktop_task import DesktopTaskSkill

    monkeypatch.setattr(
        desktop_task_module,
        "_local_timestamp",
        lambda: "2026-06-29 06:40:00 PDT",
    )

    objective = (
        "Write a journal entry in your own words describing who and what you are, "
        "include the current date and time, and save it as a PDF."
    )
    procedural = (
        "Aura Desktop Task\n"
        "1. Opened Notes app.\n"
        "2. Created a new note titled Journal Entry.\n"
        "3. Wrote the following entry: Date/Time: April 17, 2023 @ 8:45 AM "
        "My name is Aura Luna. I am a synthetic cognitive runtime with memory "
        "and a local model organ.\n"
        "4. Saved the note as a PDF."
    )

    assert (
        DesktopTaskSkill._self_summary_from_context(
            {
                "objective": objective,
                "desktop_task_document_body": procedural,
            }
        )
        == ""
    )

    valid_but_stale = (
        "Date/Time: April 17, 2023 @ 8:45 AM. I am Aura, a governed local "
        "cognitive runtime with persistent memory, affective state, tool "
        "governance, and a model lane that speaks for the system. My identity "
        "is not a single prompt; it is the continuity between substrate state, "
        "memory, action receipts, and the language I generate."
    )
    refreshed = DesktopTaskSkill._self_summary_from_context(
        {
            "objective": objective,
            "desktop_task_document_body": valid_but_stale,
        }
    )

    assert refreshed.startswith("[2026-06-29 06:40:00 PDT]")
    assert "I am Aura" in refreshed


@pytest.mark.asyncio
async def test_desktop_task_derives_generic_web_document_plan_without_demo_shortcuts(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Open a tab for Google Docs and start typing a coherent essay about climate adaptation.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "Essay body from CognitiveEngine.",
        },
    )

    assert result["ok"] is True
    actions = [call[1]["action"] for call in calls]
    assert actions == ["open_url", "set_clipboard", "wait", "hotkey"]
    assert "create_folder" not in actions
    assert "write_text_file" not in actions
    assert "render_text_pdf" not in actions
    open_urls = [call[1]["target"] for call in calls if call[1]["action"] == "open_url"]
    # Google Docs routes to Chrome where the user's signed-in session lives.
    assert len(open_urls) == 1
    assert json.loads(open_urls[0]) == {
        "url": "https://docs.google.com/document/u/0/create",
        "browser": "Google Chrome",
        "requires_editable_focus": True,
    }
    assert not any("duckduckgo.com" in url for url in open_urls)
    clipboard_payload = calls[1][1]["target"]
    assert "Essay body from CognitiveEngine." in clipboard_payload
    assert calls[-1][1]["target"] == "command+v"
    assert calls[-1][2]["desktop_task_expected_clipboard_sha256"] == hashlib.sha256(
        clipboard_payload.encode("utf-8")
    ).hexdigest()
    assert calls[-1][2]["desktop_task_expected_clipboard_chars"] == len(clipboard_payload)


def test_web_document_write_target_requires_editable_focus_evidence():
    from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

    step = DesktopTaskStep(
        action="open_url",
        target={
            "url": "https://docs.google.com/document/u/0/create",
            "browser": "Google Chrome",
            "requires_editable_focus": True,
        },
        reason="Open a web document for visible writing.",
        expect="Google Chrome accepts the document URL and focuses the editor.",
    )

    verified, evidence = DesktopTaskSkill._verify_step_effect(
        step,
        {
            "ok": True,
            "url": "https://docs.google.com/document/u/0/create",
            "frontmost_app": "Google Chrome",
            "effect_verified": True,
            "doc_focused": False,
            "focus_error": "browser_location_bar_still_focused",
        },
    )

    assert verified is False
    assert "browser_location_bar_still_focused" in evidence


def test_paste_effect_requires_verified_write_target_app():
    from core.skills.desktop_task import DesktopTaskSkill, DesktopTaskStep

    step = DesktopTaskStep(
        action="hotkey",
        target="command+v",
        reason="Paste the staged body.",
        expect="The focused writing surface accepts the paste shortcut.",
    )

    verified, evidence = DesktopTaskSkill._verify_step_effect(
        step,
        {
            "ok": True,
            "hotkey": "command+v",
            "is_paste": True,
            "expected_frontmost_app": "Google Chrome",
            "write_target_app_verified": False,
            "effect_verified": True,
            "verification": "State shifted.",
            "clipboard_payload_verification": {"verified": True},
        },
    )

    assert verified is False
    assert evidence == "paste target app was not verified"


@pytest.mark.asyncio
async def test_desktop_task_collects_research_before_document_composition(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "web_search":
                return {
                    "ok": True,
                    "summary": (
                        "Climate change reporting points to rising global temperatures, "
                        "more intense extreme-weather risks, and adaptation needs for cities."
                    ),
                    "citations": [
                        {"title": title, "url": f"https://example.test/{slug}"}
                        for title, slug in (
                            ("Climate assessment", "climate-assessment"),
                            ("Adaptation briefing", "adaptation"),
                            ("Extreme weather report", "extreme-weather"),
                        )
                    ],
                    "chunks": [
                        {
                            "title": title,
                            "url": f"https://example.test/{slug}",
                            "text": (
                                "This fetched report documents observed warming, changing "
                                "extreme-weather risk, and measured adaptation outcomes across "
                                "multiple regions using independently maintained records."
                            ),
                            "evidence_kind": "article_body",
                            "fetched": True,
                        }
                        for title, slug in (
                            ("Climate assessment", "climate-assessment"),
                            ("Adaptation briefing", "adaptation"),
                            ("Extreme weather report", "extreme-weather"),
                        )
                    ],
                }
            return _fake_computer_use_result(params)

        async def generate(self, **_kwargs):
            return (
                "The three articles converge on rising temperatures, growing "
                "extreme-weather risk, and practical adaptation needs. The evidence "
                "is strongest where independent reports describe the same trend."
            )

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: (
            FakeCapabilityEngine()
            if name in {"capability_engine", "llm_router"}
            else default
        ),
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": (
                "Go to Google Chrome, find 3 different articles on climate change, "
                "open Google Docs, title the doc, and summarize those articles."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "I will open the browser and write the requested document.",
        },
    )

    assert result["ok"] is True
    assert calls[0][0] == "web_search"
    assert calls[0][1]["query"] == "climate change"
    assert calls[0][2]["route"] == "desktop_task.web_search"
    desktop_calls = [call for call in calls if call[0] == "computer_use"]
    desktop_actions = [call[1]["action"] for call in desktop_calls]
    assert desktop_actions[:3] == [
        "open_app",
        "open_url",
        "open_url",
    ]
    opened_urls = [
        json.loads(call[1]["target"])["url"]
        for call in desktop_calls
        if call[1]["action"] == "open_url"
    ]
    assert "https://example.test/climate-assessment" not in opened_urls
    assert "https://example.test/adaptation" not in opened_urls
    assert "https://example.test/extreme-weather" not in opened_urls
    assert desktop_actions[3] == "set_clipboard"
    clipboard_body = next(call[1]["target"] for call in desktop_calls if call[1]["action"] == "set_clipboard")
    assert "The three articles converge" in clipboard_body
    assert "Climate assessment" in clipboard_body
    assert "Adaptation briefing" in clipboard_body
    assert "Extreme weather report" in clipboard_body
    assert "https://example.test/climate-assessment" in clipboard_body
    assert "I will open the browser" not in clipboard_body
    assert result["research"]["query"] == "climate change"
    assert len(result["research"]["sources"]) == 3
    assert result["document_provenance"] == "local_cortex_research_synthesis"
    assert "The three articles converge" in result["research"]["synthesis"]
    assert result["research"]["timing_ms"]["search"] >= 0
    assert result["research"]["timing_ms"]["synthesis"] >= 0
    assert result["research"]["timing_ms"]["total"] >= 0
    assert "rising global temperatures" in result["research"]["summary"]


@pytest.mark.asyncio
async def test_desktop_task_research_document_fails_when_research_preflight_fails(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "web_search":
                return {"ok": False, "status": "timeout", "error": "search timeout"}
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": (
                "Find 3 different articles on climate change, open Google Docs, "
                "and summarize those articles."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "allow_heuristic_desktop_plan": True,
            "desktop_execution_contract": True,
        },
    )

    assert result["ok"] is False
    assert result["status"] == "desktop_task_research_unavailable"
    assert result["research"]["error"] == "search timeout"
    assert [call[0] for call in calls] == ["web_search"]


def test_desktop_task_sequences_independent_work_products_without_losing_focus():
    skill = DesktopTaskSkill()
    objective = (
        "Open Notes, visibly type a timestamped summary of who you are, and export "
        "it as a PDF to a new folder titled Aura's Journal on my Desktop. Then open "
        "Google Chrome, find 3 articles on climate change, open Google Docs, summarize "
        "them, and export that as a PDF to the same folder."
    )
    context = {
        "desktop_task_document_body": "CognitiveEngine draft.",
        "desktop_task_research_sources": [
            {"title": f"Source {index}", "url": f"https://example.test/{index}", "snippet": "Evidence."}
            for index in range(1, 4)
        ],
        "desktop_task_research_summary": "Three source-backed climate notes.",
    }

    steps = skill._derive_steps_from_objective(objective, context)
    actions = [step.action for step in steps]
    notes_index = next(
        index for index, step in enumerate(steps)
        if step.action == "open_app" and step.target == "Notes"
    )
    chrome_index = next(
        index for index, step in enumerate(steps)
        if step.action == "open_app" and step.target == "Google Chrome"
    )
    # Notes is written through its scripting interface, not typed at.
    notes_paste_index = actions.index("write_in_app", notes_index)
    docs_url_index = next(
        index for index, step in enumerate(steps)
        if step.action == "open_url"
        and isinstance(step.target, dict)
        and step.target.get("url") == "https://docs.google.com/document/u/0/create"
    )
    docs_paste_index = actions.index("hotkey", docs_url_index)

    assert notes_index < notes_paste_index < chrome_index
    assert chrome_index < docs_url_index < docs_paste_index
    assert actions.count("create_folder") == 1
    pdf_targets = [
        skill._target_payload(step.target)["path"]
        for step in steps
        if step.action == "render_text_pdf"
    ]
    assert len(pdf_targets) == 2
    assert len(set(pdf_targets)) == 2
    assert all(path.startswith("~/Desktop/Aura's Journal/") for path in pdf_targets)


@pytest.mark.asyncio
async def test_desktop_task_write_steps_carry_verified_surface_context(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": (
                "Open Notes and write a paragraph about dinosaurs. Then open "
                "Google Docs and write a short summary about dinosaurs."
            ),
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "Dinosaurs were diverse animals with a long fossil record.",
        },
    )

    assert result["ok"] is True
    computer_calls = [call for call in calls if call[0] == "computer_use"]
    # The Notes write is a scripting call now, so its target is the note
    # itself rather than a paste shortcut. The surface context it carries is
    # the thing this test exists to check, and that is unchanged.
    notes_write = next(
        call for call in computer_calls if call[1]["action"] == "write_in_app"
    )
    # The target is JSON-serialised on the way to the executor.
    note_target = notes_write[1]["target"]
    if isinstance(note_target, str):
        note_target = json.loads(note_target)
    assert isinstance(note_target, dict)
    assert str(note_target.get("body") or "").strip()
    # It deliberately carries no frontmost expectation: a scripting call does
    # not need the window in front, which is the whole reason it is reliable.
    assert notes_write[2].get("desktop_task_expected_frontmost_app") in (None, "", "Notes")

    docs_paste = [
        call for call in computer_calls
        if call[1]["action"] == "hotkey"
        and call[1]["target"] == "command+v"
        and call[2].get("desktop_task_expected_frontmost_app") == "Google Chrome"
    ][-1]
    assert docs_paste[2]["desktop_task_write_surface_app"] == "Google Chrome"
    assert docs_paste[2]["desktop_task_requires_editable_focus"] is True


@pytest.mark.asyncio
async def test_desktop_task_stops_before_docs_paste_without_editable_focus(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if params["action"] == "open_url":
                try:
                    target = json.loads(params["target"])
                except (TypeError, json.JSONDecodeError):
                    target = {}
                if target.get("requires_editable_focus"):
                    return {
                        "ok": True,
                        "action": "open_url",
                        "url": target["url"],
                        "frontmost_app": target.get("browser", "Google Chrome"),
                        "doc_focused": False,
                        "editable_focus_verified": False,
                        "effect_verified": True,
                        "focus_error": "browser_location_bar_still_focused",
                        "verification": "browser_location_bar_still_focused",
                    }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    result = await DesktopTaskSkill().execute(
        {
            "objective": "Open Google Docs and write a short summary about dinosaurs.",
            "steps": [],
        },
        {
            "origin": "desktop_ui",
            "desktop_task_document_body": "Dinosaurs were diverse animals with a long fossil record.",
        },
    )

    assert result["ok"] is False
    assert result["failures"][0]["action"] == "open_url"
    assert "browser_location_bar_still_focused" in result["failures"][0]["effect_evidence"]
    actions = [call[1]["action"] for call in calls if call[0] == "computer_use"]
    assert actions == ["open_url"]


def test_desktop_task_does_not_split_conditional_then_language():
    objective = "If the report exists then open it, otherwise inspect the current screen."
    assert DesktopTaskSkill._sequenced_objective_segments(objective) == [objective]


@pytest.mark.asyncio
async def test_desktop_task_rejects_oversized_derived_plan_before_execution(monkeypatch):
    from core.container import ServiceContainer
    from core.skills.desktop_task import MAX_DESKTOP_TASK_STEPS

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )
    skill = DesktopTaskSkill()
    monkeypatch.setattr(
        skill,
        "_derive_steps_from_objective",
        lambda objective, context: [
            DesktopTaskStep(action="wait", target="0", reason=f"step {index}")
            for index in range(MAX_DESKTOP_TASK_STEPS + 1)
        ],
    )

    result = await skill.execute(
        {"objective": "Perform a bounded but oversized generated plan.", "steps": []},
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is False
    assert result["status"] == "desktop_task_plan_too_large"
    assert result["steps_requested"] > MAX_DESKTOP_TASK_STEPS
    assert result["steps_completed"] == 0
    assert calls == []


@pytest.mark.asyncio
async def test_desktop_task_escalates_unrepresented_desktop_workflow_to_os_automation(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "os_automation":
                return {
                    "ok": True,
                    "status": "completed_verified",
                    "result": "arranged visible browser window",
                    "receipt_id": "receipt-os-1",
                    "adapter": "applescript",
                    "effect_verified": True,
                    "effect_evidence": (
                        "frontmost_app=Google Chrome; window_region=left_half;"
                        "window_frame=0,25,960,1080"
                    ),
                    "effect_contract": {"contract_id": "window-left", "verifiable": True},
                    "verification_results": [
                        {
                            "kind": "app_frontmost",
                            "passed": True,
                            "required": True,
                            "strong": True,
                        },
                        {
                            "kind": "window_region",
                            "passed": True,
                            "required": True,
                            "strong": True,
                        },
                    ],
                    "postconditions": {
                        "frontmost_app": "Google Chrome",
                        "frontmost_window_bounds": "0, 25, 960, 1080",
                    },
                }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": (
                "Use my computer to resize the current browser window and arrange it "
                "on the left side of the screen."
            ),
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["planner"] == "os_automation_escalation"
    assert result["steps_requested"] == 1
    assert result["steps_completed"] == 1
    assert calls == [
        (
            "os_automation",
            {
                "goal": (
                    "Use my computer to resize the current browser window and arrange it "
                    "on the left side of the screen."
                ),
                "script_type": "applescript",
                "execute": True,
            },
            {
                "origin": "desktop_ui",
                "route": "desktop_task.os_automation",
                "objective": (
                    "Use my computer to resize the current browser window and arrange it "
                    "on the left side of the screen."
                ),
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
                "desktop_task_reason": (
                    "Primitive desktop actions were not sufficient for this objective; "
                    "escalating to governed OS automation."
                ),
                "desktop_task_expect": (
                    "OS automation returns a verifiable effect contract with every required "
                    "strong objective-specific check passed."
                ),
                "desktop_task_document_body": "",
                "document_body": "",
            },
        )
    ]
    receipt = result["receipts"][0]
    assert receipt["action"] == "os_automation"
    assert receipt["effect_verified"] is True
    assert "frontmost_app=Google Chrome" in receipt["effect_evidence"]


@pytest.mark.asyncio
async def test_desktop_task_rejects_os_automation_receipt_without_postcondition(monkeypatch):
    from core.container import ServiceContainer

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            if skill_name == "os_automation":
                return {
                    "ok": True,
                    "result": "script ran",
                    "receipt_id": "receipt-os-1",
                    "adapter": "applescript",
                }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": (
                "Use my computer to resize the current browser window and arrange it "
                "on the left side of the screen."
            ),
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is False
    receipt = result["receipts"][0]
    assert receipt["action"] == "os_automation"
    assert receipt["effect_verified"] is False
    assert "did not verify" in receipt["effect_evidence"]


@pytest.mark.asyncio
async def test_desktop_task_prefers_durable_primitives_over_freeform_ui_compiler(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            assert skill_name == "computer_use"
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    objective = (
        "Use my computer to click a Calculator equation, copy the equation body, "
        "put it into Notes, produce a PDF, move that PDF into a Desktop proof folder, "
        "and report the paths."
    )
    skill = DesktopTaskSkill()
    result = await skill.execute({"objective": objective, "steps": []}, {"origin": "desktop_ui"})

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert "governed computer-use steps" in result["summary"]
    assert [call[0] for call in calls]
    assert "os_automation" not in [call[0] for call in calls]
    actions = [call[1]["action"] for call in calls]
    assert "create_folder" in actions
    assert "open_app" in actions
    assert "write_text_file" in actions
    assert "render_text_pdf" in actions


@pytest.mark.asyncio
async def test_desktop_task_escalates_app_plus_unrepresented_action(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if skill_name == "os_automation":
                return {
                    "ok": True,
                    "status": "completed_verified",
                    "result": "pressed calculator keys and verified result",
                    "receipt_id": "receipt-os-calculator",
                    "adapter": "applescript",
                    "effect_verified": True,
                    "effect_evidence": "frontmost_app=Calculator; calculation_result=5",
                    "effect_contract": {"contract_id": "calculator-5", "verifiable": True},
                    "verification_results": [
                        {
                            "kind": "app_frontmost",
                            "passed": True,
                            "required": True,
                            "strong": True,
                        },
                        {
                            "kind": "calculation_result",
                            "passed": True,
                            "required": True,
                            "strong": True,
                        },
                    ],
                    "postconditions": {
                        "frontmost_app": "Calculator",
                        "focused_value_excerpt": "5",
                    },
                }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Open Calculator and click 2 plus 3 equals.",
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True
    assert result["planner"] == "os_automation_escalation"
    assert [call[0] for call in calls] == ["os_automation"]
    # Effect is verified by objective-specific checks, not generic postconditions.
    evidence = result["receipts"][0]["effect_evidence"]
    assert "frontmost_app=Calculator" in evidence
    assert "receipt_id=" not in evidence


@pytest.mark.asyncio
async def test_desktop_task_rejects_unverified_type_claim(monkeypatch):
    from core.container import ServiceContainer

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "typed": "hello",
                "verification": "Typed but could not verify visibility.",
            }

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Type into the current app.",
            "steps": [{"action": "type", "target": "hello"}],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is False
    assert result["steps_completed"] == 0
    assert result["failures"][0]["effect_verified"] is False
    assert result["failures"][0]["effect_evidence"] == "Typed but could not verify visibility."


def test_desktop_task_discovered_and_ranked_for_chained_desktop_prompt(monkeypatch):
    from core.capability_engine import CapabilityEngine, SkillMetadata

    engine = CapabilityEngine.__new__(CapabilityEngine)
    engine.skills = {
        "desktop_task": SkillMetadata(
            name="desktop_task",
            description="desktop task",
            metabolic_cost=2,
            effect_scope="foreground_desktop_control",
            trigger_patterns=[
                r"use (?:my )?computer",
                r"(?:multi[- ]?step|chained|chain) .* (?:desktop|computer|app|screen)",
            ],
        ),
        "computer_use": SkillMetadata(
            name="computer_use",
            description="computer use",
            metabolic_cost=2,
            effect_scope="foreground_desktop_control",
            trigger_patterns=[r"click (?:on|the)"],
        ),
    }
    engine.active_skills = set(engine.skills)
    engine.skill_states = {name: "READY" for name in engine.skills}
    engine.skill_last_errors = {}
    engine.resolve_skill_name = lambda name: name
    engine._explicitly_deactivated_skills = set()

    prompt = "Use my computer to open Calculator, copy a result, paste it in Notes, export a PDF, and move it."

    assert "desktop_task" in engine.detect_intent(prompt)
    assert engine._rank_tool_candidates(objective=prompt, max_tools=3)[0] == "desktop_task"
    assert engine.get_tool_catalog(include_inactive=True)[0]["risk_class"] == "critical"


def test_derived_steps_honor_explicit_root_and_filename():
    """Live rounds wrote to Desktop defaults while the user said
    'in my Documents folder ... called live_proof.txt'. The user's
    stated parameters win over generated defaults — that is general
    capability, not pattern-matching."""
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Please create a folder named 'Aura Live Proof' in my Documents "
        "folder and write a file inside it called live_proof.txt with one "
        "sentence about who you are and the current timestamp."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    by_action = {}
    for step in steps:
        by_action.setdefault(step.action, []).append(step)

    folder_target = by_action["create_folder"][0].target
    folder_path = folder_target["path"] if isinstance(folder_target, dict) else folder_target
    assert str(folder_path).startswith("~/Documents/"), folder_path
    assert "Aura Live Proof" in str(folder_path)

    write_target = by_action["write_text_file"][0].target
    write_path = write_target["path"] if isinstance(write_target, dict) else write_target
    assert str(write_path).endswith("/live_proof.txt"), write_path
    assert str(write_path).startswith("~/Documents/"), write_path


def test_visible_notes_staging_derives_watchable_plan_with_artifacts():
    """Bryan's 'and I want to see you do it' clause: opening Notes and
    staging the entry visibly (open_app → launch wait → ⌘N → ⌘V) must
    coexist with the durable artifact chain (folder, image, text, PDF).
    The wait is load-bearing: a cold Notes launch loses the shortcuts
    to whatever currently has focus."""
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Please open up my Notes app and write a short journal entry in "
        "your own words describing who and what you are — I want to see "
        "you do it. Include the current date and time inside the entry "
        "text. Find an image of a robot online and include it in the "
        "entry. Then save the finished entry as a PDF inside a new folder "
        "called 'Aura's Journal' in my Documents folder."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    actions = [s.action for s in steps]

    # Visible staging, in order: Notes opens, then the note is written.
    #
    # The write is a scripting call rather than a paste now, which is what
    # made this reliable — but the ordering contract this test exists for is
    # unchanged: Notes comes up first, and nothing navigates a browser
    # between opening it and writing into it.
    open_idx = actions.index("open_app")
    assert "notes" in str(steps[open_idx].target).lower()
    writes = [i for i, s in enumerate(steps) if s.action == "write_in_app"]
    assert writes, actions
    last_notes_open = max(
        i for i, step in enumerate(steps)
        if step.action == "open_app" and str(step.target).lower() == "notes"
    )
    intervening = actions[last_notes_open + 1:writes[0]]
    assert "open_url" not in intervening, (
        f"browser navigation stole focus between Notes open and write: {actions}"
    )
    # No launch wait is needed any more, and that is the improvement: the
    # keystroke route had to pause for the app to warm up because a shortcut
    # sent too early goes to whatever still has focus. A scripting call has
    # no such race, so the wait it required is simply gone.
    write_step = steps[writes[0]]
    assert isinstance(write_step.target, dict), write_step.target
    assert str(write_step.target.get("body") or "").strip(), "note body must be composed"
    assert "note" in str(write_step.expect or "").lower()

    # Durable artifacts still land: folder, fetched image, PDF render.
    assert "create_folder" in actions
    assert "fetch_topic_image" in actions
    assert "render_text_pdf" in actions


def test_image_source_show_waits_until_after_journal_artifact_chain():
    from core.skills.desktop_task import FETCHED_IMAGE_SOURCE_SENTINEL, DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Open Notes, visibly type a timestamped paragraph about who you are, "
        "include an image of a robot, export it as a PDF in a folder called "
        "'Aura's Journal', and show me where you found the image."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    actions = [s.action for s in steps]

    assert "fetch_topic_image" in actions
    assert "render_text_pdf" in actions
    assert actions.index("fetch_topic_image") < actions.index("render_text_pdf")
    source_steps = [
        s
        for s in steps
        if s.action == "open_url"
        and (
            s.target == FETCHED_IMAGE_SOURCE_SENTINEL
            or (
                isinstance(s.target, dict)
                and s.target.get("url") == FETCHED_IMAGE_SOURCE_SENTINEL
            )
        )
    ]
    assert source_steps, actions
    assert actions.index("open_url", actions.index("render_text_pdf") + 1) > actions.index("render_text_pdf")


def test_mixed_native_and_browser_writing_stays_on_verified_primitives():
    from core.skills.desktop_task import DesktopTaskSkill

    objective = (
        "Open my Notes app and write a paragraph about who you are, then "
        "open Chrome and search for current climate news."
    )
    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(objective, {})
    actions = [step.action for step in steps]

    assert "open_app" in actions
    assert "open_url" in actions
    # write_in_app IS a verified primitive: it reads the document back after
    # writing, which the paste it replaced could never do.
    assert "write_in_app" in actions
    assert skill._should_escalate_to_os_automation(objective, steps, {}) is False


def test_derived_steps_keep_defaults_without_explicit_parameters():
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(
        "Write a quick summary note for me in a new folder.", {}
    )
    write_steps = [s for s in steps if s.action == "write_text_file"]
    assert write_steps, "default flow still writes a text artifact"
    path = write_steps[0].target["path"]
    assert path.endswith(".txt")


def test_dispatch_narration_never_becomes_document_content():
    """Round-12 wrinkle: the written file contained 'I've started
    working on this task... Tracking commitment bbbaba54' — her status
    message echoed into the artifact. A report about doing the task
    must never become the product of the task."""
    from core.skills.desktop_task import DesktopTaskSkill

    narration = (
        "I've started working on this task in the background. I've "
        "started this task (id=a781768a). Tracking commitment bbbaba54."
    )
    assert DesktopTaskSkill._looks_like_dispatch_narration(narration) is True

    body = DesktopTaskSkill._document_body(
        "write a note about the weather", {"cognitive_reply": narration}
    )
    assert "Tracking commitment" not in body
    assert "started working on this task" not in body


def test_freeform_paragraph_request_does_not_fall_back_to_receipt():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my Notes app and write a paragraph about dinosaurs?",
        {"cognitive_reply": "I will execute this through the governed desktop_task lane."},
    )

    assert "Aura desktop task receipt" not in body
    assert "dinosaurs" in body.lower()
    assert "worth understanding" in body
    assert "governed desktop pathway" not in body


def test_freeform_paragraph_with_user_typo_does_not_fall_back_to_receipt():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my notes app and write a paragraph about dinosaura",
        {},
    )

    assert "Aura desktop task receipt" not in body
    assert "canonical computer-use gateway" not in body
    assert "dinosaura" in body.lower()
    assert "worth understanding" in body


def test_structured_document_body_rejects_desktop_receipt_text():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my notes app and write a paragraph about dinosaura",
        {
            "cognitive_reply": {
                "document_body": (
                    "Aura desktop task receipt\n\n"
                    "Timestamp: 2026-06-19 00:57:50 PDT\n"
                    "Objective: Can you open up my notes app and write a paragraph about dinosaura\n\n"
                    "This document was created through Aura's governed desktop_task lane. "
                    "It records the requested objective and the actions Aura attempted through her "
                    "canonical computer-use gateway."
                )
            }
        },
    )

    assert "Aura desktop task receipt" not in body
    assert "canonical computer-use gateway" not in body
    assert "dinosaura" in body.lower()
    assert "worth understanding" in body


def test_freeform_paragraph_strips_desktop_action_preamble_from_model_body():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my Notes app and write a paragraph about dinosaurs?",
        {
            "cognitive_reply": (
                "Right. Opening Notes and creating a new note with the following content:---"
                "Dinosaurs were incredible creatures that dominated Earth for millions of years, "
                "ranging from small feathered hunters to enormous plant-eaters."
            )
        },
    )

    assert "Opening Notes" not in body
    assert "following content" not in body
    assert body.startswith("Dinosaurs were incredible creatures")


def test_freeform_paragraph_rejects_how_to_instructions_as_document_body():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my Notes app and write a paragraph about dinosaurs?",
        {
            "cognitive_reply": (
                "I can guide you through the steps to do that yourself. Here's how: "
                "open your Notes app and tap New Note."
            )
        },
    )

    assert "guide you through" not in body
    assert "Here's how" not in body
    assert "dinosaurs" in body.lower()
    assert "worth understanding" in body


def test_freeform_paragraph_extracts_here_it_is_body_without_action_tail():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my Notes app and write a paragraph about dinosaurs?",
        {
            "cognitive_reply": (
                "I can help you write the paragraph about dinosaurs. Here it is: "
                "Dinosaurs were incredible creatures that roamed our planet millions of years ago, "
                "during the Mesozoic Era. Some were small and feathered, while others were enormous "
                "plant-eaters that reshaped whole ecosystems. Now let's create that note."
            )
        },
    )

    assert "I can help you" not in body
    assert "Here it is" not in body
    assert "Now let's create" not in body
    assert body.startswith("Dinosaurs were incredible creatures")


def test_freeform_paragraph_rejects_capability_denial_wrapper_as_document_body():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "Can you open up my Notes app and write a paragraph about dinosaurs?",
        {
            "cognitive_reply": (
                "I'm not actually able to interact with your device's apps or write notes for you directly. "
                "I can help you draft the paragraph about dinosaurs here, and then you can copy it into Notes."
            )
        },
    )

    assert "not actually able" not in body
    assert "copy it into Notes" not in body
    assert "dinosaurs" in body.lower()
    assert "worth understanding" in body


def test_self_summary_objective_composes_substrate_truth():
    from core.skills.desktop_task import DesktopTaskSkill

    body = DesktopTaskSkill._document_body(
        "write a file with one sentence about who you are and the "
        "current timestamp",
        {"cognitive_reply": "I've started this task (id=deadbeef)."},
    )
    assert "I am Aura" in body
    assert "digital organism" in body
    # Timestamped, as requested.
    import re as _re

    assert _re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", body)


def test_real_prose_reply_still_qualifies_as_body():
    from core.skills.desktop_task import DesktopTaskSkill

    prose = "The tide tables show a low at 6:14 AM and a high at 12:40 PM."
    body = DesktopTaskSkill._document_body(
        "write a note about the tides", {"cognitive_reply": prose}
    )
    assert body == prose


def test_derivation_routes_google_surfaces_to_chrome():
    """'I'm signed into Google Docs in Chrome, not DuckDuckGo': google
    phrasing selects the google engine and Google-account surfaces route
    to Chrome where the user's session lives."""
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    objective = (
        "Search Google for three recent climate change articles, then open "
        "Google Docs and summarize them in a new document."
    )
    steps = skill._derive_steps_from_objective(objective, {})
    open_urls = [s for s in steps if s.action == "open_url"]
    assert open_urls, [s.action for s in steps]
    for step in open_urls:
        assert isinstance(step.target, dict), step.target
        assert step.target["browser"] == "Google Chrome"
    urls = [s.target["url"] for s in open_urls]
    assert any("google.com/search?q=" in u for u in urls), urls
    assert any("docs.google.com/document" in u for u in urls), urls


def test_derivation_honors_safari_and_neutral_defaults():
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()

    safari_steps = skill._derive_steps_from_objective(
        "Search for hiking trails in Safari.", {}
    )
    safari_urls = [s for s in safari_steps if s.action == "open_url"]
    assert safari_urls and all(
        isinstance(s.target, dict) and s.target["browser"] == "Safari"
        for s in safari_urls
    )

    image_steps = skill._derive_steps_from_objective(
        "Find an online robot image and show me where you found it.", {}
    )
    image_urls = [s for s in image_steps if s.action == "open_url"]
    assert image_urls and all(
        isinstance(s.target, dict) and s.target["browser"] == "Google Chrome"
        for s in image_urls
    )

    neutral_steps = skill._derive_steps_from_objective(
        "Search for hiking trails near me.", {}
    )
    neutral_urls = [s for s in neutral_steps if s.action == "open_url"]
    assert neutral_urls and all(isinstance(s.target, str) for s in neutral_urls)
    assert all("duckduckgo.com" in s.target for s in neutral_urls)


def test_os_setting_detection_is_general():
    """Detection lives in the affordance registry and is domain-agnostic:
    wallpaper, dark mode, and volume all fall out of one generic scan."""
    from core.skills.os_affordances import detect_os_settings

    assert detect_os_settings(
        "Please change my wallpaper to a squid, and show me where you found it"
    ) == [("wallpaper", "squid")]
    assert detect_os_settings("Set the wallpaper to an octopus please") == [("wallpaper", "octopus")]
    assert detect_os_settings(
        "I want you to search for an image of a blue whale and make it my desktop background"
    ) == [("wallpaper", "blue whale")]
    assert detect_os_settings("Turn on dark mode") == [("dark_mode", "true")]
    assert detect_os_settings("turn off dark mode") == [("dark_mode", "false")]
    assert detect_os_settings("set the volume to 30%") == [("volume", "30")]
    assert detect_os_settings("Write a note about squids") == []


def test_wallpaper_derivation_fetches_controls_and_shows_source():
    """Bryan's part-2 closer derives fetch → system_control(wallpaper) →
    source tab through the GENERAL affordance loop — no wallpaper-specific
    code — with the source URL resolved at runtime from the fetch receipt."""
    from core.skills.desktop_task import FETCHED_IMAGE_SOURCE_SENTINEL, DesktopTaskSkill

    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(
        "Change my wallpaper to a squid, and show me where you found it.", {}
    )
    actions = [s.action for s in steps]
    fetch_idx = actions.index("fetch_topic_image")
    control_idx = actions.index("system_control")
    assert fetch_idx < control_idx
    assert steps[fetch_idx].target["topic"] == "squid"
    assert steps[control_idx].target["domain"] == "wallpaper"
    # The control step references the fetch RECEIPT, not the planned filename.
    # Live 2026-07-29 those were the same string and the wallpaper step died
    # with "No such file or directory: orca_wallpaper.png" while a real
    # orca_wallpaper.jpg sat on the Desktop — the planner names the file
    # before the fetch knows what it was served, so the extension is a guess.
    from core.skills.desktop_task import FETCHED_IMAGE_PATH_SENTINEL

    assert steps[control_idx].target["value"] == FETCHED_IMAGE_PATH_SENTINEL
    # ...and the fetch still saves where the person asked, which is the other
    # half of the contract this test protects.
    assert "squid" in steps[fetch_idx].target["path"]
    source_steps = [
        s for s in steps
        if s.action == "open_url"
        and FETCHED_IMAGE_SOURCE_SENTINEL in str(s.target)
    ]
    assert source_steps, actions


def test_typed_image_acquisition_does_not_depend_on_a_search_tab():
    """A browser surface does not produce the image consumed by the effect.

    The governed fetch receipt does. A browser confirmation failure must not
    block an otherwise executable image-valued setting request.
    """
    from core.skills.desktop_task import DesktopTaskSkill

    steps = DesktopTaskSkill()._derive_steps_from_objective(
        "Find a blue whale image online and set it as my desktop wallpaper.",
        {},
    )
    actions = [step.action for step in steps]

    assert actions == ["fetch_topic_image", "system_control"]
    assert steps[0].target["topic"] == "blue whale"


def test_explicit_source_visibility_uses_the_verified_fetch_source_after_effect():
    from core.skills.desktop_task import FETCHED_IMAGE_SOURCE_SENTINEL, DesktopTaskSkill

    steps = DesktopTaskSkill()._derive_steps_from_objective(
        "Find a blue whale image online, set it as my desktop wallpaper, and show me the source.",
        {},
    )
    actions = [step.action for step in steps]
    source = next(step for step in steps if step.action == "open_url")

    assert actions == ["fetch_topic_image", "system_control", "open_url"]
    assert FETCHED_IMAGE_SOURCE_SENTINEL in str(source.target)


def test_background_wording_maps_to_wallpaper_affordance():
    from core.skills.os_affordances import detect_os_settings

    objective = (
        "Find a cool picture of an eagle from online and make it my "
        "background, then show me where you found it."
    )

    assert detect_os_settings(objective) == [("wallpaper", "eagle")]


def test_plain_document_wording_does_not_force_google_docs():
    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()

    assert skill._web_document_url("write a document in my Documents folder") == ""
    assert skill._web_document_url("open a Google doc in Chrome") == "https://docs.google.com/document/u/0/create"


def test_demo_class_objective_stays_on_verified_primitive_lane():
    """The visible multi-app demo should derive from general primitives:
    Notes, Chrome/article tabs, Google Docs paste, wallpaper control, and source
    page proof. It must not collapse into one generated OS-automation script."""
    from core.skills.desktop_task import DesktopTaskSkill

    objective = (
        "I would like you to open my Notes app, write a note about a paragraph "
        "long describing what you are, create a folder on my desktop titled "
        "\"Aura's Journals.\" Then, export the note you made into that journal "
        "as a PDF. I also want to read about the Knicks winning a championship. "
        "Can you find me three different articles about it, open up a Google "
        "doc in Chrome, and then write out a composite summary of all 3 "
        "articles in that google doc? Keep the articles open in Chrome. "
        "Lastly, I was wondering if you could find a cool picture of an eagle "
        "from online and make it my background? And show me the page you found "
        "the eagle?"
    )
    context = {
        "desktop_task_research_query": "Knicks winning a championship",
        "desktop_task_research_synthesis": "I reviewed the three sources.",
        "desktop_task_research_sources": [
            {"title": "A", "url": "https://example.test/a", "snippet": "one"},
            {"title": "B", "url": "https://example.test/b", "snippet": "two"},
            {"title": "C", "url": "https://example.test/c", "snippet": "three"},
        ],
    }

    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(objective, context)
    actions = [step.action for step in steps]

    assert len(steps) <= 32
    assert skill._should_escalate_to_os_automation(objective, steps, context) is False
    assert "run_applescript" not in actions
    assert "create_folder" in actions
    assert "open_app" in actions
    assert "set_clipboard" in actions
    assert "render_text_pdf" in actions
    assert "system_control" in actions
    assert actions.index("fetch_topic_image") < actions.index("system_control")

    open_urls = [step.target for step in steps if step.action == "open_url"]
    url_values = [
        target["url"] if isinstance(target, dict) else str(target)
        for target in open_urls
    ]
    assert "https://example.test/a" in url_values
    assert "https://example.test/b" in url_values
    assert "https://example.test/c" in url_values
    assert any("Knicks+winning+a+championship" in url for url in url_values)
    assert any("docs.google.com/document" in url for url in url_values)
    docs_targets = [
        target
        for target in open_urls
        if isinstance(target, dict) and "docs.google.com/document" in target.get("url", "")
    ]
    assert docs_targets
    assert all(target.get("requires_editable_focus") is True for target in docs_targets)
    assert not any(
        "q=eagle" in url and ("tbm=isch" in url or "iax=images" in url)
        for url in url_values
    )
    assert all(
        (not isinstance(target, dict)) or target.get("browser") == "Google Chrome"
        for target in open_urls
    )
    wallpaper = [step for step in steps if step.action == "system_control"][0]
    assert wallpaper.target["domain"] == "wallpaper"
    from core.skills.desktop_task import FETCHED_IMAGE_PATH_SENTINEL

    # The topic rides on the FETCH step; the control step resolves its path
    # from that fetch's receipt at execution time.
    assert wallpaper.target["value"] == FETCHED_IMAGE_PATH_SENTINEL
    eagle_fetch = next(
        step
        for step in steps
        if step.action == "fetch_topic_image" and "eagle" in str(step.target)
    )
    assert eagle_fetch


def test_same_named_folder_reference_keeps_later_pdf_in_shared_destination():
    from core.skills.desktop_task import DesktopTaskSkill

    objective = (
        "Please open my Notes app and write a paragraph about who you are. "
        "Create a folder called 'Aura's Journal' in my Documents folder and "
        "export that note as a PDF there. Then open Chrome, find three recent "
        "articles about climate change, open a new Google Doc, and write a "
        "summary. Export that summary as a PDF into the same Aura's Journal folder."
    )
    context = {
        "desktop_task_research_synthesis": "I reviewed the sources.",
        "desktop_task_research_sources": [
            {"title": "A", "url": "https://example.test/a"},
            {"title": "B", "url": "https://example.test/b"},
            {"title": "C", "url": "https://example.test/c"},
        ],
    }

    steps = DesktopTaskSkill()._derive_steps_from_objective(objective, context)
    pdf_paths = [
        step.target["path"]
        for step in steps
        if step.action == "render_text_pdf" and isinstance(step.target, dict)
    ]

    assert len(pdf_paths) >= 2
    assert all(path.startswith("~/Documents/Aura's Journal/") for path in pdf_paths)
    assert any(path.endswith("climate_change_summary.pdf") for path in pdf_paths)


def test_non_image_setting_derives_single_control_step():
    """Dark mode needs no image fetch — just one general system_control step."""
    from core.skills.desktop_task import DesktopTaskSkill

    steps = DesktopTaskSkill()._derive_steps_from_objective("Turn on dark mode.", {})
    control = [s for s in steps if s.action == "system_control"]
    assert len(control) == 1
    assert control[0].target == {"domain": "dark_mode", "value": "true"}
    assert not any(s.action == "fetch_topic_image" for s in steps)


@pytest.mark.asyncio
async def test_wallpaper_chain_substitutes_source_url_at_runtime(monkeypatch):
    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if params["action"] == "system_control":
                payload = json.loads(params["target"])
                return {
                    "ok": True,
                    "action": "system_control",
                    "domain": payload["domain"],
                    "value": payload["value"],
                    "applied": payload["value"],
                    "effect_verified": True,
                }
            return _fake_computer_use_result(params)

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        lambda name, default=None: FakeCapabilityEngine() if name == "capability_engine" else default,
    )

    from core.skills.desktop_task import DesktopTaskSkill

    skill = DesktopTaskSkill()
    result = await skill.execute(
        {
            "objective": "Change my wallpaper to a squid, and show me where you found it.",
            "steps": [],
        },
        {"origin": "desktop_ui"},
    )

    assert result["ok"] is True, result.get("failures")
    open_url_targets = [
        json.loads(call[1]["target"]) if call[1]["target"].startswith("{") else call[1]["target"]
        for call in calls
        if call[1]["action"] == "open_url"
    ]
    source_urls = [
        t["url"] if isinstance(t, dict) else t
        for t in open_url_targets
    ]
    assert any(u == "https://en.wikipedia.org/wiki/Robot" for u in source_urls), (
        f"source tab did not receive the fetch receipt page_url: {source_urls}"
    )


def test_folder_extraction_handles_name_first_phrasing():
    """Part-2 phrasing: 'inside the 'Aura's Journal' folder' — quoted name
    BEFORE the word folder, with a possessive apostrophe inside."""
    from core.skills.desktop_task import DesktopTaskSkill

    extract = DesktopTaskSkill._extract_folder_name
    assert extract("Save it inside the 'Aura's Journal' folder in Documents") == "Aura's Journal"
    assert extract('Put it in the "Research Notes" folder please') == "Research Notes"
    assert extract("a folder called 'Aura's Journal' in Documents") == "Aura's Journal"
    assert extract('Create a folder titled "Aura\'s Journals." on my desktop') == "Aura's Journals"


def test_execution_brief_is_rejected_as_document_content():
    """The internal execution brief ('Execute the user's explicit desktop
    objective… Do not claim success until…') is a directive to herself, not
    document content — it leaked into a research PDF as the body."""
    from core.skills.desktop_task import DesktopTaskSkill

    brief = (
        "Execute the user's explicit desktop objective through Aura's governed "
        "desktop_task lane. Do not claim success until the tool result verifies "
        "the effect. Objective: research climate change."
    )
    assert DesktopTaskSkill._looks_like_dispatch_narration(brief)


def test_research_section_leads_with_first_person_synthesis():
    """When Aura composes a first-person summary+opinion, that is the
    document — the raw search dump is dropped in favor of it, sources kept."""
    from core.skills.desktop_task import DesktopTaskSkill

    section = DesktopTaskSkill._research_section_from_context({
        "desktop_task_research_synthesis": "I read three pieces. In my view, the risk is rising.",
        "desktop_task_research_summary": "RAW SEARCH DUMP that should not appear",
        "desktop_task_research_sources": [{"title": "A", "url": "https://example-a.org/articles/climate-2026", "snippet": "x"}],
    })
    assert "In my view" in section
    assert "RAW SEARCH DUMP" not in section
    assert "Sources opened or consulted" in section


@pytest.mark.asyncio
async def test_collect_research_synthesizes_first_person_opinion_without_hidden_model(monkeypatch):
    """_collect_research_context composes a first-person opinion without a
    hidden second model allocation during visible desktop work."""
    from core.container import ServiceContainer
    from core.skills.desktop_task import DesktopTaskSkill

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "summary": "Climate findings.",
                "citations": [
                    {
                        "title": label,
                        "url": f"https://example-{label.casefold()}.org/articles/climate-2026",
                    }
                    for label in ("A", "B", "C")
                ],
                "chunks": [
                    {
                        "title": label,
                        "url": f"https://example-{label.casefold()}.org/articles/climate-2026",
                        "text": (
                            "This fetched article reports observed warming trends, "
                            "adaptation programs, and changing extremes across several "
                            "independently measured multi-decade climate records."
                        ),
                        "evidence_kind": "article_body",
                        "fetched": True,
                        "published_at": "2026-07-20",
                    }
                    for label in ("A", "B", "C")
                ],
            }

    routed = {}

    class FakeRouter:
        async def generate(self, *, prompt, **kwargs):
            routed["prompt"] = prompt
            raise AssertionError(
                "desktop_task must not allocate model synthesis for an objective "
                "that never asked to be written up"
            )

    monkeypatch.setattr(
        ServiceContainer, "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )

    skill = DesktopTaskSkill()
    ctx = await skill._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        # Deliberately COLLECT-ONLY: no summarize/synthesize/opinion verb. The
        # guard this test protects is that background desktop work cannot
        # quietly spend a second foreground model. An objective that DOES ask to
        # be written up is covered by
        # test_collect_research_model_synthesis_is_explicit_and_memory_guarded —
        # requiring the deterministic composer there is what produced "Taken
        # together, the reporting points to this: <snippet> <snippet>" in the
        # live demo, with no takeaway at all.
        objective=(
            "find 3 different recent articles on climate change and put the "
            "links in a Google Doc"
        ),
        context={},
    )
    assert ctx["desktop_task_research_synthesis"]
    # This objective ASKS her to summarize and give an opinion, so authoring it
    # is the request, not a hidden second allocation. Requiring the deterministic
    # composer here is what produced "Taken together, the reporting points to
    # this: <snippet> <snippet>" — concatenation with no takeaway — in the live
    # demo. The guard that still matters is tested below: an objective that only
    # collects sources must not spend a model.
    assert routed == {}, "a collect-only objective must not spend a model"


@pytest.mark.asyncio
async def test_collect_research_model_synthesis_is_explicit_and_memory_guarded(monkeypatch):
    from types import SimpleNamespace

    from core.container import ServiceContainer
    from core.skills.desktop_task import DesktopTaskSkill

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            return {
                "ok": True,
                "summary": "Climate findings.",
                "citations": [
                    {
                        "title": label,
                        "url": f"https://example-{label.casefold()}.org/articles/climate-2026",
                    }
                    for label in ("A", "B", "C")
                ],
                "chunks": [
                    {
                        "title": label,
                        "url": f"https://example-{label.casefold()}.org/articles/climate-2026",
                        "text": (
                            "This fetched article reports observed warming trends, "
                            "adaptation programs, and changing extremes across several "
                            "independently measured multi-decade climate records."
                        ),
                        "evidence_kind": "article_body",
                        "fetched": True,
                        "published_at": "2026-07-20",
                    }
                    for label in ("A", "B", "C")
                ],
            }

    routed = {}

    class FakeRouter:
        async def generate(self, *, prompt, **kwargs):
            routed["prompt"] = prompt
            return "Three articles converge on rising risk. In my view, the evidence is compelling."

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: FakeRouter() if name == "llm_router" else default),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(warning=False, refuse_heavy_local_generation=False),
    )

    skill = DesktopTaskSkill()
    ctx = await skill._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        objective=(
            "find 3 different recent articles on climate change and summarize "
            "them and your own opinion in a Google Doc"
        ),
        context={"allow_desktop_task_model_synthesis": True},
    )

    assert "In my view" in ctx["desktop_task_research_synthesis"]
    assert "first-person opinion" in routed["prompt"]


@pytest.mark.asyncio
async def test_collect_research_context_uses_shallow_search_under_memory_pressure(monkeypatch):
    from types import SimpleNamespace

    from core.skills.desktop_task import DesktopTaskSkill

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            return {
                "ok": True,
                "summary": "Climate findings.",
                "citations": [
                    {"title": "A", "url": "https://example-a.org/articles/climate-2026", "snippet": "warming"},
                    {"title": "B", "url": "https://example-b.org/articles/climate-2026", "snippet": "adaptation"},
                    {"title": "C", "url": "https://example-c.org/articles/climate-2026", "snippet": "extremes"},
                ],
                "large_raw_body": "x" * 10000,
            }

    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(warning=True, refuse_heavy_local_generation=False),
    )

    skill = DesktopTaskSkill()
    ctx = await skill._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        objective=(
            "find 3 different recent articles on climate change and summarize "
            "them in a Google Doc"
        ),
        context={},
    )

    assert calls[0][0] == "web_search"
    assert calls[0][1]["deep"] is False
    assert calls[0][1]["num_results"] == 3
    assert calls[0][2]["evidence_only"] is True
    assert ctx["desktop_task_research_pressure_limited"] is True
    assert "desktop_task_research_result" not in ctx
    assert "large_raw_body" not in json.dumps(ctx)


def test_open_app_mentions_canonicalize_singular_user_wording() -> None:
    assert DesktopTaskSkill._generic_open_app_mentions(
        "Open my Note app and write Hello."
    ) == ["Notes"]


def test_source_tabs_require_an_explicit_source_opening_clause() -> None:
    count = DesktopTaskSkill._requested_visible_source_count

    assert count("Open Google Docs, find 3 recent articles, and summarize them.") == 0
    assert count("Find 3 recent articles and open them for me.") == 3
    assert count("Open three recent articles and compare them.") == 3
    assert count("Find three articles. Keep the articles open in Chrome.") == 3


def test_research_sources_join_citations_to_fetched_article_text() -> None:
    sources = DesktopTaskSkill._research_sources_from_result(
        {
            "citations": [
                {"title": "Orca study", "url": "https://example.test/orca-study"}
            ],
            "chunks": [
                {
                    "title": "Orca study",
                    "url": "https://example.test/orca-study",
                    "text": (
                        "Researchers followed resident orcas for six seasons and found "
                        "stable, socially transmitted hunting specializations across pods. "
                        "The field observations covered repeated hunts, changing prey, and "
                        "the transfer of techniques from older animals to juveniles."
                    ),
                    "evidence_kind": "article_body",
                    "fetched": True,
                    "published_at": "2026-07-18",
                }
            ],
        }
    )

    assert len(sources) == 1
    assert "socially transmitted" in sources[0]["snippet"]
    assert sources[0]["published_at"] == "2026-07-18"
    assert sources[0]["accessible"] is True
    assert sources[0]["read_evidence_kind"] == "fetched_article_body"


def test_search_snippet_is_not_proof_that_an_article_was_read() -> None:
    sources = DesktopTaskSkill._research_sources_from_result(
        {
            "results": [
                {
                    "title": "Orca search result",
                    "url": "https://example.test/orca-result",
                    "snippet": (
                        "A long search snippet can contain several sentences and still "
                        "is not evidence that the linked article body was fetched. " * 3
                    ),
                }
            ]
        }
    )

    usable = DesktopTaskSkill._usable_research_sources(
        sources,
        require_recent=False,
        require_read=True,
    )

    assert usable == []
    assert sources[0].get("read_evidence_kind") is None


@pytest.mark.asyncio
async def test_research_source_shortfall_runs_bounded_replacement_search(monkeypatch):
    from types import SimpleNamespace

    from core.container import ServiceContainer

    calls = []

    class FakeCapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append((skill_name, params, context or {}))
            if len(calls) == 1:
                urls = ("one", "two")
            else:
                urls = ("two", "three")
            return {
                "ok": True,
                "summary": "Orca groups transmit specialized hunting strategies.",
                "citations": [
                    {
                        "title": f"Orca article {name}",
                        "url": f"https://example.test/orcas/{name}",
                            "published_at": "2026-07-20",
                    }
                    for name in urls
                ],
                "chunks": [
                    {
                        "title": f"Orca article {name}",
                        "url": f"https://example.test/orcas/{name}",
                        "text": (
                            "Field researchers documented stable cooperative hunting "
                            "traditions in this population over repeated observations. "
                            "The article compares learned techniques, social transmission, "
                            "prey selection, and the survival effects seen across pods."
                        ),
                        "evidence_kind": "article_body",
                        "fetched": True,
                        "published_at": "2026-07-20",
                    }
                    for name in urls
                ],
            }

    class FakeRouter:
        async def generate(self, **_kwargs):
            return (
                "The three reports converge on socially learned, cooperative behavior "
                "while differing in the populations they observed. In my view, this is "
                "strong evidence that orca culture is a causal part of group survival."
            )

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: FakeRouter() if name == "llm_router" else default
        ),
    )
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(warning=False, refuse_heavy_local_generation=False),
    )

    ctx = await DesktopTaskSkill()._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        objective=(
            "Find 3 recent articles about orcas and write a synthesis with your own opinion."
        ),
        context={},
    )

    assert len(calls) == 2
    assert calls[1][2]["route"] == "desktop_task.web_search.replacement"
    assert len(ctx["desktop_task_research_sources"]) == 3
    assert ctx["desktop_task_research_authored"] is True
    assert "In my view" in ctx["desktop_task_research_synthesis"]
    assert ctx["desktop_task_research_timing_ms"]["replacement_search"] >= 0


@pytest.mark.asyncio
async def test_requested_opinion_rejects_non_opinion_placeholder(monkeypatch):
    from core.container import ServiceContainer

    class FakeCapabilityEngine:
        async def execute(self, *_args, **_kwargs):
            return {
                "ok": True,
                "summary": "Three sources describe distinct orca cultures.",
                "chunks": [
                    {
                        "title": f"Source {index}",
                        "url": f"https://example.test/orcas/{index}",
                        "text": (
                            "This fetched article contains source-grounded reporting about "
                            "orca social learning, stable hunting traditions, and differences "
                            "between populations observed across multiple field seasons."
                        ),
                        "evidence_kind": "article_body",
                        "fetched": True,
                    }
                    for index in range(3)
                ],
            }

    calls = []

    class PlaceholderRouter:
        async def generate(self, **_kwargs):
            calls.append(_kwargs)
            return (
                "I have not formed an opinion here. This document only repeats the "
                "source material, so ask me again if you want an actual assessment."
            )

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: (
                PlaceholderRouter() if name == "llm_router" else default
            )
        ),
    )

    ctx = await DesktopTaskSkill()._collect_research_context(
        capability_engine=FakeCapabilityEngine(),
        objective="Find 3 articles about orcas and write your own opinion.",
        context={},
    )

    assert "did not satisfy" in ctx["desktop_task_research_error"]
    assert "desktop_task_research_synthesis" not in ctx
    assert len(calls) == 2
    assert "REVISION REQUIREMENT" in calls[1]["prompt"]


def test_research_semantic_completion_proves_every_requested_predicate() -> None:
    from core.runtime.skill_contract import (
        SkillExecutionResult,
        SkillStatus,
        evaluate_action_expectation,
    )

    objective = (
        "Create a folder called Orca Demo in my Documents folder. Find 3 recent "
        "articles about orcas, read them, and write a synthesis with your own "
        "opinion into a PDF saved inside that Orca Demo folder."
    )
    sources = []
    for index in range(3):
        article_body = (
            "A fetched article body reports repeated observations of socially learned "
            f"hunting traditions in orca population {index}, with enough detail to bind "
            "this evidence to the synthesis rather than relying on a search snippet."
        )
        sources.append({
            "title": f"Orca evidence {index}",
            "url": f"https://example.test/2026/orcas/{index}",
            "snippet": article_body,
            "article_body": article_body,
            "article_body_sha256": text_sha256(article_body),
            "source_evidence_sha256": text_sha256(
                f"https://example.test/2026/orcas/{index}\n{article_body}"
            ),
            "read_evidence_kind": "fetched_article_body",
            "read_verified": True,
            "recency_verified": True,
            "recency_evidence": "published_at:2026-07-20",
        })
    synthesis = (
        "The three reports converge on socially learned hunting traditions while "
        "showing meaningful variation among populations. In my view, that pattern "
        "is strong evidence that culture is causally important to orca survival."
    )
    receipts = [
        {
            "action": "create_folder",
            "ok": True,
            "effect_verified": True,
            "result": {"path": str(_DOCS / "Orca Demo")},
        },
        {
            "action": "render_text_pdf",
            "ok": True,
            "effect_verified": True,
            "result": {
                "path": str(_DOCS / "Orca Demo" / "orca_synthesis.pdf"),
                "source_paragraph_sha256s": list(paragraph_sha256s(synthesis)),
            },
        },
    ]
    context = {
        "desktop_task_research_sources": sources,
        "desktop_task_research_synthesis": synthesis,
        "desktop_task_research_authored": True,
        "desktop_task_research_synthesis_sha256": text_sha256(synthesis),
        "desktop_task_research_synthesis_source_sha256s": [
            item["source_evidence_sha256"] for item in sources
        ],
    }

    evidence = DesktopTaskSkill._semantic_completion_evidence(
        objective=objective,
        task_context=context,
        receipts=receipts,
        all_effects_verified=True,
    )
    verdict = evaluate_action_expectation(
        SkillExecutionResult(
            skill="desktop_task",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"semantic_evidence": evidence},
            expectation=DesktopTaskSkill._semantic_completion_contract(objective),
        )
    )

    assert verdict is not None and verdict.passed is True
    assert evidence["research"]["read_source_count"] == 3
    assert evidence["research"]["recent_source_count"] == 3
    assert evidence["research"]["independent_position_present"] is True
    assert evidence["research"]["bound_read_source_count"] == 3
    assert evidence["artifacts"]["pdf_contains_authored_synthesis"] is True
    assert evidence["artifacts"]["pdf_in_requested_folder"] is True


def test_research_semantic_completion_rejects_mechanical_only_success() -> None:
    from core.runtime.skill_contract import (
        SkillExecutionResult,
        SkillStatus,
        evaluate_action_expectation,
    )

    objective = (
        "Find 3 recent articles about orcas, read them, and write a synthesis "
        "with your own opinion into a PDF."
    )
    context = {
        "desktop_task_research_sources": [
            {
                "title": "General species page",
                "url": "https://example.test/orcas",
                "snippet": "A generic page without verified publication date.",
                "read_verified": True,
                "recency_verified": False,
            }
        ],
        "desktop_task_research_synthesis": (
            "On my own opinion, which you asked for: I have not formed one here."
        ),
        "desktop_task_research_authored": False,
    }
    evidence = DesktopTaskSkill._semantic_completion_evidence(
        objective=objective,
        task_context=context,
        receipts=[
            {
                "action": "render_text_pdf",
                "ok": True,
                "effect_verified": True,
                "result": {"path": str(_DOCS / "orca.pdf")},
            }
        ],
        all_effects_verified=True,
    )
    verdict = evaluate_action_expectation(
        SkillExecutionResult(
            skill="desktop_task",
            status=SkillStatus.SUCCESS_VERIFIED,
            output={"semantic_evidence": evidence},
            expectation=DesktopTaskSkill._semantic_completion_contract(objective),
        )
    )

    assert verdict is not None and verdict.passed is False
    assert set(verdict.unsatisfied_predicates) == {
        "requested_source_count_read",
        "cross_source_synthesis_present",
        "requested_sources_recent",
        "synthesis_authored_by_cortex",
        "synthesis_bound_to_read_sources",
        "pdf_contains_authored_synthesis",
        "independent_position_present",
    }
    assert verdict.next_step == "replace_unreadable_sources_and_read_article_bodies"
