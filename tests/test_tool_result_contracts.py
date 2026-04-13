from core.runtime.tool_result_contracts import compact_result_payload


def test_compact_result_payload_preserves_task_tracking_fields():
    payload = compact_result_payload(
        {
            "status": "started",
            "summary": "I started the task and attached a commitment.",
            "task_id": "task-123",
            "commitment_id": "commit-456",
            "objective": "Fix the failing pytest in core/runtime/conversation_support.py",
            "error": "",
        }
    )

    assert payload["status"] == "started"
    assert payload["task_id"] == "task-123"
    assert payload["commitment_id"] == "commit-456"
    assert payload["objective"] == "Fix the failing pytest in core/runtime/conversation_support.py"
    assert "started the task" in payload["summary"]


def test_compact_result_payload_preserves_execution_loop_signals():
    payload = compact_result_payload(
        {
            "status": "completed",
            "phase": "repairing",
            "active_step": "Re-run the failing pytest",
            "steps_completed": 2,
            "steps_total": 3,
            "repair_count": 1,
            "evidence": [
                "[Inspect failing assertion]: AssertionError: expected coding block",
                "[Re-run pytest]: 1 passed in 0.40s",
            ],
            "files": [
                "core/runtime/conversation_support.py",
                "tests/test_runtime_service_access.py",
            ],
        }
    )

    assert payload["phase"] == "repairing"
    assert payload["steps_completed"] == 2
    assert payload["steps_total"] == 3
    assert payload["repair_count"] == 1
    assert "conversation_support.py" in payload["files"][0]
    assert "AssertionError" in payload["evidence"][0]
