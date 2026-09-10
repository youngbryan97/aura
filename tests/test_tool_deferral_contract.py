import pytest

from core.runtime.tool_result_contracts import compact_result_payload, tool_result_is_deferred


@pytest.mark.parametrize("result, expected", [
    ({"status": "deferred"}, True),
    ({"status": "deferred_by_executive"}, True),
    ({"deferred": True}, True),
    ({"error": "background_deferred:busy"}, True),
    ({"reason": "background_deferred:busy"}, True),
    ("background_deferred:busy", True),
    ({"ok": True, "deferred": False}, False),
    ({"ok": False, "error": "permission_denied"}, False),
    ({"ok": False, "retryable": True, "error": "connection_lost"}, False),
    ({"summary": "the previous operation was deferred"}, False),
])
def test_shared_deferral_contract_and_constitution_agree(result, expected):
    from core.constitution import _tool_result_is_deferred

    assert tool_result_is_deferred(result) is expected
    assert _tool_result_is_deferred(result) is expected


def test_compaction_retains_admission_reason_and_retry_contract():
    result = {"ok": False, "status": "deferred_by_executive", "deferred": True,
              "reason": "foreground_generation_active", "retryable": True, "retry_after_s": 4.0}
    assert compact_result_payload(result) == result
