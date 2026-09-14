"""A partial check and a broken engine do not establish a conclusive pass."""

import pytest

from core.brain.verifiers.base import VerificationResult, combine_results


@pytest.mark.parametrize("checked,ok,infrastructure_failed,verdict", [
    (False, True, False, "UNCHECKED"),
    (True, True, False, "PASSED"),
    (True, False, False, "FAILED"),
    (False, True, True, "UNVERIFIABLE"),
    (True, True, True, "UNVERIFIABLE"),
    (True, False, True, "UNVERIFIABLE"),
])
def test_all_conclusive_consumers_agree(checked, ok, infrastructure_failed, verdict):
    result = VerificationResult("task", ok, checked, infrastructure_failed=infrastructure_failed)
    assert result.verdict == verdict
    assert result.conclusively_ok is (verdict == "PASSED")
    assert result.to_dict()["verdict"] == verdict
    assert result.ok is ok


def test_combined_success_and_engine_failure_remain_inconclusive():
    passed = VerificationResult("code", True, True, score=1, engine="syntax")
    unavailable = VerificationResult("code", True, False, engine="execution",
                                     infrastructure_failed=True, issues=["sandbox unavailable"])
    result = combine_results("code", [passed, unavailable])
    assert result.checked
    assert result.ok
    assert result.verdict == "UNVERIFIABLE"
    assert not result.conclusively_ok
    assert result.detail["sub"][0]["verdict"] == "PASSED"
    assert "sandbox unavailable" in result.issues


def test_unnecessary_engine_does_not_cancel_a_real_pass():
    passed = VerificationResult("code", True, True, engine="execution")
    unnecessary = VerificationResult("code", True, False, engine="no equations")
    assert combine_results("code", [passed, unnecessary]).conclusively_ok
