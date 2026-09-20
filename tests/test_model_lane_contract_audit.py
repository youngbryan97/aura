from __future__ import annotations

from pathlib import Path

from tools.closeout.audit_model_lane_contract import ROOT, audit


def _passing_ownership(**_kwargs):
    return {
        "passed": True,
        "inventory_entries": 1,
        "owned_paths": 1,
        "load_references": 1,
        "findings": [],
    }


#: The files the contract is read from. `mlx_client.py` is only half of the
#: subject now: the worker lifecycle lives in a mixin beside it, so a fixture
#: that copies the parent alone gives the audit a class with no
#: `_ensure_worker_alive` to read and the mutation lands nowhere.
SUBJECT_FILES = (
    "core/runtime/control_plane.py",
    "core/runtime/model_lane_control.py",
    "core/runtime/subprocess_gateway.py",
    "core/brain/llm/mlx_client.py",
    "tools/live_resource_pressure_proof.py",
)


def _copy_subject(tmp_path: Path, *, replace: tuple[str, str] | None = None) -> int:
    """Copy the subject into `tmp_path`, optionally making one substitution.

    The substitution is applied wherever the text actually is rather than to
    a named file, because the lift moved several of these call sites into
    sibling modules and a fixture that names the old file silently mutates
    nothing. Returns how many files it changed, so a caller can assert the
    mutation landed.
    """
    relatives = list(SUBJECT_FILES)
    for name in SUBJECT_FILES:
        home = ROOT / name
        relatives += [
            str(sibling.relative_to(ROOT))
            for sibling in sorted(home.parent.glob(f"{home.stem}_*.py"))
        ]
    changed = 0
    for relative in dict.fromkeys(relatives):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (ROOT / relative).read_text(encoding="utf-8")
        if replace is not None and replace[0] in source:
            source = source.replace(replace[0], replace[1], 1)
            changed += 1
        target.write_text(source, encoding="utf-8")
    return changed


def test_repository_model_lane_contract_is_complete() -> None:
    report = audit(ROOT, ownership_runner=_passing_ownership)

    assert report["passed"] is True
    assert report["issues"] == []
    assert report["checked_functions"] >= 21
    assert report["unified_admission_contract"] == {
        "policy_owner": "ResourceAdmissionController",
        "durable_capacity_owner": "ModelLaneController",
        "production_transaction": "_model_load_admission_context",
        "anti_thrash_owner": "MLXLocalClient._ensure_worker_alive",
    }


def test_model_lane_contract_fails_closed_when_required_call_is_removed(
    tmp_path: Path,
) -> None:
    assert _copy_subject(
        tmp_path,
        replace=(
            "await self.reconcile_expired_compensations()",
            "await self._persist_missing_terminal_receipts()",
        ),
    ) == 1

    report = audit(tmp_path, ownership_runner=_passing_ownership)

    assert report["passed"] is False
    assert any(
        "ModelLaneController.reserve lost calls ['reconcile_expired_compensations']"
        in issue
        for issue in report["issues"]
    )


def test_model_lane_contract_rejects_scheduler_capacity_split(tmp_path: Path) -> None:
    assert _copy_subject(
        tmp_path,
        replace=(
            "lane_decision = await lane_controller.reserve(",
            "lane_decision = await lane_controller.cancel(",
        ),
    ) == 1

    report = audit(tmp_path, ownership_runner=_passing_ownership)

    assert report["passed"] is False
    assert any(
        "_model_load_admission_context lost calls ['reserve']" in issue
        for issue in report["issues"]
    )


def test_model_lane_contract_rejects_removed_retry_storm_backoff(tmp_path: Path) -> None:
    assert _copy_subject(
        tmp_path,
        replace=(
            "if request_is_background and self._model_load_admission_backoff_active():",
            "if request_is_background and False:",
        ),
    ) == 1

    report = audit(tmp_path, ownership_runner=_passing_ownership)

    assert report["passed"] is False
    assert any(
        "MLXLocalClient._ensure_worker_alive lost calls "
        "['_model_load_admission_backoff_active']" in issue
        for issue in report["issues"]
    )


def test_model_lane_contract_propagates_ownership_failure() -> None:
    report = audit(
        ROOT,
        ownership_runner=lambda **_kwargs: {
            "passed": False,
            "findings": [{"code": "unowned_model_load"}],
        },
    )

    assert report["passed"] is False
    assert "model load ownership audit did not pass" in report["issues"]
