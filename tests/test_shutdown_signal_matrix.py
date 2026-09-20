from __future__ import annotations

from typing import Any

from tools.shutdown_signal_matrix import (
    CASE_SPECS,
    COORDINATOR_PHASE_CASES,
    REQUIRED_OWNER_CLASSES,
    SHUTDOWN_PHASES,
    _is_aura_main_process,
    _is_competing_model_owner,
    _parse_cases,
    aggregate_owner_class_witnesses,
    evaluate_terminal_report,
)


def _clean_terminal_report() -> dict[str, Any]:
    return {
        "schema": "aura.shutdown_verdict.v1",
        "pid": 4242,
        "stage": "root_process_exit",
        "final": True,
        "terminal_receipt_sequence": 1,
        "request": {
            "first_reason": "desktop_signal:SIGTERM",
            "request_count": 4,
        },
        "admission": {"counts": {"survived": 0}},
        "components": {
            "coordinator": {
                "clean": True,
                "completed_phases": list(SHUTDOWN_PHASES),
            },
            "container": {"clean": True},
            "runtime_hygiene": {"clean": True},
            "final_tasks": {"count": 0},
        },
        "root_exit": {
            "lock_released": True,
            "multiprocessing_finalizers_completed": True,
            "logging_shutdown_completed": True,
            "resources": {"clean": True, "blockers": []},
            "exit_code": 0,
        },
        "verdict": {"clean": True, "blockers": []},
    }


def test_terminal_report_requires_every_shutdown_and_root_exit_invariant() -> None:
    checks = evaluate_terminal_report(
        _clean_terminal_report(),
        root_pid=4242,
        expected_first_reason="desktop_signal:SIGTERM",
        minimum_signal_requests=2,
    )

    assert checks
    assert all(checks.values())


def test_terminal_report_fails_on_resurrection_and_out_of_order_phases() -> None:
    report = _clean_terminal_report()
    report["admission"]["counts"]["survived"] = 1
    report["components"]["coordinator"]["completed_phases"] = list(
        reversed(SHUTDOWN_PHASES)
    )

    checks = evaluate_terminal_report(
        report,
        root_pid=4242,
        expected_first_reason="desktop_signal:SIGTERM",
        minimum_signal_requests=2,
    )

    assert checks["no_shutdown_resurrection"] is False
    assert checks["all_phases_once"] is False


def test_aura_process_classifier_matches_argv_not_shell_text() -> None:
    assert _is_aura_main_process(
        ["/opt/homebrew/bin/python3.12", "/repo/aura_main.py", "--desktop"]
    )
    assert not _is_aura_main_process(
        ["/bin/zsh", "-c", "ps aux | grep aura_main.py"]
    )
    assert not _is_aura_main_process(
        ["/opt/homebrew/bin/python3.12", "/repo/tools/shutdown_signal_matrix.py"]
    )
    assert _is_competing_model_owner(
        ["python", "/repo/tools/evaluate_unified_intrinsic_decoding.py"]
    )
    assert not _is_competing_model_owner(
        ["python", "/repo/tools/closeout/audit_shutdown_contract.py"]
    )


def test_case_selection_is_deduplicated_and_unknown_names_fail() -> None:
    assert _parse_cases(["ready_repeated,launcher_bootstrap", "ready_repeated"]) == [
        "ready_repeated",
        "launcher_bootstrap",
    ]
    assert _parse_cases(["all"]) == list(CASE_SPECS)

    try:
        _parse_cases(["not-a-case"])
    except ValueError as exc:
        assert "unknown case" in str(exc)
    else:
        raise AssertionError("unknown shutdown matrix case was accepted")


def test_matrix_covers_boot_ready_active_and_late_finalization_boundaries() -> None:
    expected_boundary_cases = {
        "launcher_bootstrap",
        "orchestrator_boot_repeated",
        "ready_repeated",
        "model_warmup_signal",
        "model_recovery_signal",
        "container_repeated",
        "root_finalization_repeated",
        "active_foreground_repeated",
    }
    expected_phase_cases = {f"{phase}_repeated" for phase in SHUTDOWN_PHASES}

    assert expected_boundary_cases | expected_phase_cases == set(CASE_SPECS)
    assert tuple(COORDINATOR_PHASE_CASES) == tuple(
        f"{phase}_repeated" for phase in SHUTDOWN_PHASES
    )
    for phase in SHUTDOWN_PHASES:
        assert CASE_SPECS[f"{phase}_repeated"].probe_target == f"coordinator:{phase}"
    assert CASE_SPECS["container_repeated"].probe_target == "container"
    assert CASE_SPECS["root_finalization_repeated"].probe_target == (
        "root_finalization"
    )
    assert CASE_SPECS["model_warmup_signal"].boot_mode == "headless"
    assert CASE_SPECS["model_recovery_signal"].kill_model_worker_after_trigger is True


def test_owner_class_coverage_requires_observed_and_clean_terminal_owners() -> None:
    report = _clean_terminal_report()
    report["components"]["runtime_hygiene"] = {
        "before": {
            "processes": {"active_registered": 1},
            "threads": {"active": 2},
            "tasks": {"total_observed": 4},
        },
        "after": {
            "processes": {
                "active_registered": 0,
                "active_subprocesses": 0,
                "active_multiprocessing": 0,
                "owned_descendant_processes": 0,
                "rogue_child_processes": 0,
            },
            "threads": {
                "active": 0,
                "active_non_daemon": 0,
                "stale_non_daemon": 0,
            },
            "tasks": {"active": 0, "shutdown_critical_active": 0},
            "native_resources": {"listening_socket_count": 0},
        },
    }
    report["components"]["coordinator"]["handler_statuses"] = {
        "task_supervisor:memory_sentinel_supervisor": "completed"
    }
    report["components"]["container"] = {
        "clean": True,
        "completed_services": ["actor_bus"],
    }
    verdict = {
        "checks": {"port_free": True, "singleton_lock_available": True},
        "pre_signal_evidence": {
            "port_listening": True,
            "singleton_lock_held": True,
            "model_worker_observed": True,
        },
        "shutdown_report": report,
    }

    witnesses = aggregate_owner_class_witnesses([verdict])

    assert tuple(witnesses) == REQUIRED_OWNER_CLASSES
    assert all(witnesses.values())

    verdict["pre_signal_evidence"]["model_worker_observed"] = False
    witnesses = aggregate_owner_class_witnesses([verdict])
    assert witnesses["model_worker"] is False


class TestEveryTriggerMarkerCanStillBePrinted:
    """A case waits for a log line; a line the runtime cannot write is a case
    that cannot run.

    `model_warmup_signal` waited ten minutes for "Primary 32B cortex is cold.
    Starting warmup" and then reported all twenty-four of its checks failed.
    The runtime writes that line with the resident lane's signed label, which
    stopped being a 32B when the cortex became Aura-Qwen3.8-27B, so the case
    had been dead since the model changed and said so only as a timeout.
    """

    def test_no_marker_names_a_model(self) -> None:
        import tools.shutdown_signal_matrix as matrix

        named = [
            f"{name} = {value!r}"
            for name, value in vars(matrix).items()
            if name.endswith("_MARKER")
            and isinstance(value, str)
            and any(
                tag in value
                for tag in ("32B", "27B", "8B", "1.5B", "Qwen", "Bonsai", "Ternary")
            )
        ]
        assert named == [], (
            "a marker naming a checkpoint dies the day the checkpoint does:\n"
            + "\n".join(named)
        )

    def test_the_cortex_markers_match_what_the_gate_writes(self) -> None:
        """Against the format strings, not the file's text.

        The recovery line is written as two adjacent literals and reads as
        one only after the parser joins them, so a substring search over the
        source says it is not there. The subject is what the runtime EMITS.
        """
        import ast
        import inspect
        from pathlib import Path

        import core.brain.inference_gate as gate
        import tools.shutdown_signal_matrix as matrix

        tree = ast.parse(Path(inspect.getsourcefile(gate)).read_text(encoding="utf-8"))
        written = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for marker in (
            matrix.MODEL_WARMUP_START_MARKER,
            matrix.MODEL_WARMUP_COMPLETE_MARKER,
            matrix.MODEL_RECOVERY_START_MARKER,
        ):
            assert any(marker in text for text in written), (
                f"nothing the gate can print contains {marker!r}"
            )
