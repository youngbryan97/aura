from core.tasks import managed_command, run_mutation_tests, run_rl_training


def test_pytest_target_normalizes_to_workspace_relative_command(monkeypatch):
    captured = {}

    def fake_run_project_command(command, *, timeout_s):
        captured["command"] = command
        captured["timeout_s"] = timeout_s
        return managed_command.ManagedCommandResult(command, 0, "ok", "", 0.01)

    monkeypatch.setattr(managed_command, "run_project_command", fake_run_project_command)

    result = managed_command.run_project_pytest(
        "tests/test_runtime_security_config.py::test_runtime_security_accepts_valid_bearer_token"
    )

    assert result.ok is True
    assert captured["command"][1:4] == ("-m", "pytest", "-q")
    assert captured["command"][4] == (
        "tests/test_runtime_security_config.py::test_runtime_security_accepts_valid_bearer_token"
    )


def test_pytest_target_rejects_path_escape_without_launching(monkeypatch):
    calls = []

    def fake_run_project_command(command, *, timeout_s):
        calls.append((command, timeout_s))
        return managed_command.ManagedCommandResult(command, 0, "unexpected", "", 0.01)

    monkeypatch.setattr(managed_command, "run_project_command", fake_run_project_command)

    result = managed_command.run_project_pytest("../outside.py")

    assert result.ok is False
    assert result.returncode == 127
    assert "inside the Aura workspace" in result.stderr
    assert calls == []


def test_run_mutation_tests_returns_structured_rejection_for_invalid_target():
    result = run_mutation_tests("../outside.py")

    assert result["success"] is False
    assert result["returncode"] == 127
    assert "inside the Aura workspace" in result["error"]


def test_run_rl_training_uses_managed_payload(monkeypatch):
    from core import tasks as tasks_module

    def fake_run_project_python(relative_script):
        assert relative_script == "core/rl_train.py"
        return managed_command.ManagedCommandResult((relative_script,), 0, "trained", "", 0.01)

    monkeypatch.setattr(tasks_module, "run_project_python", fake_run_project_python)

    assert run_rl_training() == {
        "status": "success",
        "stdout": "trained",
        "stderr": "",
        "returncode": 0,
    }
