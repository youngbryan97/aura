import sys

import pytest

from core.sovereign.local_sandbox import LocalSandbox


@pytest.mark.asyncio
async def test_local_sandbox_rejects_shell_control_operators(tmp_path):
    sandbox = LocalSandbox(str(tmp_path))
    sandbox.start()
    try:
        result = await sandbox.run_command(
            f"{sys.executable} -c \"print('safe')\" ; touch escaped.txt"
        )
    finally:
        sandbox.stop()

    assert result.exit_code == 126
    assert "Shell control operators" in result.stderr
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_local_sandbox_runs_argv_without_shell_expansion(tmp_path):
    sandbox = LocalSandbox(str(tmp_path))
    sandbox.start()
    try:
        result = await sandbox.run_command(
            f"{sys.executable} -c \"import sys; print(sys.argv[1])\" 'hello; touch escaped.txt'"
        )
    finally:
        sandbox.stop()

    assert result.exit_code == 0
    assert "hello; touch escaped.txt" in result.stdout
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.asyncio
async def test_local_sandbox_strips_sensitive_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_SECRET_TOKEN", "super-secret")
    sandbox = LocalSandbox(str(tmp_path))
    sandbox.start()
    try:
        result = await sandbox.run_command(
            f"{sys.executable} -c \"import os; print(os.environ.get('AURA_SECRET_TOKEN', 'missing'))\""
        )
    finally:
        sandbox.stop()

    assert result.exit_code == 0
    assert result.stdout.strip() == "missing"
