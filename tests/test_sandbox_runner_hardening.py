from core.sandbox.runner import DEFAULT_CODE_BYTES, run_untrusted


def test_run_untrusted_returns_structured_stdout():
    result = run_untrusted("print('hello')", timeout=2)

    assert result["status"] == "ok"
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == ""
    assert result["returncode"] == 0


def test_run_untrusted_blocks_import_escape():
    result = run_untrusted("__import__('os')", timeout=2)

    assert result["status"] == "error"
    assert "NameError" in result["repr"]
    assert "__import__" in result["traceback"]


def test_run_untrusted_rejects_oversized_code():
    result = run_untrusted("x" * (DEFAULT_CODE_BYTES + 1), timeout=2)

    assert result["status"] == "rejected"
    assert "exceeds" in result["stderr"]
    assert result["returncode"] is None


def test_run_untrusted_reports_cpu_limit_as_timeout():
    result = run_untrusted("i = 0\nwhile 1:\n    i += 1\n", timeout=1)

    assert result["status"] == "timeout"
    assert "limit" in result["stderr"]
