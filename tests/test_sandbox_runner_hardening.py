from core.sandbox.runner import DEFAULT_CODE_BYTES, DEFAULT_MEM_BYTES, run_untrusted


def test_sandbox_default_memory_budget_is_bounded_but_realistic():
    assert 256 * 1024 * 1024 <= DEFAULT_MEM_BYTES <= 1024 * 1024 * 1024


def test_run_untrusted_returns_structured_stdout():
    result = run_untrusted("print('hello')", timeout=2)

    assert result["status"] == "ok"
    assert result["stdout"] == "hello\n"
    assert result["stderr"] == ""
    assert result["returncode"] == 0


def test_run_untrusted_blocks_import_escape():
    result = run_untrusted("__import__('os')", timeout=2)

    assert result["status"] == "error"
    # The refusal is an ImportError now, and it names what IS available —
    # the sandbox has the pure-computation standard library. What matters
    # here is that os is not in it.
    assert "cannot be imported here" in result["repr"]
    assert "os" in result["repr"]
    assert "cannot be imported here" in result["traceback"]


def test_run_untrusted_rejects_oversized_code():
    result = run_untrusted("x" * (DEFAULT_CODE_BYTES + 1), timeout=2)

    assert result["status"] == "rejected"
    assert "exceeds" in result["stderr"]
    assert result["returncode"] is None


def test_run_untrusted_reports_cpu_limit_as_timeout():
    result = run_untrusted("i = 0\nwhile 1:\n    i += 1\n", timeout=1)

    # "cpu_exhausted" rather than "timeout": the two are different faults
    # and the caller is usually the model that wrote the code, being asked
    # to try again. A wall-clock timeout means the machine was busy; this
    # means the code was.
    assert result["status"] in ("cpu_exhausted", "timeout")
    # It says where the budget went, which is the thing the caller needs:
    # "cpu time limit exceeded" and nothing else left the model that wrote
    # the code with no idea it had written a loop.
    assert "computation budget" in result["stderr"]
    assert "line" in result["stderr"], "it must say where it was"
