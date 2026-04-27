from pathlib import Path

from core.runtime.flagship_readiness import scan_codebase


def test_flagship_gate_flags_raw_asyncio_task_in_production(tmp_path: Path):
    prod = tmp_path / "core" / "bad.py"
    prod.parent.mkdir(parents=True)
    prod.write_text("import asyncio\nasync def f():\n    asyncio.create_task(asyncio.sleep(0))\n", encoding="utf-8")
    report = scan_codebase(tmp_path)
    assert any(issue.code == "RAW_ASYNCIO_TASK" for issue in report.issues)


def test_flagship_gate_allows_tests_to_create_tasks(tmp_path: Path):
    test = tmp_path / "tests" / "test_example.py"
    test.parent.mkdir(parents=True)
    test.write_text("import asyncio\nasync def test_f():\n    asyncio.create_task(asyncio.sleep(0))\n", encoding="utf-8")
    report = scan_codebase(tmp_path)
    assert not any(issue.code == "RAW_ASYNCIO_TASK" for issue in report.issues)


def test_flagship_gate_flags_async_sys_exit(tmp_path: Path):
    prod = tmp_path / "core" / "shutdown.py"
    prod.parent.mkdir(parents=True)
    prod.write_text("import sys\nasync def stop():\n    sys.exit(0)\n", encoding="utf-8")
    report = scan_codebase(tmp_path)
    assert any(issue.code == "ASYNC_SYS_EXIT" for issue in report.issues)


def test_flagship_gate_flags_import_time_async_primitives(tmp_path: Path):
    prod = tmp_path / "core" / "singleton.py"
    prod.parent.mkdir(parents=True)
    prod.write_text("import asyncio\nLOCK = asyncio.Lock()\n", encoding="utf-8")
    report = scan_codebase(tmp_path)
    assert any(issue.code == "IMPORT_TIME_ASYNC_PRIMITIVE" for issue in report.issues)


def test_task_ownership_imports():
    from core.runtime.task_ownership import close_awaitable, fire_and_forget
    assert callable(close_awaitable)
    assert callable(fire_and_forget)
