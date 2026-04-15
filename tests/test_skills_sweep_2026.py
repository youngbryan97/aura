################################################################################

"""
tests/test_skills_sweep_2026.py
───────────────────────────────
Phase VIII — Skills Sweep hardening tests.
Validates that 2026 code-quality standards are met across /skills.
"""

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest

from core.skills.sovereign_network import NetworkInput, SovereignNetworkSkill

SKILLS_DIR = Path(__file__).resolve().parent.parent / "core" / "skills"


# ── Fix 1: manifest_to_device must not import sync requests ────────────
def test_manifest_no_sync_requests():
    """manifest_to_device.py should use httpx, not requests."""
    src = (SKILLS_DIR / "manifest_to_device.py").read_text()
    assert "import requests" not in src, "sync requests still imported"
    assert "httpx" in src, "httpx not used"


# ── Fix 2: sovereign_network must not call sync subprocess in async ────
def test_sovereign_network_no_sync_subprocess():
    """sovereign_network.py should not call subprocess.run / check_output synchronously."""
    src = (SKILLS_DIR / "sovereign_network.py").read_text()
    # Must not have a bare subprocess.run(...) call — only asyncio.to_thread(subprocess.run, ...)
    lines = src.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments and string literals
        if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
            continue
        # Detect bare subprocess.run( or subprocess.check_output( NOT preceded by to_thread
        if re.search(r"(?<!to_thread\()subprocess\.(run|check_output)\(", stripped):
            if "to_thread" not in line and "import" not in line:
                pytest.fail(f"L{i}: sync subprocess call: {stripped}")


# ── Fix 3: system_proprioception must not use f-string loggers ─────────
def test_proprioception_no_fstring_loggers():
    """system_proprioception.py must use %s formatting for logger calls."""
    src = (SKILLS_DIR / "system_proprioception.py").read_text()
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r'self\.logger\.\w+\(f"', line) or re.search(r"self\.logger\.\w+\(f'", line):
            pytest.fail(f"L{i}: f-string logger found: {line.strip()}")


# ── Fix 4: test_skills.py must be parseable ────────────────────────────
def test_test_skills_parseable():
    """tests/test_skills.py must not have syntax errors."""
    test_file = Path(__file__).resolve().parent / "test_skills.py"
    src = test_file.read_text()
    try:
        ast.parse(src, filename=str(test_file))
    except SyntaxError as e:
        pytest.fail(f"test_skills.py has syntax error: {e}")


# ── Fix 5: no bare except clauses in sovereign_network ─────────────────
def test_sovereign_network_no_bare_except():
    """sovereign_network.py must not have bare `except:` clauses."""
    src = (SKILLS_DIR / "sovereign_network.py").read_text()
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped == "except:" or re.match(r"except:\s", stripped):
            pytest.fail(f"L{i}: bare except clause: {stripped}")


# ── Fix 1 (extra): manifest uses httpx.AsyncClient ────────────────────
def test_manifest_uses_async_httpx():
    """manifest_to_device.py should use httpx.AsyncClient for requests."""
    src = (SKILLS_DIR / "manifest_to_device.py").read_text()
    assert "AsyncClient" in src, "httpx.AsyncClient not found"


# ── Global: all skill execute() methods are async def ──────────────────
def test_all_skills_execute_is_async():
    """Every skill's execute() method must be async."""
    for py_file in sorted(SKILLS_DIR.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        src = py_file.read_text()
        try:
            tree = ast.parse(src, filename=str(py_file))
        except SyntaxError:
            continue  # Skip unparseable files
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "execute":
                        pytest.fail(
                            f"{py_file.name}:{item.lineno}: "
                            f"execute() is sync (def), must be async def"
                        )


@pytest.mark.asyncio
async def test_sovereign_network_discovery_falls_back_without_nmap(monkeypatch):
    """Peer discovery should remain useful when Homebrew nmap is unavailable."""
    skill = SovereignNetworkSkill()

    async def missing_nmap(*_args, **_kwargs):
        raise FileNotFoundError("nmap")

    class FakeWriter:
        def close(self):
            pass

        async def wait_closed(self):
            pass

    async def fake_open_connection(host, port):
        if host == "192.168.1.2" and port == 8000:
            return object(), FakeWriter()
        raise OSError("closed")

    monkeypatch.setattr("asyncio.create_subprocess_exec", missing_nmap)
    monkeypatch.setattr("asyncio.open_connection", fake_open_connection)

    result = await skill.execute(
        NetworkInput(mode="discovery", target="192.168.1.0/30", ports="8000"),
        {},
    )

    assert result["ok"] is True
    assert result["fallback"] == "tcp_connect"
    assert result["peers"] == [{"address": "192.168.1.2", "rpc_port": 8000, "source": "tcp_connect"}]


##
