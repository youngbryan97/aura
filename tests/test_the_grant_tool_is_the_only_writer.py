"""The grant is written by a person at a terminal, and by nothing else.

The design turns on one property: nothing on an autonomous path may write a
standing authorization. The clearest way to keep that true is for the only
caller of the writer to be a tool an operator runs, so that is what these
check — the tool exists, it refuses the shapes that would make a grant
something other than scoped and bounded, and no runtime module calls the
writer at all.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TOOL = _ROOT / "tools" / "grant_cortex_activation.py"
_PY = sys.executable


def _run(*args: str):
    return subprocess.run(
        [_PY, str(_TOOL), *args], capture_output=True, text=True, cwd=_ROOT
    )


def test_the_tool_exists_and_explains_itself():
    said = _run("--help")
    assert said.returncode == 0
    assert "standing authorization" in said.stdout.lower()


def test_it_refuses_a_grant_that_names_no_operator():
    said = _run("--days", "1", "--model-prefix", "/models/")
    assert said.returncode != 0
    assert "granted-by" in (said.stderr + said.stdout)


def test_it_refuses_a_grant_with_no_expiry(tmp_path):
    said = _run(
        "--granted-by", "an operator",
        "--model-prefix", "/models/",
        "--fused-model-dir", str(tmp_path),
    )
    assert said.returncode != 0
    assert "--days" in (said.stderr + said.stdout)


def test_it_writes_and_reads_back_a_scoped_grant(tmp_path):
    written = _run(
        "--granted-by", "an operator",
        "--model-prefix", "/models/Qwen3.9-",
        "--days", "7",
        "--activations", "2",
        "--reason", "a planned window",
        "--fused-model-dir", str(tmp_path),
    )
    assert written.returncode == 0, written.stderr
    shown = _run("--show", "--fused-model-dir", str(tmp_path))
    assert shown.returncode == 0
    assert "an operator" in shown.stdout
    assert "0 of 2 activation(s) used" in shown.stdout


def test_showing_nothing_is_not_an_error(tmp_path):
    said = _run("--show", "--fused-model-dir", str(tmp_path))
    assert said.returncode == 0
    assert "no standing grant" in said.stdout


def test_the_written_grant_says_what_it_does_not_authorize(tmp_path):
    """An operator reading the confirmation should not have to infer the scope."""
    said = _run(
        "--granted-by", "an operator",
        "--model-prefix", "/models/Qwen3.9-",
        "--days", "7",
        "--fused-model-dir", str(tmp_path),
    )
    assert "does not replace the evidence" in said.stdout


@pytest.mark.parametrize("tree", ["core", "interface", "skills"])
def test_no_runtime_module_writes_a_grant(tree):
    found = subprocess.run(
        ["grep", "-rn", "write_standing_grant", "--include=*.py", str(_ROOT / tree)],
        capture_output=True, text=True,
    ).stdout.splitlines()
    callers = [one for one in found if "standing_authorization.py" not in one]
    assert not callers, f"{tree} can write its own grant: {callers}"
