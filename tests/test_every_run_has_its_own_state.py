"""Two runs with one fingerprint were two different experiments.

No tool that builds the subject-core organism named a state root, so under the
test profile every campaign and every test shared `~/.aura-test`. The world
model a run started from had 170,936 steps and 87,613 training updates behind
it, trained by whichever runs came first, and its latent was already
saturated. These pin the helper that gives each run its own root.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from core.subject import isolation

pytestmark = pytest.mark.unit


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.delenv("AURA_STATE_ROOT", raising=False)
    monkeypatch.setattr(isolation, "_DEFAULT_ROOT", None)
    return monkeypatch


def test_a_run_gets_a_root_inside_its_own_directory(fresh, tmp_path: Path) -> None:
    from core.runtime.state_ownership import state_root

    run = tmp_path / "run_001"
    root = isolation.isolate_state(run)
    assert root == run / isolation.STATE_DIR
    assert os.environ["AURA_STATE_ROOT"] == str(root)
    assert os.path.realpath(str(state_root())) == os.path.realpath(str(root))


def test_a_root_that_does_not_move_refuses_the_run(fresh, tmp_path: Path) -> None:
    import core.runtime.state_ownership as ownership

    shared = tmp_path / "shared"
    fresh.setattr(ownership, "state_root", lambda: shared)
    with pytest.raises(isolation.StateIsolationError, match="not the experiment"):
        isolation.isolate_state(tmp_path / "run_001")


def test_a_module_still_holding_a_path_into_the_shared_root_is_reported(fresh, tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    run = tmp_path / "run_001"
    fresh.setattr(isolation, "_DEFAULT_ROOT", os.path.realpath(str(shared)))
    fresh.setenv("AURA_STATE_ROOT", str(run / "state"))
    stale = types.ModuleType("core._isolation_probe_stale")
    stale.PERSIST_PATH = shared / "data" / "world_model" / "vrnn_state.npz"
    moved = types.ModuleType("core._isolation_probe_moved")
    moved.PERSIST_PATH = run / "state" / "data" / "world_model" / "vrnn_state.npz"
    fresh.setitem(sys.modules, stale.__name__, stale)
    fresh.setitem(sys.modules, moved.__name__, moved)
    leaks = isolation.state_leaks()
    assert any(entry.startswith("core._isolation_probe_stale.PERSIST_PATH") for entry in leaks)
    assert not any(entry.startswith("core._isolation_probe_moved") for entry in leaks)


def test_a_sibling_directory_with_a_shared_prefix_is_not_the_shared_root(fresh, tmp_path: Path) -> None:
    """`.aura-test` begins with `.aura`, and a prefix is not a parent."""
    shared = tmp_path / ".aura"
    fresh.setattr(isolation, "_DEFAULT_ROOT", os.path.realpath(str(shared)))
    fresh.setenv("AURA_STATE_ROOT", str(tmp_path / "run_001" / "state"))
    sibling = types.ModuleType("core._isolation_probe_sibling")
    sibling.PERSIST_PATH = tmp_path / ".aura-test" / "data" / "x.json"
    fresh.setitem(sys.modules, sibling.__name__, sibling)
    assert not any("_isolation_probe_sibling" in entry for entry in isolation.state_leaks())


def test_nothing_is_reported_before_a_run_was_isolated(fresh) -> None:
    assert isolation.state_leaks() == []


REPO = Path(__file__).resolve().parents[1]

#: Every tool that builds the organism. A new one is caught by the test after.
TOOLS: tuple[str, ...] = (
    "tools/run_subject_core.py",
    "tools/run_subject_core_v25.py",
    "tools/run_content_geometry.py",
    "tools/run_ontogeny_lives.py",
    "tools/run_dose_response.py",
    "tools/subject_core_probe.py",
    "tools/audit_read_only_service_asks.py",
)


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_that_builds_the_organism_isolates_its_state_first(tool: str) -> None:
    text = (REPO / tool).read_text(encoding="utf-8")
    assert "isolate_state(" in text, f"{tool} builds the organism in a shared state root"
    assert text.index("isolate_state(") < text.index("build_runtime("), (
        f"{tool} isolates its state after it has already built the organism"
    )


def test_no_tool_builds_the_organism_without_isolating_it() -> None:
    """The list above is only as good as its last edit, so the tree is read too."""
    offenders = []
    for path in sorted((REPO / "tools").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "build_runtime(" in text and "def build_runtime" not in text and "isolate_state(" not in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"these build the organism in a shared state root: {offenders}"
