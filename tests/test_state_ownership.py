"""A test run must not be able to change what Aura believes.

Two live cases:

* Terminal-grid tests read and wrote the live user-global learning
  directory. Learned risk from live state reached 1.0 and vetoed a benign
  move; test episodes were written back into the organism's real memory.
* A fixture left a fail-closed Mycelium registered after its test ended,
  and dozens of later tests failed only inside the full suite.

The guard is not a warning. The damage above was silent.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.runtime.state_ownership import (
    RuntimeProfile,
    StateOwnershipViolation,
    assert_state_path_allowed,
    is_live_state_path,
    live_state_root,
    runtime_identity,
    runtime_instance_id,
    runtime_profile,
    stamp_record,
    state_root,
)

ROOT = Path(__file__).resolve().parents[1]


def test_this_very_test_run_is_not_a_live_runtime():
    """If this fails, everything below is testing the wrong thing."""
    assert runtime_profile() is RuntimeProfile.TEST
    assert not runtime_profile().may_touch_live_state


def test_the_test_root_is_not_inside_the_live_root():
    """A subdirectory is one `..` from the thing it was meant to be apart from."""
    root, live = state_root(), live_state_root()
    assert root != live
    assert live not in root.parents


def test_writing_live_state_from_a_test_raises(tmp_path):
    with pytest.raises(StateOwnershipViolation, match="live instance state"):
        assert_state_path_allowed(live_state_root() / "data" / "aura_memory.db")
    # The run's own root is fine.
    assert_state_path_allowed(state_root() / "data" / "anything.json")
    assert_state_path_allowed(tmp_path / "scratch.json")


def test_the_violation_says_how_to_fix_it():
    """An enforcement message that does not name the remedy just blocks work."""
    with pytest.raises(StateOwnershipViolation) as caught:
        assert_state_path_allowed(live_state_root() / "data" / "x.json", source="probe")
    message = str(caught.value)
    assert "AURA_STATE_ROOT" in message
    assert "probe" in message


def test_traversal_back_into_live_state_is_caught():
    """Resolved, not string-compared, or the check is decorative."""
    sneaky = state_root() / ".." / live_state_root().name / "data" / "memory.db"
    assert is_live_state_path(sneaky)
    with pytest.raises(StateOwnershipViolation):
        assert_state_path_allowed(sneaky)


def test_a_path_that_does_not_exist_yet_is_still_checked():
    """Writes create things; checking only existing paths would miss every write."""
    assert is_live_state_path(live_state_root() / "not" / "created" / "yet.json")


def test_the_write_gateway_enforces_it():
    """The guard has to sit at the chokepoint, not in a helper nobody calls."""
    from core.runtime.file_write_gateway import get_file_write_gateway

    with pytest.raises(StateOwnershipViolation):
        get_file_write_gateway().write_text(
            live_state_root() / "data" / "test_should_never_write_this.json",
            "{}",
            source="test_state_ownership",
        )


def test_the_gateway_guard_is_wired_at_the_shared_coercion_point():
    """AST: every write path funnels through _coerce_target, so it must guard."""
    source = (ROOT / "core" / "runtime" / "file_write_gateway.py").read_text("utf-8")
    tree = ast.parse(source)
    for name in ("_coerce_target", "_coerce_path_allow_dir"):
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        calls = {
            node.func.id
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "assert_state_path_allowed" in calls, (
            f"{name} does not check state ownership; writes routed through it "
            "would reach live state unguarded"
        )


def test_identity_is_immutable_for_the_process():
    assert runtime_instance_id() == runtime_instance_id()
    assert runtime_instance_id().startswith("test-")


def test_state_ownership_is_a_standard_library_bootstrap_boundary():
    """State identity cannot depend on services whose paths it owns."""

    source = ROOT / "core" / "runtime" / "state_ownership.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    project_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("core"):
            project_imports.append(node.module)
        elif isinstance(node, ast.Import):
            project_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("core")
            )
    assert project_imports == []


def test_bootstrap_flags_remain_typed_and_enumerable():
    from core.runtime.flags import declared_flags, flag_report

    declared = declared_flags()
    assert declared["AURA_STATE_ROOT"].owner == "core.runtime.state_ownership"
    report = {entry["name"]: entry for entry in flag_report()}
    assert report["AURA_STATE_ROOT"]["kind"] == "string"
    assert report["AURA_STATE_ROOT"]["owner"] == "core.runtime.state_ownership"


def test_persistent_records_name_the_runtime_that_wrote_them():
    stamped = stamp_record({"episode": 1}, model_identity="qwen-32b@abc123")
    assert stamped["episode"] == 1
    assert stamped["_runtime"]["runtime_instance_id"] == runtime_instance_id()
    assert stamped["_runtime"]["runtime_profile"] == "test"
    assert stamped["_runtime"]["model_identity"] == "qwen-32b@abc123"


def test_stamping_does_not_mutate_the_caller_or_overwrite_an_origin():
    original = {"episode": 1}
    stamp_record(original)
    assert "_runtime" not in original, "stamp_record mutated its input"

    forwarded = {"episode": 2, "_runtime": {"runtime_instance_id": "live-original"}}
    assert stamp_record(forwarded)["_runtime"]["runtime_instance_id"] == "live-original"


def test_runtime_identity_is_serializable_and_carries_no_secrets():
    import json

    identity = runtime_identity()
    json.dumps(identity)
    assert set(identity) == {
        "runtime_instance_id",
        "runtime_profile",
        "state_root",
        "started_at",
        "pid",
        "host",
    }


def test_config_paths_resolve_under_this_runs_root_not_the_live_one():
    """The structural fix: stores get their own world, not a blocked write."""
    from core.config import config

    assert not is_live_state_path(config.paths.data_dir)
    assert str(config.paths.data_dir).startswith(str(state_root()))


@pytest.mark.parametrize(
    "module,attribute",
    [
        ("core.security.trust_engine", "TRUST_LOG_PATH"),
        ("core.memory.conversation_persistence", "DEFAULT_PERSIST_DIR"),
        ("core.reaper", "DEFAULT_REAPER_MANIFEST_DIR"),
        ("core.memory.scar_formation", "_DATA_DIR"),
        ("core.agi.skill_synthesizer", "PERSIST_PATH"),
    ],
)
def test_module_level_state_paths_follow_the_profile(module, attribute):
    """These were `Path.home() / ".aura"`, evaluated at import, bypassing config."""
    import importlib

    value = getattr(importlib.import_module(module), attribute)
    assert not is_live_state_path(value), (
        f"{module}.{attribute} still resolves into live instance state"
    )


def test_no_module_resolves_the_live_root_for_itself_any_more():
    """The whole class, pinned. 243 sites did this before.

    A store that finds its own root from Path.home() cannot be redirected,
    only blocked — and a blocked subsystem is not a working one.
    """
    offenders: list[str] = []
    allowed = {"core/runtime/state_ownership.py"}
    for path in (ROOT / "core").rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        if rel in allowed or "__pycache__" in rel:
            continue
        text = path.read_text("utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if 'Path.home() / ".aura"' in line or "expanduser(\"~/.aura\")" in line:
                offenders.append(f"{rel}:{line_no}")
    assert not offenders, (
        "these modules resolve the live state root themselves instead of "
        f"asking state_root(): {offenders[:10]}"
    )


# ------------------------------------------------------------ shared assets
#
# Separate state by what a runtime WRITES, not by everything it touches.
# Model weights are large, content-addressed, read-only and shared by every
# runtime on the host. Diverting them broke a real checkpoint load
# (manifest_missing) for no safety benefit.


def test_model_assets_resolve_to_the_real_root_even_under_test():
    from core.runtime.state_ownership import shared_asset_root

    assert shared_asset_root() == live_state_root()
    assert is_live_state_path(shared_asset_root() / "models")


def test_asset_root_is_still_overridable():
    """A harness with its own model cache must be able to say so."""
    import os
    from unittest import mock

    from core.runtime.state_ownership import shared_asset_root

    # resolve() on the expectation too: on macOS /tmp is a symlink to
    # /private/tmp, and comparing an unresolved literal would be testing
    # the platform rather than the override.
    with mock.patch.dict(os.environ, {"AURA_ASSET_ROOT": "/tmp/aura-assets"}):
        assert shared_asset_root() == Path("/tmp/aura-assets").resolve()


def test_assets_are_read_not_written_through_the_gateway():
    """The distinction only holds if nothing WRITES to the asset root.

    If a runtime mutates something there it is state, not an asset, and it
    belongs under state_root(). The write gateway still refuses.
    """
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.state_ownership import shared_asset_root

    with pytest.raises(StateOwnershipViolation):
        get_file_write_gateway().write_text(
            shared_asset_root() / "models" / "should_never_write.json",
            "{}",
            source="test_state_ownership",
        )


def test_a_spawned_child_is_not_a_test_runtime_just_because_it_inherited_env():
    """PYTEST_VERSION is inherited by subprocesses; running pytest is not.

    A delegated child launched with its own HOME derives its state root
    from that HOME — which is what its parent arranged. Env-based detection
    sent it to a `-test` sibling the parent never created and broke a
    working model-lane delegation.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as home:
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json;"
                "from core.runtime.state_ownership import runtime_profile, state_root;"
                "print(json.dumps({'profile': runtime_profile().value,"
                " 'root': str(state_root())}))",
            ],
            cwd=str(ROOT),
            env={**os.environ, "HOME": home},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        result = json.loads(child.stdout.strip())

    assert result["profile"] == "live", (
        "a plain subprocess inherited PYTEST_VERSION and called itself a test "
        "runtime; it is an ordinary program"
    )
    assert result["root"].endswith("/.aura"), result["root"]


def test_an_unresolvable_path_is_treated_as_live_state():
    """"I could not check" must never read as "it is safe".

    Found by tools/guard_mutation.py: this branch had no test, and it returned
    False — which callers read as "not live state" and allow. A path that
    cannot be resolved has not been shown to be outside the live root; it has
    only declined to answer. Fail closed.

    A null byte makes Path.resolve raise ValueError, which is the cheapest way
    to reach the branch without depending on filesystem state.
    """
    assert is_live_state_path("/tmp/\x00/definitely-not-resolvable")


def test_an_unresolvable_path_is_refused_by_the_guard(monkeypatch):
    """And the refusal actually reaches the enforcement point."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "probe")
    if runtime_profile().may_touch_live_state:
        pytest.skip("a LIVE runtime is permitted to touch live state")
    with pytest.raises(StateOwnershipViolation):
        assert_state_path_allowed("/tmp/\x00/unresolvable", source="probe")


def test_a_worktree_may_write_into_the_checkout_it_was_forked_from():
    """The subject-core battery runs from a worktree and writes to main.

    A campaign pins the commit under test in a linked worktree so the source
    cannot move under three runs, and writes every artifact into the one
    campaign directory in the main checkout. That directory sits under
    ~/.aura/live-source, inside the live root, and the guard knew only about
    the checkout the running module belongs to — so the write read as an
    attempt on live instance state and the run died on its first action.

    A checkout is not instance state whichever checkout it is.
    """
    from core.runtime.state_ownership import _source_roots

    roots = _source_roots()
    assert ROOT in roots, roots
    for root in roots:
        assert not is_live_state_path(root / "artifacts" / "subject_core" / "run_001")


def test_the_sibling_checkout_is_read_off_disk_not_guessed(tmp_path, monkeypatch):
    """A linked worktree's .git names its git directory; its grandparent is main."""
    from core.runtime import state_ownership

    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "leaf").mkdir(parents=True)
    leaf = tmp_path / "leaf"
    leaf.mkdir()
    (leaf / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'leaf'}\n")
    monkeypatch.setattr(
        state_ownership, "__file__", str(leaf / "core" / "runtime" / "state_ownership.py")
    )
    state_ownership._source_roots.cache_clear()
    try:
        assert state_ownership._source_roots() == (leaf, main)
    finally:
        state_ownership._source_roots.cache_clear()


def test_the_live_instance_state_is_still_refused_from_every_checkout():
    """Widening the source set must not widen what a test run may write."""
    live = live_state_root()
    for tail in ("data/memory.db", "run/aura.pid", "logs/desktop-launch.log"):
        assert is_live_state_path(live / tail), tail
