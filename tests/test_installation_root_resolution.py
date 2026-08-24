"""The model store belongs to the installation, not to one source checkout.

Both `get_models_dir` and `get_fused_model_root` already documented that. The
base directory did not honour it: it resolved to the directory holding the
registry module, so a linked worktree looked for models and for the active
cortex manifest underneath itself, found neither, and reported the pointer
invalid. Smoke in a worktree failed for that reason and not for the reason it
appeared to.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm.model_registry import resolve_installation_root


def _primary(tmp_path: Path) -> Path:
    checkout = tmp_path / "live-source"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "models").mkdir()
    return checkout


def _worktree(tmp_path: Path, primary: Path, name: str = "feature") -> Path:
    tree = primary / ".claude" / "worktrees" / name
    tree.mkdir(parents=True)
    gitdir = primary / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    (tree / ".git").write_text(f"gitdir: {gitdir}\n")
    return tree


def test_a_canonical_checkout_resolves_to_itself():
    # Its .git is a directory, so there is nothing to follow.
    def _run(tmp):
        primary = _primary(tmp)
        assert resolve_installation_root(primary) == primary

    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        _run(Path(raw))


def test_a_linked_worktree_resolves_to_its_primary(tmp_path):
    primary = _primary(tmp_path)
    tree = _worktree(tmp_path, primary)
    assert resolve_installation_root(tree) == primary
    assert resolve_installation_root(tree) != tree


def test_a_relative_gitdir_pointer_is_resolved_against_the_worktree(tmp_path):
    primary = _primary(tmp_path)
    tree = primary / ".claude" / "worktrees" / "rel"
    tree.mkdir(parents=True)
    gitdir = primary / ".git" / "worktrees" / "rel"
    gitdir.mkdir(parents=True)
    (tree / ".git").write_text("gitdir: ../../../.git/worktrees/rel\n")
    assert resolve_installation_root(tree) == primary


def test_a_pointer_to_a_missing_primary_falls_back_to_the_checkout(tmp_path):
    # A wrong path that exists is worse than the local one, and this runs at
    # import time where raising would take the process down.
    tree = tmp_path / "orphan"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /nonexistent/.git/worktrees/gone\n")
    assert resolve_installation_root(tree) == tree


def test_a_malformed_marker_falls_back_to_the_checkout(tmp_path):
    for content in ("", "not a gitdir line\n", "gitdir:\n", "gitdir: \n"):
        tree = tmp_path / f"tree{abs(hash(content))}"
        tree.mkdir()
        (tree / ".git").write_text(content)
        assert resolve_installation_root(tree) == tree


def test_a_marker_with_no_git_component_falls_back(tmp_path):
    tree = tmp_path / "weird"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {tmp_path / 'somewhere' / 'else'}\n")
    assert resolve_installation_root(tree) == tree


def test_a_checkout_with_no_git_marker_at_all_resolves_to_itself(tmp_path):
    plain = tmp_path / "exported"
    plain.mkdir()
    assert resolve_installation_root(plain) == plain


def test_an_unreadable_marker_falls_back(tmp_path, monkeypatch):
    tree = tmp_path / "locked"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /somewhere/.git/worktrees/x\n")

    def _raise(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", _raise)
    assert resolve_installation_root(tree) == tree


def test_an_explicit_env_override_wins_over_both(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere"
    override.mkdir()
    monkeypatch.setenv("AURA_ROOT", str(override))
    import importlib

    from core.brain.llm import model_registry

    reloaded = importlib.reload(model_registry)
    try:
        assert reloaded.BASE_DIR == override
        assert reloaded.get_models_dir() == override / "models"
        assert (
            reloaded.get_fused_model_root() == override / "training" / "fused-model"
        )
    finally:
        monkeypatch.delenv("AURA_ROOT", raising=False)
        importlib.reload(model_registry)


def test_this_worktree_sees_the_installations_active_manifest():
    """The regression itself, asserted against the real checkout."""
    from core.brain.llm.model_registry import get_fused_model_root, get_models_dir

    root = get_fused_model_root()
    if not root.parent.parent.exists():
        pytest.skip("installation root is not present in this environment")
    # Running from a worktree, these must NOT point inside the worktree.
    here = Path(__file__).resolve().parents[1]
    if (here / ".git").is_file():
        assert not str(root).startswith(str(here))
        assert not str(get_models_dir()).startswith(str(here))


# ── The lane audit cache guards work its own key was doing ──────────────


def test_the_audit_cache_key_costs_no_filesystem_walk(monkeypatch):
    """A cache whose key does the work it guards is not a cache.

    The key resolved each lane's runtime PATH, which is a realpath per lane on
    every call including a hit — the same blocking disk work the docstring says
    the cache exists to avoid during health probes. It read as free only while
    the model directories were unreachable, which a worktree made permanent.
    """
    from core.brain.llm import model_registry

    monkeypatch.setenv("AURA_LANE_AUDIT_CACHE_TTL_S", "60")
    model_registry._LANE_AUDIT_CACHE.update(key=None, at=0.0, result=None)
    calls = {"n": 0}
    real = model_registry.os.path.realpath

    def _counting(path, *args, **kwargs):
        calls["n"] += 1
        return real(path, *args, **kwargs)

    monkeypatch.setattr(model_registry.os.path, "realpath", _counting)
    try:
        first = model_registry.audit_lane_assignments(force_refresh=True)
        after_first = calls["n"]
        second = model_registry.audit_lane_assignments()
        assert second == first
        assert calls["n"] == after_first
    finally:
        model_registry._LANE_AUDIT_CACHE.update(key=None, at=0.0, result=None)


def test_a_models_directory_configured_after_import_still_wins(monkeypatch, tmp_path):
    """MODEL_PATHS is baked at import; an override after it must still apply.

    Patching BASE_DIR alone never moved a lookup, and nothing noticed because
    in a worktree the baked paths did not exist either way.
    """
    from core.brain.llm import model_registry

    shared = tmp_path / "primary" / "models"
    (shared / "Qwen2.5-32B-Instruct-8bit").mkdir(parents=True)
    monkeypatch.setenv("AURA_MODELS_DIR", str(shared))
    monkeypatch.setattr(model_registry, "_cortex_path_cache", None)
    resolved = model_registry.get_model_path("Qwen2.5-32B-Instruct-8bit")
    assert resolved == str((shared / "Qwen2.5-32B-Instruct-8bit").resolve())


def test_an_explicitly_configured_lane_path_is_not_rebased(monkeypatch, tmp_path):
    """Only entries baked under the import-time models dir may be rebased.

    A lane pointed somewhere by configuration was put there deliberately.
    """
    from core.brain.llm import model_registry

    elsewhere = tmp_path / "custom" / "brainstem"
    elsewhere.mkdir(parents=True)
    monkeypatch.setitem(model_registry.MODEL_PATHS, "Qwen3.5-9B-4bit", elsewhere)
    monkeypatch.setenv("AURA_MODELS_DIR", str(tmp_path / "shared"))
    assert model_registry._configured_model_location("Qwen3.5-9B-4bit") == elsewhere
