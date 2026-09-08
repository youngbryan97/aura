"""The foreground guard on the architecture index, against the real methods.

tests/conftest.py replaces ``ArchitectureIndex.build`` and
``schedule_background_build`` with no-ops for every test, because a real build
AST-parses every file under core/ and interface/ and took one suite file from 14
seconds to over 155. This file is about those two methods, so it opts out.
"""

import pytest

pytestmark = pytest.mark.real_architecture_index


def test_architecture_index_defers_build_during_foreground_quiet_window(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    monkeypatch.setenv("AURA_FOREGROUND_ARCHITECTURE_INDEX_QUIET_S", "180")

    index = ArchitectureIndex(project_root=tmp_path)

    assert index.build() == 0
    assert index.query("sample module") == ""
    assert str(index._deferred_reason).startswith("foreground_quiet_window:")


def test_architecture_index_force_build_overrides_foreground_deferral(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")

    index = ArchitectureIndex(project_root=tmp_path)

    assert index.build(force=True) == 1
    assert "sample module" in index.query("sample module").lower()


def test_architecture_index_defers_during_desktop_safe_boot(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_FOREGROUND_ARCHITECTURE_INDEX_QUIET_S", "180")

    index = ArchitectureIndex(project_root=tmp_path)

    assert index.build() == 0
    assert str(index._deferred_reason).startswith("foreground_quiet_window:")


def test_architecture_query_never_blocks_interactive_runtime_on_full_scan(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")

    index = ArchitectureIndex(project_root=tmp_path)
    scheduled = []
    monkeypatch.setattr(index, "schedule_background_build", lambda: scheduled.append(True))
    monkeypatch.setattr(
        index,
        "_walk_and_index",
        lambda: (_ for _ in ()).throw(AssertionError("foreground query performed a full scan")),
    )

    assert index.query("memory gateway") == ""
    assert scheduled == [True]


def test_architecture_index_getter_does_not_start_foreground_build(monkeypatch):
    from core.self import architecture_index as module

    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    monkeypatch.setenv("AURA_FOREGROUND_ARCHITECTURE_INDEX_QUIET_S", "180")
    module._index = None

    try:
        index = module.get_architecture_index()
        assert index._index == {}
        assert str(index._deferred_reason).startswith("foreground_quiet_window:")
    finally:
        module._index = None


def test_architecture_index_does_not_spawn_background_thread_in_live_foreground(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ALLOW_FOREGROUND_ARCHITECTURE_INDEX", raising=False)

    index = ArchitectureIndex(project_root=tmp_path)

    assert index.query("sample module") == ""
    assert index._build_thread is None
    assert index._index == {}
    assert str(index._deferred_reason).startswith("foreground_quiet_window:")


def test_architecture_index_foreground_thread_requires_explicit_opt_in(monkeypatch, tmp_path):
    from core.self.architecture_index import ArchitectureIndex

    (tmp_path / "core").mkdir()
    (tmp_path / "interface").mkdir()
    (tmp_path / "core" / "sample.py").write_text(
        '"""sample module"""\nclass Sample:\n    """sample class"""\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.setenv("AURA_ALLOW_FOREGROUND_ARCHITECTURE_INDEX", "1")

    index = ArchitectureIndex(project_root=tmp_path)
    index.schedule_background_build()

    assert index._build_thread is not None
    index._build_thread.join(timeout=2.0)
    assert index._index
