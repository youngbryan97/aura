"""Six stores resolved under the home directory whatever the state root said.

The pending chat queue, the memory persister's retry queue and dedup index,
the research sessions, the curated-media progress log and the deferred
retention queue each fell back to `Path.home() / ".aura/data/..."`. A campaign
that gave itself a state root still resolved them into the live instance's
data directory, outside the run and outside the fork.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

OVERRIDES = (
    "AURA_PENDING_CHAT_QUEUE_PATH",
    "AURA_MEMORY_PERSIST_RETRY_QUEUE_PATH",
    "AURA_MEMORY_PERSIST_DEDUP_PATH",
    "AURA_RESEARCH_SESSIONS_DIR",
    "AURA_CURATED_MEDIA_PROGRESS_PATH",
    "AURA_DEFERRED_RETENTION_PATH",
)


@pytest.fixture
def root(monkeypatch, tmp_path: Path) -> Path:
    state = tmp_path / "run_001" / "state"
    state.mkdir(parents=True)
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    return state


def test_every_default_lands_under_the_state_root(root: Path) -> None:
    from core.autonomy.autonomous_research_orchestrator import _default_sessions_dir
    from core.autonomy.content_progress_tracker import _default_progress_path
    from core.autonomy.memory_persister import _default_dedup_path, _default_queue_path
    from core.conversation.chat_preflight import _resolve_pending_queue_path
    from core.memory.deferred_retention import _default_queue_path as _deferred_queue_path

    resolved = {
        "pending chat": _resolve_pending_queue_path(),
        "persist retry": _default_queue_path(),
        "persist dedup": _default_dedup_path(),
        "research sessions": _default_sessions_dir(),
        "curated progress": _default_progress_path(),
        "deferred retention": _deferred_queue_path(),
    }
    outside = {name: path for name, path in resolved.items() if root not in path.parents}
    assert not outside, f"stores resolved outside the state root: {outside}"


def test_no_store_module_fixes_its_path_at_import() -> None:
    """A path taken at import stays under whatever root the process began with."""
    import core.autonomy.autonomous_research_orchestrator as orchestrator
    import core.autonomy.content_progress_tracker as tracker
    import core.autonomy.memory_persister as persister
    import core.conversation.chat_preflight as preflight

    fixed = {
        f"{module.__name__}.{name}"
        for module, name in (
            (preflight, "PENDING_QUEUE_PATH"),
            (persister, "QUEUE_PATH"),
            (persister, "DEDUP_PATH"),
            (orchestrator, "SESSIONS_DIR"),
            (tracker, "DEFAULT_PROGRESS_PATH"),
        )
        if hasattr(module, name)
    }
    assert not fixed, fixed
