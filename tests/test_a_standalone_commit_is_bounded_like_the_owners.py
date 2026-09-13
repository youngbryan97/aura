"""A state written by a process with no vault is bounded the way the vault's is.

`~/.aura/data/state/aura_state.db` reached 46GB: 707 rows from March-April
2026, single rows of 932MB, written by standalone runs that booted the
container without a vault. The proxy's fallback serialized the whole state
every version with no payload cap and no pruning, while the owner's commit
capped a row, fell back to the bounded snapshot, and pruned. Both go through
the owner's commit now.
"""

from __future__ import annotations

import sqlite3

import pytest

from core.state.aura_state import AuraState
from core.state.state_repository import StateRepository


@pytest.mark.asyncio
async def test_a_standalone_commit_never_writes_more_than_the_payload_cap(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_STRICT_RUNTIME", raising=False)
    db = tmp_path / "standalone.db"
    repo = StateRepository(db_path=str(db), is_vault_owner=False)
    monkeypatch.setattr(repo, "_resolve_transport", lambda: None)

    state = AuraState.default()
    # A state that would have been a giant row: well past the cap.
    padding = "x" * (repo.DB_PAYLOAD_MAX_BYTES + 1_000_000)
    state.world.facts["padding"] = padding

    try:
        await repo.commit(state, cause="test_standalone")
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT LENGTH(state_json) FROM state_log").fetchall()
    finally:
        # The aiosqlite worker is a non-daemon thread; a repository left open
        # keeps the interpreter from exiting, and the test hangs at teardown
        # rather than failing.
        await repo.close()
    assert len(rows) == 1
    (written,) = rows[0]
    assert written <= repo.DB_PAYLOAD_MAX_BYTES, (
        f"a standalone commit wrote {written} bytes; the owner's cap is {repo.DB_PAYLOAD_MAX_BYTES}"
    )


@pytest.mark.asyncio
async def test_a_standalone_commit_counts_toward_pruning(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_STRICT_RUNTIME", raising=False)
    db = tmp_path / "standalone.db"
    repo = StateRepository(db_path=str(db), is_vault_owner=False)
    monkeypatch.setattr(repo, "_resolve_transport", lambda: None)
    before = repo._commit_counter
    try:
        await repo.commit(AuraState.default(), cause="test_standalone")
    finally:
        await repo.close()
    assert repo._commit_counter == before + 1, "the fallback commit did not enter the prune schedule"
