"""Her own housekeeping declares itself rather than being refused.

Three lanes wrote her own state, with nobody's decision behind them because
nobody had asked, and every one of them was an effect sink — so the write was
refused and the only trace was a line in the log.

Live counts on 2026-09-04: 103 refusals building a blinded workspace, each one
a self-improvement attempt that could not start; 18 where the database took a
state commit and its mirror into shared memory did not, leaving every reader
of the shared state on an older version than the disk; 7 where the model of
herself on disk fell behind the one in memory.
"""

from __future__ import annotations

import inspect


def test_the_blinded_workspace_declares_its_scope():
    from core.self_improvement.blinded_workspace import BlindedWorkspaceFactory

    source = inspect.getsource(BlindedWorkspaceFactory.create)
    assert "local_internal_governed_scope(" in source
    assert 'domain="file_write"' in source


def test_the_whole_workspace_is_built_inside_one_scope():
    from core.self_improvement.blinded_workspace import BlindedWorkspaceFactory

    source = inspect.getsource(BlindedWorkspaceFactory.create)
    # Every write of it happens under the scope, not only the first.
    assert "return self._create(spec, original_module_path)" in source


def test_a_state_commit_without_a_decision_is_scoped_like_one_with():
    from core.state import state_repository

    source = inspect.getsource(state_repository)
    at = source.index('"state_repository.commit"')
    nearby = source[at - 400 : at + 600]
    assert "local_internal_governed_scope(" in nearby
    assert "_commit_to_db(new_state, serialized_data)" in nearby
    assert "_sync_to_shm(new_state, serialized_data)" in nearby


def test_the_self_model_saves_itself_inside_a_scope():
    from core.self_model import SelfModel

    source = inspect.getsource(SelfModel._persist_with_decision)
    assert "local_internal_governed_scope(" in source
    assert 'domain="memory_write"' in source


def test_an_existing_scope_is_still_used_rather_than_a_second_one():
    from core.self_model import SelfModel

    source = inspect.getsource(SelfModel._persist_with_decision)
    active = source.index("get_active_governance() is not None")
    declared = source.index("local_internal_governed_scope(")
    assert active < declared
