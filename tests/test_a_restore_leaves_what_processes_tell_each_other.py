"""A restore rewinds her stores and leaves alone what processes tell each other.

The fork held everything under the run's state root, the root's `run`
directory among it: which process holds which model, pid files, heartbeats.
In a whole dry run on 24 September the embedding engine took its model lease
after the snapshot, the next anchor's restore put back a lane file without it,
and the lease's heartbeat found no owner and asked the runtime to shut down.
Every report arm after that got no answer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.config import config
from core.runtime.model_lane_control import (
    ModelLaneController,
    ProcessIdentity,
    acquire_synchronous_in_process_model_lane,
)
from core.runtime.receipts import ReceiptStore
from core.subject import snapshot as fork

pytestmark = pytest.mark.unit


@pytest.fixture
def root(monkeypatch, tmp_path: Path) -> Path:
    state = tmp_path / "run_001" / "state"
    state.mkdir(parents=True)
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    monkeypatch.setenv("AURA_RUNTIME_LEASE_DIR", str(tmp_path / "leases"))
    for name in (*fork.REDIRECTED_DIRECTORIES, *fork.REDIRECTED_FILES, "AURA_MODEL_LANE_STATE_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config.paths, "home_dir_override", state)
    monkeypatch.setattr(fork, "_STORE_CACHE", {})
    return state


def _alive(identity: ProcessIdentity) -> bool:
    return identity.pid == os.getpid() and identity.started_at > 0.0


def test_a_model_lease_taken_after_the_snapshot_survives_the_restore(root: Path, tmp_path: Path) -> None:
    controller = ModelLaneController(
        state_path=root / "run" / "model_lane_control.json",
        receipt_store=ReceiptStore(tmp_path / "receipts"),
        process_alive=_alive,
        process_discovery=None,
    )
    (root / "memory.json").write_text('{"held": "before"}')
    saved = fork._store_state()
    lease = acquire_synchronous_in_process_model_lane(
        owner_id="in-process-sync:embedding-engine",
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        purpose="serve",
        request_gb=0.25,
        controller=controller,
    )
    try:
        fork._restore_stores(saved)
        assert controller.heartbeat_owner_sync(
            lease.decision.owner_id, fencing_token=lease.decision.fencing_token
        ) is True
    finally:
        lease.release(reason="restore-test-complete")


def test_her_own_store_is_still_rewound(root: Path) -> None:
    store = root / "data" / "memory.json"
    store.parent.mkdir(parents=True)
    store.write_text('{"held": "before"}')
    saved = fork._store_state()
    store.write_text('{"held": "during the arm"}')
    (root / "data" / "made-in-the-arm.json").write_text("{}")
    fork._restore_stores(saved)
    assert store.read_text() == '{"held": "before"}'
    assert not (root / "data" / "made-in-the-arm.json").exists()


def test_files_processes_write_to_each_other_are_neither_rewound_nor_removed(root: Path, tmp_path: Path) -> None:
    run = root / "run"
    run.mkdir()
    (run / "model_lane_control.json").write_text('{"owners": {}}')
    saved = fork._store_state()
    (run / "model_lane_control.json").write_text('{"owners": {"embedding": 2}}')
    (run / "heartbeat.pulse").write_text("alive")
    leases = tmp_path / "leases"
    leases.mkdir()
    (leases / "aura-runtime.json").write_text('{"holder": "this process"}')
    fork._restore_stores(saved)
    assert (run / "model_lane_control.json").read_text() == '{"owners": {"embedding": 2}}'
    assert (run / "heartbeat.pulse").read_text() == "alive"
    assert (leases / "aura-runtime.json").exists()
    assert not any("run" in Path(name).relative_to(root).parts for name in saved["entries"])


def test_a_lane_file_moved_out_of_the_root_is_left_alone_too(root: Path, monkeypatch, tmp_path: Path) -> None:
    lane = tmp_path / "elsewhere" / "lanes.json"
    lane.parent.mkdir()
    monkeypatch.setenv("AURA_MODEL_LANE_STATE_PATH", str(lane))
    lane.write_text("before")
    saved = fork._store_state()
    lane.write_text("after")
    fork._restore_stores(saved)
    assert lane.read_text() == "after"
