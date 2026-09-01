"""One append-only log, and a state that is a fold over it.

Cards A5.1-A5.18, A2.5, A2.6, A2.7, A2.18.
"""
from __future__ import annotations

import pytest

from core.runtime.event_spine import (
    EventLog,
    Lane,
    OwnershipViolation,
    Projection,
    reset_spine_for_test,
)


def _spine():
    spine = reset_spine_for_test()

    def conversation(state, event):
        state["messages"] = state.get("messages", 0) + 1

    def work(state, event):
        state["files"] = sorted(set(state.get("files", [])) | {event.payload["path"]})

    spine.projection.register("conversation", ["said"], ["messages"], conversation)
    spine.projection.register("work", ["edited"], ["files"], work)
    return spine


# ── the log ───────────────────────────────────────────────────────────────

def test_the_log_has_no_delete():
    log = EventLog()
    assert not any(name in dir(log) for name in ("delete", "remove", "truncate", "pop"))


def test_sequence_numbers_are_monotonic_and_never_reused():
    log = EventLog()
    seqs = [log.append("x", {}).seq for _ in range(5)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5


def test_compaction_drops_only_what_is_before_the_snapshot():
    spine = _spine()
    for i in range(10):
        spine.emit("edited", {"path": f"{i}.py"}, lane=Lane.WORK)
    state = spine.projection.state()
    result = spine.log.compact(state, through=5)
    assert result["dropped"] == 5 and result["remaining"] == 5
    assert spine.log.events(since=0)[0].seq == 6


def test_compaction_leaves_the_fold_landing_in_the_same_place():
    spine = _spine()
    for i in range(10):
        spine.emit("edited", {"path": f"{i}.py"}, lane=Lane.WORK)
    before = spine.projection.state()
    spine.log.compact(before, through=6)
    assert spine.projection.at(spine.log.head) == before


def test_compacting_past_the_head_is_refused():
    spine = _spine()
    spine.emit("said", {}, lane=Lane.CONVERSATION)
    with pytest.raises(ValueError, match="the log ends at"):
        spine.log.compact({}, through=99)


def test_rebuilding_before_a_compaction_boundary_is_refused_rather_than_wrong():
    spine = _spine()
    for i in range(6):
        spine.emit("edited", {"path": f"{i}.py"}, lane=Lane.WORK)
    spine.log.compact(spine.projection.state(), through=4)
    with pytest.raises(ValueError, match="events before it are gone"):
        spine.projection.at(2)


# ── the projection ────────────────────────────────────────────────────────

def test_the_state_is_a_fold_and_replays_to_the_same_place():
    spine = _spine()
    spine.emit("said", {"text": "hi"}, lane=Lane.CONVERSATION)
    spine.emit("edited", {"path": "a.py"}, lane=Lane.WORK)
    assert spine.projection.at(spine.log.head) == spine.projection.state()


def test_a_reducer_that_writes_a_key_it_does_not_own_raises():
    spine = reset_spine_for_test()
    spine.projection.register(
        "greedy", ["x"], ["mine"], lambda state, event: state.update({"yours": 1})
    )
    with pytest.raises(OwnershipViolation, match="does not own"):
        spine.emit("x", {})


def test_two_reducers_cannot_claim_the_same_key():
    spine = _spine()
    with pytest.raises(OwnershipViolation, match="already owned"):
        spine.projection.register("rogue", ["said"], ["files"], lambda s, e: None)


def test_a_reducer_that_declares_no_keys_is_refused():
    spine = reset_spine_for_test()
    with pytest.raises(ValueError, match="free-for-all"):
        spine.projection.register("anything", ["x"], [], lambda s, e: None)


def test_every_key_in_the_state_has_an_owner():
    spine = _spine()
    spine.emit("said", {}, lane=Lane.CONVERSATION)
    spine.emit("edited", {"path": "a.py"}, lane=Lane.WORK)
    assert spine.projection.report()["unowned_keys"] == []


# ── checkpoints and rewind ────────────────────────────────────────────────

def test_a_checkpoint_costs_nothing_and_is_just_a_sequence_number():
    spine = _spine()
    spine.emit("said", {}, lane=Lane.CONVERSATION)
    mark = spine.projection.checkpoint("before")
    assert mark.seq == spine.log.head


def test_rewinding_returns_the_earlier_state_and_leaves_the_log_alone():
    spine = _spine()
    spine.emit("edited", {"path": "a.py"}, lane=Lane.WORK)
    spine.projection.checkpoint("before")
    spine.emit("edited", {"path": "b.py"}, lane=Lane.WORK)
    rewound = spine.projection.rewind("before")
    assert rewound["files"] == ["a.py"]
    assert spine.projection.state()["files"] == ["a.py", "b.py"]
    assert spine.log.head == 2


def test_code_and_conversation_rewind_separately():
    """Undo a bad edit without losing the discussion that led to it."""
    spine = _spine()
    spine.emit("said", {"text": "hi"}, lane=Lane.CONVERSATION)
    spine.emit("edited", {"path": "a.py"}, lane=Lane.WORK)
    spine.projection.checkpoint("before")
    spine.emit("said", {"text": "try b"}, lane=Lane.CONVERSATION)
    spine.emit("edited", {"path": "b.py"}, lane=Lane.WORK)

    conversation_only = spine.projection.at(spine.log.head, lanes=[Lane.CONVERSATION])
    assert conversation_only["messages"] == 2
    assert "files" not in conversation_only


def test_rewinding_an_unknown_checkpoint_raises():
    spine = _spine()
    with pytest.raises(KeyError):
        spine.projection.rewind("never-taken")


# ── the stream as a trajectory ────────────────────────────────────────────

def test_events_carry_their_causal_parent_so_the_stream_is_a_trajectory():
    spine = _spine()
    first = spine.emit("said", {}, lane=Lane.CONVERSATION)
    second = spine.emit("edited", {"path": "a.py"}, lane=Lane.WORK, causal_parent=first.seq)
    assert second.causal_parent == first.seq


def test_a_lane_can_be_read_on_its_own():
    spine = _spine()
    spine.emit("said", {}, lane=Lane.CONVERSATION)
    spine.emit("edited", {"path": "a.py"}, lane=Lane.WORK)
    assert [e.kind for e in spine.log.events(lane=Lane.WORK)] == ["edited"]


def test_the_report_says_what_is_retained_and_what_was_compacted_away():
    spine = _spine()
    for i in range(8):
        spine.emit("edited", {"path": f"{i}.py"}, lane=Lane.WORK)
    spine.log.compact(spine.projection.state(), through=3)
    report = spine.report()["log"]
    assert report["events_ever"] == 8
    assert report["compacted_away"] == 3
    assert report["events_retained"] == 5


def test_the_clock_is_injectable_so_a_replay_is_deterministic():
    log = EventLog()
    event = log.append("x", {}, clock=lambda: 1234.5)
    assert event.at == 1234.5


# ── the workspace an agent acts through ───────────────────────────────────

def _tmp_workspace(tmp_path, **kw):
    from core.runtime.agent_workspace import LocalWorkspace

    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    return LocalWorkspace(root=tmp_path / "in", **kw)


def test_a_workspace_with_no_writable_set_is_read_only(tmp_path):
    from core.runtime.agent_workspace import WorkspaceRefusal

    workspace = _tmp_workspace(tmp_path)
    with pytest.raises(WorkspaceRefusal, match="read-only"):
        workspace.write("a.txt", "hi")


def test_a_relative_escape_is_refused(tmp_path):
    from core.runtime.agent_workspace import WorkspaceRefusal

    workspace = _tmp_workspace(tmp_path, writable=[tmp_path / "in"])
    workspace.write("a.txt", "hi")
    with pytest.raises(WorkspaceRefusal, match="outside the granted set"):
        workspace.write("../out/b.txt", "hi")


def test_a_symlink_out_of_the_root_is_refused_because_paths_resolve_first(tmp_path):
    import os

    from core.runtime.agent_workspace import WorkspaceRefusal

    workspace = _tmp_workspace(tmp_path, writable=[tmp_path / "in"])
    os.symlink(tmp_path / "out", tmp_path / "in" / "escape")
    with pytest.raises(WorkspaceRefusal, match="outside the granted set"):
        workspace.write("escape/c.txt", "hi")


def test_a_path_whose_parent_does_not_exist_is_refused(tmp_path):
    from core.runtime.agent_workspace import WorkspaceRefusal

    workspace = _tmp_workspace(tmp_path, writable=[tmp_path / "in"])
    with pytest.raises(WorkspaceRefusal, match="parent directory does not exist"):
        workspace.write("nope/deeper/x.txt", "hi")


def test_a_research_workspace_cannot_be_rooted_at_a_production_tree(tmp_path):
    from core.runtime.agent_workspace import LocalWorkspace, Purpose, WorkspaceRefusal

    (tmp_path / "in").mkdir()
    with pytest.raises(WorkspaceRefusal, match="not a convention"):
        LocalWorkspace(
            root=tmp_path / "in", purpose=Purpose.RESEARCH, production_trees=[tmp_path / "in"]
        )


def test_refusals_are_recorded_so_a_pattern_is_visible(tmp_path):
    from core.runtime.agent_workspace import WorkspaceRefusal

    workspace = _tmp_workspace(tmp_path, writable=[tmp_path / "in"])
    for _ in range(3):
        with pytest.raises(WorkspaceRefusal):
            workspace.write("../out/x", "hi")
    assert len(workspace.refusals) == 3


def test_local_and_remote_would_implement_the_same_shape(tmp_path):
    from core.runtime.agent_workspace import Workspace

    workspace = _tmp_workspace(tmp_path, writable=[tmp_path / "in"])
    assert isinstance(workspace, Workspace)


# ── the architecture lint ─────────────────────────────────────────────────

def test_the_architecture_lint_is_clean_over_what_it_covers():
    import importlib.util

    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "architecture_lint", root / "tools/architecture_lint.py"
    )
    module = importlib.util.module_from_spec(spec)
    import sys as _sys

    _sys.modules["architecture_lint"] = module
    spec.loader.exec_module(module)
    findings, coverage = module.run()
    assert not findings, [f.render() for f in findings]
    assert sum(len(v) for v in coverage.values()) >= 15


def test_every_cross_organ_contract_carries_a_schema_version():
    from core.cognition.action_receipt import SCHEMA_VERSION as receipt
    from core.evidence.packet import SCHEMA_VERSION as packet
    from core.evidence.state_ref import SCHEMA_VERSION as state_ref

    assert len({packet, state_ref, receipt}) == 3
    assert all(v.startswith("aura.") and v.endswith(".v1") for v in (packet, state_ref, receipt))


# ── rewind computes; revert acts ──────────────────────────────────────────
#
# Card A2.18. rewind() hands back what the state WAS and nothing adopts it, so
# a caller who rewound and kept working still had the abandoned work in the
# projection — a correction that lands in a report and not in what she does.


def _steering_projection():
    log = EventLog(capacity=1000)
    projection = Projection(log)

    def _work(state, event):
        if event.kind == "work.step":
            state.setdefault("steps", []).append(event.payload["step"])
        elif event.kind == "work.done":
            state["finished"] = event.payload["goal"]

    def _talk(state, event):
        state.setdefault("said", []).append(event.payload["text"])

    projection.register("work", ("work.step", "work.done"), ("steps", "finished"), _work)
    projection.register("talk", ("said",), ("said",), _talk)
    return log, projection


def test_revert_makes_the_rewound_state_the_live_state():
    log, projection = _steering_projection()
    for i in range(3):
        log.append("work.step", {"step": f"a{i}"}, lane=Lane.WORK)
    projection.advance()
    projection.checkpoint("mark", lane=Lane.WORK)
    for i in range(3, 6):
        log.append("work.step", {"step": f"a{i}"}, lane=Lane.WORK)
    projection.advance()

    assert len(projection.state()["steps"]) == 6
    # rewind alone leaves the projection where it was.
    assert len(projection.rewind("mark")["steps"]) == 3
    assert len(projection.state()["steps"]) == 6
    # revert adopts it.
    assert len(projection.revert("mark")["steps"]) == 3
    assert len(projection.state()["steps"]) == 3


def test_reverting_one_lane_leaves_the_other_alone():
    log, projection = _steering_projection()
    log.append("said", {"text": "do the thing"}, lane=Lane.CONVERSATION)
    for i in range(3):
        log.append("work.step", {"step": f"a{i}"}, lane=Lane.WORK)
    projection.advance()
    projection.checkpoint("mark", lane=Lane.WORK)

    log.append("said", {"text": "actually, the other thing"}, lane=Lane.CONVERSATION)
    for i in range(3, 6):
        log.append("work.step", {"step": f"a{i}"}, lane=Lane.WORK)
    projection.advance()

    after = projection.revert("mark", lanes=(Lane.WORK,), reason="corrected")
    assert len(after["steps"]) == 3, "the overrun was not undone"
    # The correction that caused the revert must survive it.
    assert len(after["said"]) == 2


def test_a_revert_is_in_the_log_rather_than_being_the_absence_of_something():
    log, projection = _steering_projection()
    log.append("work.step", {"step": "a0"}, lane=Lane.WORK)
    projection.advance()
    projection.checkpoint("mark")
    log.append("work.step", {"step": "a1"}, lane=Lane.WORK)
    projection.advance()
    before = log.head

    projection.revert("mark", reason="user changed their mind")

    # The log is never rewritten: the abandoned step is still in the history.
    assert log.head > before
    steps = [e for e in log.events() if e.kind == "work.step"]
    assert len(steps) == 2
    reverts = [e for e in log.events() if e.kind == "spine.reverted"]
    assert len(reverts) == 1
    assert reverts[0].payload["checkpoint"] == "mark"
    assert reverts[0].payload["reason"] == "user changed their mind"
    assert reverts[0].lane is Lane.SYSTEM


def test_reverting_an_unknown_checkpoint_says_so():
    _log, projection = _steering_projection()
    with pytest.raises(KeyError, match="no checkpoint named"):
        projection.revert("never-set")


def test_reverting_reads_the_log_once_not_once_per_owned_key():
    """The first version scanned the whole log per key, which on a full log is
    millions of iterations to answer a question about a handful of kinds."""
    log, projection = _steering_projection()
    for i in range(4000):
        log.append("work.step", {"step": f"a{i}"}, lane=Lane.WORK)
    projection.advance()
    projection.checkpoint("mark", lane=Lane.WORK)
    for i in range(4000, 6000):
        log.append("work.step", {"step": f"a{i}"}, lane=Lane.WORK)
    projection.advance()

    reads = 0
    original = log.events

    def _counted(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original(*args, **kwargs)

    log.events = _counted  # type: ignore[method-assign]
    try:
        projection.revert("mark", lanes=(Lane.WORK,))
    finally:
        log.events = original  # type: ignore[method-assign]

    # One pass to rebuild the state, one to see which kinds moved. Five owned
    # keys across two reducers must not mean five more.
    assert reads <= 2, f"the log was scanned {reads} times"
