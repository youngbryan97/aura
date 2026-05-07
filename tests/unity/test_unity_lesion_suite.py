from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from core.state.aura_state import AuraState
from core.unity.runtime import UnityRuntime
from core.unity.unity_state import DraftBinding, ReconciledDraftSet
from core.unity.unity_state import SelfWorldBinding, TemporalWindow


def _seed_state() -> AuraState:
    state = AuraState()
    state.cognition.current_objective = "ship the patch"
    state.cognition.active_goals = [{"objective": "ship the patch", "priority": 0.9}]
    state.cognition.working_memory.append({"role": "user", "content": "ship the patch"})
    return state


def test_temporal_binding_lesion_changes_behavior():
    state = _seed_state()
    runtime = UnityRuntime()
    intact = runtime.apply_to_state(state, objective="ship the patch", tick_id="tick_intact").cognition.unity_state

    lesioned = UnityRuntime()
    with patch.object(
        lesioned.temporal_binding,
        "bind_now",
        return_value=TemporalWindow(
            tick_id="tick_lesioned",
            continuity_from_previous=0.0,
            drift_from_previous=1.0,
            phase_lag={"planner": 3.0},
        ),
    ):
        lesioned_state = lesioned.apply_to_state(_seed_state(), objective="ship the patch", tick_id="tick_lesioned").cognition.unity_state

    assert intact is not None and lesioned_state is not None
    assert lesioned_state.unity_score < intact.unity_score


def test_self_world_lesion_reduces_authorship_confidence():
    runtime = UnityRuntime()
    intact = runtime.apply_to_state(_seed_state(), objective="ship the patch", tick_id="tick_auth").cognition.unity_state

    lesioned = UnityRuntime()
    with patch.object(
        lesioned.self_world_binder,
        "bind",
        return_value=SelfWorldBinding(ownership_confidence=0.1, agency_score=0.1, boundary_integrity=0.1),
    ):
        degraded = lesioned.apply_to_state(_seed_state(), objective="ship the patch", tick_id="tick_lesioned_auth").cognition.unity_state

    assert intact is not None and degraded is not None
    assert degraded.agency_ownership_score < intact.agency_ownership_score


def test_draft_lesion_removes_conflict_preservation():
    runtime = UnityRuntime()
    with patch.object(
        runtime,
        "_draft_inputs",
        return_value=[
            {"draft_id": "a", "content": "push now", "coherence": 0.7},
            {"draft_id": "b", "content": "do not push now", "coherence": 0.72},
        ],
    ):
        intact = runtime.apply_to_state(_seed_state(), objective="push now", tick_id="tick_drafts").cognition.unity_state

    lesioned = UnityRuntime()
    with patch.object(
        lesioned.draft_reconciler,
        "reconcile",
        return_value=ReconciledDraftSet(
            chosen=DraftBinding(draft_id="only", claim="push now", support=1.0, conflict=0.0, chosen=True),
            alternatives=[],
            consensus_score=1.0,
            contradiction_score=0.0,
            unresolved_residue=[],
            memory_commit_mode="clean",
        ),
    ):
        degraded = lesioned.apply_to_state(_seed_state(), objective="push now", tick_id="tick_drafts_lesioned").cognition.unity_state

    assert intact is not None and degraded is not None
    assert len(intact.draft_bindings) > len(degraded.draft_bindings)
