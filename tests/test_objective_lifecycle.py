from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core.agency.initiative_arbiter import InitiativeArbiter
from core.continuity import ContinuityEngine, ContinuityRecord
from core.goals.goal_engine import GoalEngine
from core.goals.objective_lifecycle import (
    ForegroundObjectiveDisposition,
    classify_foreground_objective,
    finalize_foreground_turn_state,
    is_actionable_foreground_objective,
    is_ephemeral_conversation_turn,
    is_foreground_objective_origin,
    is_transient_foreground_projection,
)
from core.runtime.proposal_governance import propose_governed_initiative_to_state
from core.runtime.subprocess_gateway import SubprocessGateway
from core.state.aura_state import AuraState


def _contaminated_item(goal: str) -> dict:
    return {
        "goal": goal,
        "source": "executive_authority",
        "type": "executive_closure",
        "metadata": {
            "foreground_turn": True,
            "initiative_source": "executive_closure",
            "initiative_kind": "executive_closure",
            "workspace_focus": None,
            "workspace_source": None,
        },
    }


def _legacy_contaminated_item(goal: str) -> dict:
    return {
        "goal": goal,
        "source": "executive_authority",
        "metadata": {
            "initiative_source": "executive_closure",
            "initiative_kind": "executive_closure",
        },
    }


def test_ephemeral_classifier_separates_dialogue_from_actionable_work():
    assert is_foreground_objective_origin("desktop-ui")
    assert is_foreground_objective_origin("native_shell")
    assert not is_foreground_objective_origin("background_ui")
    assert is_ephemeral_conversation_turn("Ok. Once more. You with me?")
    assert is_ephemeral_conversation_turn(
        "Latency sample 3: answer in one short sentence that includes the sample number."
    )
    assert is_ephemeral_conversation_turn("How are you learning?")
    assert is_ephemeral_conversation_turn("What did you learn today?")
    assert is_ephemeral_conversation_turn("How can I help you build confidence?")
    assert is_ephemeral_conversation_turn(
        "Check in with your state and tell me right here: "
        "more settled or more strained than an hour ago?"
    )
    assert is_ephemeral_conversation_turn(
        "Check in with yourself: are you feeling okay after that?"
    )
    assert not is_ephemeral_conversation_turn("Can you investigate runtime pressure?")
    assert is_actionable_foreground_objective("Can you update the runtime manifest?")
    assert is_actionable_foreground_objective(
        "Check the deployment status and report any failed replicas"
    )
    assert is_actionable_foreground_objective(
        "Check in with the deployment team and document the release decision"
    )
    for quoted_chat_action in (
        "Investigate why Aura keeps asking 'are you ok' and fix it",
        "Analyze what do you think caused the health poll failure",
        "Please debug the repeated 'tell me more' objective leak",
    ):
        assert classify_foreground_objective(quoted_chat_action) is (
            ForegroundObjectiveDisposition.TASK
        )
    assert not is_ephemeral_conversation_turn("Investigate the event-loop latency regression")
    assert not is_ephemeral_conversation_turn("Improve memory retrieval accuracy")
    assert is_actionable_foreground_objective("Ensure production remains stable")
    assert classify_foreground_objective("Production stability matters") is (
        ForegroundObjectiveDisposition.UNKNOWN
    )


def test_state_kernel_and_goal_api_import_in_fresh_interpreter():
    result = SubprocessGateway().run(
        [
            sys.executable,
            "-c",
            (
                "from core.state.aura_state import AuraState; "
                "from core.kernel.aura_kernel import AuraKernel; "
                "from core.goals import GoalEngine; "
                "assert AuraState and AuraKernel and GoalEngine"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        timeout=20.0,
        read_only=True,
        source="proof_tooling:objective_lifecycle_fresh_import",
        accelerator_capability="none",
    )

    assert result.returncode == 0, result.stderr


def test_foreground_finalization_scrubs_only_derived_turn_projections():
    prompt = "Ok. Once more. You with me?"
    state = AuraState.default()
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "desktop_ui"
    ambiguous = {"goal": prompt, "source": "executive_authority"}
    state.cognition.pending_initiatives = [_contaminated_item(prompt), ambiguous]
    state.cognition.active_goals = [
        {
            "description": prompt,
            "source": "executive_closure",
            "metadata": {"foreground_turn": True},
        },
        {
            "description": prompt,
            "source": "mission_engine",
            "mission_id": "mission-1",
        },
        {"description": prompt, "source": "executive_authority"},
    ]
    state.cognition.modifiers = {
        "executive_objective": prompt,
        "executive_hysteresis": {"committed_objective": prompt},
    }
    state.response_modifiers["executive_closure"] = {
        "selected_objective": prompt,
        "committed_objective": prompt,
        "hysteresis_active": True,
    }

    receipt = finalize_foreground_turn_state(
        state,
        objective=prompt,
        origin="desktop_ui",
    )

    assert receipt["completed"] is True
    assert receipt["cleared_current"] is True
    assert receipt["removed_pending"] == 1
    assert receipt["removed_active_goals"] == 1
    assert state.cognition.current_objective is None
    assert state.cognition.current_origin is None
    assert state.cognition.pending_initiatives == [ambiguous]
    assert state.cognition.active_goals[0]["mission_id"] == "mission-1"
    assert state.cognition.active_goals[1]["source"] == "executive_authority"
    assert "executive_objective" not in state.cognition.modifiers
    assert "executive_hysteresis" not in state.cognition.modifiers
    closure = state.response_modifiers["executive_closure"]
    assert closure["selected_objective"] == ""
    assert closure["hysteresis_active"] is False


@pytest.mark.parametrize(
    "background_objective",
    [
        "Investigate thermal pressure",
        "You still with me?",
    ],
)
def test_foreground_finalization_preserves_new_background_objective(
    background_objective,
):
    state = AuraState.default()
    state.cognition.current_objective = background_objective
    state.cognition.current_origin = "curiosity"

    receipt = finalize_foreground_turn_state(
        state,
        objective="You still with me?",
        origin="api",
    )

    assert receipt["preserved_background"] is True
    assert state.cognition.current_objective == background_objective
    assert state.cognition.current_origin == "curiosity"


def test_aura_state_sanitizer_removes_rehydrated_conversation_projections():
    prompt = "Ok. Once more. You with me?"
    state = AuraState.default()
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "desktop_ui"
    state.cognition.pending_initiatives = [_contaminated_item(prompt)]
    state.cognition.active_goals = [
        {
            "description": prompt,
            "source": "executive_closure",
            "metadata": {"foreground_turn": True},
        },
        {"description": "Investigate runtime pressure", "source": "executive_closure"},
    ]

    state.cognition.sanitize_restored_autonomy_state()

    assert state.cognition.current_objective is None
    assert not state.cognition.pending_initiatives
    assert state.cognition.active_goals == [
        {"description": "Investigate runtime pressure", "source": "executive_closure"}
    ]


def test_restore_preserves_unknown_and_explicitly_bound_foreground_work():
    state = AuraState.default()
    state.cognition.current_objective = "Production stability matters"
    state.cognition.current_origin = "native-shell"
    bound = _legacy_contaminated_item("What did you learn today?")
    bound["task_id"] = "task-1"
    state.cognition.active_goals = [bound]

    state.cognition.sanitize_restored_autonomy_state()

    assert state.cognition.current_objective == "Production stability matters"
    assert state.cognition.current_origin == "native-shell"
    assert state.cognition.active_goals == [bound]


def test_ordinary_state_derivation_preserves_live_foreground_turn():
    prompt = "Ok. Once more. You with me?"
    state = AuraState.default()
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "desktop_ui"

    derived = state.derive("proprioceptive_loop")

    assert derived.cognition.current_objective == prompt
    assert derived.cognition.current_origin == "desktop_ui"


def test_state_derivation_does_not_own_autonomy_cleanup():
    prompt = "Researching The Unix Philosophy and the Art of Minimalist Tooling"
    state = AuraState.default()
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "system"

    derived = state.derive("proprioceptive_loop")

    assert derived.cognition.current_objective == prompt
    assert derived.cognition.current_origin == "system"


def test_tick_boundary_owns_autonomy_cleanup():
    prompt = "Researching The Unix Philosophy and the Art of Minimalist Tooling"
    state = AuraState.default()
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "system"

    state.prepare_tick_boundary()

    assert state.cognition.current_objective is None
    assert state.cognition.current_origin == "system"


def test_restored_state_preserves_actionable_question_and_ambiguous_goal():
    state = AuraState.default()
    state.cognition.current_objective = "Can you investigate the health poll?"
    state.cognition.current_origin = "api"
    ambiguous = {
        "description": "Could this make the health poll healthier?",
        "source": "executive_authority",
        "metadata": {"initiative_source": "executive_closure"},
    }
    state.cognition.active_goals = [ambiguous]

    state.cognition.sanitize_restored_autonomy_state()

    assert state.cognition.current_objective == "Can you investigate the health poll?"
    assert state.cognition.active_goals == [ambiguous]


def test_continuity_does_not_rehydrate_or_narrate_completed_chat_turn():
    prompt = "Ok. Once more. You with me?"
    engine = ContinuityEngine()
    engine._record = ContinuityRecord(
        last_shutdown=engine._boot_time - 3600.0,
        last_shutdown_reason="graceful",
        total_uptime_seconds=10_000.0,
        session_count=4,
        last_conversation_summary="A normal conversation ended cleanly.",
        identity_hash="identity",
        current_objective=prompt,
        pending_initiatives=1,
        pending_initiative_details=[prompt],
        active_goal_details=[prompt, "Investigate runtime pressure"],
    )
    engine._gap_seconds = 3600.0

    obligations = engine.get_obligations()
    state = engine.apply_to_state(AuraState.default())
    waking_context = engine.get_waking_context()

    assert obligations["current_objective"] == ""
    assert obligations["pending_initiatives"] == []
    assert obligations["active_goals"] == ["Investigate runtime pressure"]
    assert state.cognition.current_objective is None
    assert prompt not in waking_context


def test_continuity_purges_restored_introspective_check_in_everywhere():
    prompt = (
        "Check in with your state and tell me right here: "
        "more settled or more strained than an hour ago?"
    )
    engine = ContinuityEngine()
    engine._record = ContinuityRecord(
        last_shutdown=engine._boot_time - 900.0,
        last_shutdown_reason="graceful",
        total_uptime_seconds=10_000.0,
        session_count=8,
        last_conversation_summary="<answer>more settled</answer>",
        identity_hash="identity",
        current_objective=prompt,
        pending_initiatives=1,
        pending_initiative_details=[prompt],
        active_goal_details=[prompt, "Investigate runtime pressure"],
        subject_thread=f"Mode=sovereign | Objective={prompt} | Commitments=none",
    )
    engine._gap_seconds = 900.0

    obligations = engine.get_obligations()
    state = engine.apply_to_state(AuraState.default())
    waking_context = engine.get_waking_context()

    assert obligations["current_objective"] == ""
    assert obligations["pending_initiatives"] == []
    assert obligations["active_goals"] == ["Investigate runtime pressure"]
    assert state.cognition.current_objective is None
    assert prompt not in obligations["subject_thread"]
    assert prompt not in waking_context


@pytest.mark.asyncio
async def test_proposal_and_arbiter_quarantine_transient_foreground_projection():
    prompt = "Ok. Once more. You with me?"
    state = AuraState.default()

    unchanged, decision = await propose_governed_initiative_to_state(
        state,
        prompt,
        source="executive_closure",
        kind="executive_closure",
        metadata={"foreground_turn": True},
    )

    assert unchanged is state
    assert decision["action"] == "quarantined"
    assert decision["reason"] == "transient_foreground_projection"

    state.cognition.pending_initiatives = [_contaminated_item(prompt)]
    assert is_transient_foreground_projection(state.cognition.pending_initiatives[0])
    assert await InitiativeArbiter().arbitrate(state) is None
    assert state.cognition.pending_initiatives == []


@pytest.mark.asyncio
async def test_goal_engine_abandons_but_retains_contaminated_durable_row(tmp_path):
    path = tmp_path / "goal_lifecycle.db"
    first = GoalEngine(db_path=str(path))
    contaminated = await first.add_goal(
        "Ok. Once more. You with me?",
        source="executive_authority",
        status="in_progress",
        horizon="long_term",
        metadata={
            "foreground_turn": True,
            "initiative_source": "executive_closure",
            "initiative_kind": "executive_closure",
        },
    )
    legacy_contaminated = await first.add_goal(
        "What did you learn today?",
        source="executive_authority",
        status="in_progress",
        horizon="long_term",
        metadata={
            "initiative_source": "executive_closure",
            "initiative_kind": "executive_closure",
        },
    )
    legitimate = await first.add_goal(
        "Investigate runtime pressure",
        source="executive_authority",
        status="in_progress",
        horizon="long_term",
        metadata={
            "initiative_source": "executive_closure",
            "initiative_kind": "executive_closure",
        },
    )
    first.close()

    restarted = GoalEngine(db_path=str(path))
    repaired = restarted.get_goal(contaminated["id"])
    legacy_repaired = restarted.get_goal(legacy_contaminated["id"])
    preserved = restarted.get_goal(legitimate["id"])

    assert repaired is not None
    assert repaired["status"] == "abandoned"
    assert repaired["metadata"]["quarantine"]["reason"] == (
        "transient_foreground_projection"
    )
    assert legacy_repaired is not None
    assert legacy_repaired["status"] == "abandoned"
    assert legacy_repaired["metadata"]["quarantine"]["reason"] == (
        "transient_foreground_projection"
    )
    assert preserved is not None
    assert preserved["status"] == "in_progress"
    restarted.close()
