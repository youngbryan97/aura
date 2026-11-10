"""The faults day run 4's fifth start stood on until it was stopped, 10 November 2026.

All of her background work went to the 1.5B model on the CPU, because the brainstem needed 43 GB free with her cortex
already loaded, and there it could not read its prompts inside a 34.4 s first-token budget: 167 stalls in six hours and
no journal entry. Her cortisol crossed the crisis line at 08:32 and each block the crisis made fed it again. The social
store was asked for a relationship with nobody there; a goal stalled by a fault stayed stalled after the fix; and a
resting cycle held back on purpose counted as a turn that failed a person.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from types import SimpleNamespace

import pytest

from core.brain import llm_background_deferral as deferral
from core.consciousness import neurochemical_system as chemistry
from core.consciousness.substrate_authority import (
    ActionCategory,
    AuthorizationDecision,
    SubstrateAuthority,
)


class _LoadedCortex:
    def is_alive(self):
        return True

    def get_supervision_status(self):
        return {"state": "ready"}


class _WarmingCortex(_LoadedCortex):
    def get_supervision_status(self):
        return {"state": "warming"}


def _gate(client):
    return SimpleNamespace(_mlx_client=client, _primary_load_required_gb=lambda: 24.0)


def test_a_loaded_cortex_keeps_no_reserve_for_its_own_load():
    assert deferral.what_the_cortex_still_needs_gb(_gate(_LoadedCortex())) == 0.0
    assert deferral.what_the_cortex_still_needs_gb(_gate(_WarmingCortex())) == 24.0, "a warming cortex has not loaded"
    assert deferral.what_the_cortex_still_needs_gb(_gate(None)) == 24.0


def test_the_brainstem_loads_beside_a_loaded_cortex_with_the_room_the_shed_keeps(monkeypatch):
    import core.brain.llm.how_big_is_the_checkpoint as size
    import core.brain.llm.model_registry as registry
    import core.runtime.service_access as access

    monkeypatch.setattr(registry, "get_brainstem_path", lambda: "/models/brainstem")
    monkeypatch.setattr(size, "_projected_model_footprint_gb", lambda path: 11.0)
    monkeypatch.setenv("AURA_FALLBACK_RESIDENT_RESERVE_GB", "8.0")
    monkeypatch.setattr(access, "resolve_inference_gate", lambda: _gate(_LoadedCortex()))
    # 11 GB of model above the shed's 8: loadable with the 21 to 33 GB the day had free.
    assert deferral._floor_for_the_model_on_this_lane() == pytest.approx(19.0)
    monkeypatch.setattr(access, "resolve_inference_gate", lambda: _gate(None))
    assert deferral._floor_for_the_model_on_this_lane() == pytest.approx(43.0), "the day's number, cortex not loaded"


def _first_token_lane(device: str):
    from core.brain.llm.mlx_client import MLXLocalClient

    class Lane:
        model_path = "/models/Qwen2.5-1.5B-Instruct-4bit"
        _current_request_prompt_chars = 3131
        _last_generation_completed_at = 1.0
        _first_token_sla = MLXLocalClient._first_token_sla
        _first_token_hard_ceiling = MLXLocalClient._first_token_hard_ceiling
        _first_token_absolute_ceiling = MLXLocalClient._first_token_absolute_ceiling

        def _is_deep_solver_lane(self):
            return False

        def _pressure_adaptive_stretch(self):
            return 1.0, ""

    lane = Lane()
    lane.device = device
    return lane


def test_a_lane_on_the_cpu_is_given_time_to_read_its_prompt(monkeypatch):
    import core.brain.llm.mlx_client as mlx

    monkeypatch.setattr(mlx, "_model_is_heavy_lane", lambda path: False)
    monkeypatch.delenv("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", raising=False)
    on_the_gpu, on_the_cpu = _first_token_lane("gpu"), _first_token_lane("cpu")
    assert on_the_gpu._first_token_hard_ceiling(foreground_request=True) == pytest.approx(30.0)
    # The day's 34.4 s was 8 x 1.8 + 20 under the CPU lane's own 120 s ceiling.
    assert on_the_cpu._first_token_sla(foreground_request=True) > 90.0
    assert on_the_cpu._first_token_hard_ceiling(foreground_request=True) == pytest.approx(120.0)


class _Chemistry:
    def __init__(self, cortisol: float):
        self.chemicals = {
            "cortisol": SimpleNamespace(effective=cortisol, uptake_rate=0.008),
            "gaba": SimpleNamespace(effective=0.5),
            "dopamine": SimpleNamespace(effective=0.5),
            "norepinephrine": SimpleNamespace(effective=0.4),
        }
        self.frustrations: list[float] = []
        self._recent_events = deque([(time.monotonic(), "threat", 0.6), (time.monotonic(), "frustration", 0.15)])

    def on_frustration(self, amount):
        self.frustrations.append(amount)


def _authority(cortisol: float) -> tuple[SubstrateAuthority, _Chemistry]:
    authority = SubstrateAuthority()
    chem = _Chemistry(cortisol)
    authority._neurochemical_ref = chem
    return authority, chem


def test_a_block_the_chemistry_made_alone_does_not_feed_the_chemistry():
    authority, chem = _authority(0.93)
    verdict = authority.authorize("reach out", "drive_social", ActionCategory.INITIATIVE)
    assert verdict.decision == AuthorizationDecision.BLOCK
    assert chem.frustrations == [], "the crisis fed the cortisol that made it"


def test_a_block_some_other_gate_made_still_frustrates_her(monkeypatch):
    authority, chem = _authority(0.30)
    monkeypatch.setattr(authority, "_get_somatic_state", lambda content, source, priority: (-1.0, 1.0, True))
    verdict = authority.authorize("reach out", "drive_social", ActionCategory.INITIATIVE)
    assert verdict.decision == AuthorizationDecision.BLOCK
    assert chem.frustrations == [0.15]


def test_a_crisis_says_its_levels_and_what_moved_them_as_it_begins_and_ends(caplog):
    authority, chem = _authority(0.93)
    with caplog.at_level(logging.INFO, logger="Consciousness.SubstrateAuthority"):
        authority.authorize("reach out", "drive_social", ActionCategory.INITIATIVE)
        authority.authorize("reach out again", "drive_social", ActionCategory.INITIATIVE)
        chem.chemicals["cortisol"].effective = 0.31
        authority.authorize("reach out", "drive_social", ActionCategory.INITIATIVE)
    began = [r.getMessage() for r in caplog.records if "began" in r.getMessage()]
    ended = [r.getMessage() for r in caplog.records if "ended after" in r.getMessage()]
    assert len(began) == 1 and "cortisol=0.930" in began[0], "said once, as it starts"
    assert "threat 1x (0.60)" in began[0] and "frustration 1x (0.15)" in began[0]
    assert len(ended) == 1 and "cortisol_crisis" in ended[0]


def test_her_chemistry_keeps_the_events_that_still_move_her_cortisol():
    system = chemistry.NeurochemicalSystem()
    system.on_frustration(0.15)
    system.on_threat(0.4)
    system.on_frustration(0.2)
    moved = chemistry.what_moved_it(system)
    assert moved["frustration"] == (2, pytest.approx(0.35)) and moved["threat"] == (1, pytest.approx(0.4))
    assert chemistry._how_long_an_event_moves_cortisol(system) == pytest.approx(3.0 / 0.008)
    system._recent_events.appendleft((time.monotonic() - 10_000.0, "threat", 0.9))
    assert chemistry.what_moved_it(system)["threat"][0] == 1, "an event gone from her cortisol is not counted"


def test_with_nobody_there_no_relationship_is_asked_for(monkeypatch):
    import core.memory.social_memory as social
    from core.memory.intentional_retrieval import IntentionalRetriever, MemoryStoreType

    class NobodyHere:
        active_user_id = ""

        @property
        def milestones(self):
            raise ValueError("agent_id must be non-empty")

    monkeypatch.setattr(social, "SocialMemory", NobodyHere)
    retrieval = IntentionalRetriever()
    retrieval.wire_default_stores()
    adapter = retrieval._adapters[MemoryStoreType.SOCIAL]
    assert adapter("how is bryan", 3) == []


def test_a_resting_cycle_held_on_purpose_is_deferred_not_a_failed_turn(monkeypatch):
    from core.runtime import turn_outcome
    from core.runtime.turn_outcome import OutcomeStatus, TurnOutcome, UserVisibleState

    recorded = []
    monkeypatch.setattr(turn_outcome, "record_degradation", lambda *a, **k: recorded.append((a, k)))
    turn = TurnOutcome(origin="resting_cycle")
    turn.record_deferral(reason="resting_cycle", authority="response_generation")
    turn.mark_served("", state=UserVisibleState.NOTHING_SERVED)
    receipt = turn.finalize(subsystem="cognitive_engine")
    assert receipt.status is OutcomeStatus.DEFERRED and recorded == []


def test_the_thinking_loop_tells_the_turn_a_held_objective_was_deferred():
    from pathlib import Path

    from source_support import inlined_function_source

    loop = inlined_function_source(
        Path(__file__).resolve().parents[1] / "core/brain/cognitive_engine_thinking_loop.py",
        "_RunsTheThinkingLoop._run_thinking_loop",
    )
    held = loop.index('_empty_thought(mode, "background_deferred")')
    window = loop[held : held + 1200]
    assert "record_deferral(" in window and window.index("record_deferral(") < window.index("return held")


@pytest.fixture
def her_goals(tmp_path, monkeypatch):
    from core.agency import her_goal_at_work as work

    monkeypatch.setattr(work, "records_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(work, "workspaces_dir", lambda: tmp_path / "her_goals")
    monkeypatch.setattr(work, "MEASURING_HOLD", tmp_path / "measuring.hold")
    monkeypatch.setattr(work, "ensure_workspace", lambda record: tmp_path / "her_goals" / record.slug)
    return work


class _OnePass:
    def __init__(self):
        self.asked = 0

    async def think_and_act(self, objective, **kwargs):
        self.asked += 1
        return {"content": "", "turns": 1, "tool_calls": [{"id": "c1", "tool": "read_source", "args": {}, "result": "x"}]}


def test_a_goal_stalled_on_older_source_is_worked_again(her_goals, monkeypatch):
    work = her_goals
    goal = "[AUTONOMOUS INITIATIVE] Make the journal wait for its lane"
    record = work.GoalRecord(goal=goal, slug=work.slug_of(goal), status=work.STALLED, quiet_passes=2,
                             stalled_on="a" * 40)
    asyncio.run(work.save(record))
    monkeypatch.setattr(work, "source_head", lambda: "b" * 40)
    router = _OnePass()
    after = asyncio.run(work.work_a_while(goal, router=router))
    assert router.asked == 1 and after.status == work.WORKING and after.quiet_passes == 0


def test_a_goal_stalled_on_this_source_stays_stalled(her_goals, monkeypatch):
    work = her_goals
    goal = "[AUTONOMOUS INITIATIVE] Something she stopped working"
    record = work.GoalRecord(goal=goal, slug=work.slug_of(goal), status=work.STALLED, quiet_passes=2,
                             stalled_on="b" * 40)
    asyncio.run(work.save(record))
    monkeypatch.setattr(work, "source_head", lambda: "b" * 40)
    router = _OnePass()
    after = asyncio.run(work.work_a_while(goal, router=router))
    assert router.asked == 0 and after.status == work.STALLED


def test_a_goal_records_the_source_it_stalled_on(her_goals, monkeypatch):
    work = her_goals
    goal = "[AUTONOMOUS INITIATIVE] Two quiet passes"
    monkeypatch.setattr(work, "source_head", lambda: "c" * 40)

    class Quiet:
        async def think_and_act(self, objective, **kwargs):
            return {"content": "I will look at this later.", "turns": 1, "tool_calls": []}

    asyncio.run(work.work_a_while(goal, router=Quiet()))
    after = asyncio.run(work.work_a_while(goal, router=Quiet()))
    assert after.status == work.STALLED and after.stalled_on == "c" * 40
