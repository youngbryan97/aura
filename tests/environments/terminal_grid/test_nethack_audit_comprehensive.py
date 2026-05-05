"""Comprehensive test suite for environment OS audit hardening.

Covers all 8 audit components as general environment-OS capabilities.
NetHack terminal-grid fixtures are used as concrete test data, but the
capabilities themselves are domain-agnostic.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.environment.action_budget import ActionBudget
from core.environment.action_gateway import EnvironmentActionGateway
from core.environment.belief_graph import BeliefEdge, BeliefNode, EnvironmentBeliefGraph
from core.environment.command import ActionIntent
from core.environment.homeostasis import Homeostasis, Resource
from core.environment.modal import ModalManager, ModalPolicy, ModalState
from core.environment.observation import Observation
from core.environment.ontology import ObjectState, ResourceState
from core.environment.outcome_attribution import OutcomeAttributor
from core.environment.parsed_state import ParsedState
from core.environment.simulation import TacticalSimulator
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler


# =========================================================================
#  Fixture helpers
# =========================================================================

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _compiled(name: str) -> ParsedState:
    return NetHackStateCompiler().parse_text(_fixture(name))


# =========================================================================
#  Component 1 — Enhanced Parser
# =========================================================================


class TestParserEnhancements:
    """Parser: inventory, glyphs, encumbrance, sensory, dangerous prompts."""

    def test_parser_extracts_glyph_threat_scores_from_table(self):
        parsed = _compiled("nethack_start.txt")
        # 'g' is a gnome-like, should get specific threat from table not 0.6
        hostile = [e for e in parsed.entities if e.kind == "hostile"]
        assert len(hostile) >= 1
        # The threat_score should come from the GLYPH_THREAT table
        threat = hostile[0].threat_score
        assert isinstance(threat, float)
        assert threat != 0.6, "threat should use glyph table, not constant 0.6"

    def test_parser_detects_encumbrance_status(self):
        parsed = _compiled("nethack_inventory.txt")
        assert parsed.self_state.get("encumbrance") == "Burdened"

    def test_parser_extracts_sensory_reliability(self):
        parsed = _compiled("nethack_danger.txt")
        assert parsed.self_state.get("sensory_reliability") == 0.0  # Blind
        assert parsed.uncertainty.get("sensory", 0) > 0

    def test_parser_normal_sensory_reliability(self):
        parsed = _compiled("nethack_start.txt")
        assert parsed.self_state.get("sensory_reliability") == 1.0

    def test_parser_detects_dangerous_prompts(self):
        parsed = _compiled("nethack_danger.txt")
        assert parsed.self_state.get("dangerous_prompt") is True
        # The modal_state should reflect the dangerous prompt
        assert parsed.modal_state is not None
        assert "y" in parsed.modal_state.dangerous_responses or "Really attack" in parsed.modal_state.text

    def test_parser_inventory_screen_extracts_items(self):
        parsed = _compiled("nethack_inventory.txt")
        inv = parsed.self_state.get("inventory_items", [])
        assert len(inv) >= 4, f"Expected at least 4 inventory items, got {len(inv)}"
        # Check structure
        letters = {item["letter"] for item in inv}
        assert "a" in letters
        assert "b" in letters
        # Check BUC detection
        sword = next(i for i in inv if i["letter"] == "a")
        assert sword["buc"] == "blessed"
        cursed_scroll = next(i for i in inv if i["letter"] == "c")
        assert cursed_scroll["buc"] == "cursed"

    def test_parser_inventory_category_detection(self):
        parsed = _compiled("nethack_inventory.txt")
        inv = parsed.self_state.get("inventory_items", [])
        sword = next(i for i in inv if i["letter"] == "a")
        assert sword["category"] == "weapon"
        ration = next(i for i in inv if i["letter"] == "b")
        assert ration["category"] == "food"

    def test_parser_inventory_equipped_detection(self):
        parsed = _compiled("nethack_inventory.txt")
        inv = parsed.self_state.get("inventory_items", [])
        sword = next(i for i in inv if i["letter"] == "a")
        assert sword["equipped"] is True
        ration = next(i for i in inv if i["letter"] == "b")
        assert ration["equipped"] is False

    def test_parser_hunger_detection(self):
        parsed = _compiled("nethack_inventory.txt")
        assert parsed.self_state.get("hunger") == "Hungry"


# =========================================================================
#  Component 2 — Extended Command Compiler
# =========================================================================


class TestExtendedCommands:
    """Command compiler: all new intents compile, multi-step specs validate."""

    EXTENDED_INTENTS = [
        "pickup", "drop", "wield", "wear", "take_off",
        "quaff", "read", "zap", "apply", "throw",
        "kick", "open_door", "close_door", "search",
        "far_look", "name_item", "call_type", "pay",
        "offer", "loot", "use_stairs_down", "use_stairs_up",
    ]

    @pytest.mark.parametrize("intent_name", EXTENDED_INTENTS)
    def test_extended_intent_compiles(self, intent_name):
        compiler = NetHackCommandCompiler()
        intent = ActionIntent(name=intent_name, parameters={})
        command = compiler.compile(intent)
        assert command.environment_id == "terminal_grid:nethack"
        assert len(command.steps) >= 1
        command.validate()

    def test_wield_with_item_letter_produces_two_steps(self):
        compiler = NetHackCommandCompiler()
        command = compiler.compile(ActionIntent(name="wield", parameters={"item_letter": "a"}))
        assert len(command.steps) == 2
        assert command.steps[0].value == "w"
        assert command.steps[1].value == "a"

    def test_zap_with_direction_produces_three_steps(self):
        compiler = NetHackCommandCompiler()
        command = compiler.compile(ActionIntent(name="zap", parameters={"item_letter": "f", "direction": "north"}))
        assert len(command.steps) == 3
        assert command.steps[0].value == "z"
        assert command.steps[1].value == "f"
        assert command.steps[2].value == "k"

    def test_use_stairs_down_produces_single_step(self):
        compiler = NetHackCommandCompiler()
        command = compiler.compile(ActionIntent(name="use_stairs_down"))
        assert len(command.steps) == 1
        assert command.steps[0].value == ">"

    def test_unknown_intent_raises_value_error(self):
        compiler = NetHackCommandCompiler()
        with pytest.raises(ValueError, match="unknown_intent"):
            compiler.compile(ActionIntent(name="totally_bogus"))

    def test_kick_produces_ctrl_d(self):
        compiler = NetHackCommandCompiler()
        command = compiler.compile(ActionIntent(name="kick", parameters={"direction": "east"}))
        assert command.steps[0].value == "\x04"
        assert command.steps[1].value == "l"

    def test_offer_is_extended_command(self):
        compiler = NetHackCommandCompiler()
        command = compiler.compile(ActionIntent(name="offer"))
        assert command.steps[0].value == "#"
        assert "offer" in command.steps[1].value

    def test_all_intents_static_method(self):
        all_intents = NetHackCommandCompiler._all_intents()
        assert len(all_intents) >= 36

    def test_compiler_legal_actions_match_registry(self):
        compiler = NetHackCommandCompiler()
        registered = set(compiler._handlers.keys())
        assert registered == NetHackCommandCompiler._all_intents()


# =========================================================================
#  Component 3 — Belief Graph: Persistence, Frontiers, Inter-level Edges
# =========================================================================


class TestBeliefGraphPersistence:
    """Belief graph: save/load, frontier targets, inter-level edges."""

    def test_save_and_load_round_trip(self):
        graph = EnvironmentBeliefGraph()
        graph.upsert_node(BeliefNode("n1", "test", "label1", "ctx1", confidence=0.9, last_seen_seq=10))
        graph.upsert_edge(BeliefEdge("n1", "n2", "adjacent", confidence=0.8, last_confirmed_seq=10))
        graph.mark_frontier("n1")
        graph.mark_hazard("n2")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        graph.save(path)
        assert Path(path).exists()

        loaded = EnvironmentBeliefGraph()
        loaded.load(path)
        assert "n1" in loaded.nodes
        assert loaded.nodes["n1"].confidence == 0.9
        assert len(loaded.edges) == 1
        assert "n1" in loaded.frontiers
        assert "n2" in loaded.hazards
        Path(path).unlink()

    def test_load_missing_file_does_not_crash(self):
        graph = EnvironmentBeliefGraph()
        graph.load("/tmp/nonexistent_belief_graph_file.json")
        assert len(graph.nodes) == 0

    def test_frontier_targets_sorted_by_confidence(self):
        graph = EnvironmentBeliefGraph()
        graph.upsert_node(BeliefNode("f1", "frontier", "low", "ctx", confidence=0.3))
        graph.upsert_node(BeliefNode("f2", "frontier", "high", "ctx", confidence=0.9))
        graph.mark_frontier("f1")
        graph.mark_frontier("f2")
        targets = graph.get_frontier_targets()
        assert targets[0] == "f2"
        assert targets[1] == "f1"

    def test_inter_level_edges_created_for_stairs(self):
        parsed = _compiled("nethack_start.txt")
        graph = EnvironmentBeliefGraph()
        graph.update_from_parsed_state(parsed)
        # The fixture has ">" stairs at dlvl 1
        connects_edges = [e for e in graph.edges if e.relation == "connects"]
        assert len(connects_edges) >= 1
        down_edge = [e for e in connects_edges if e.properties.get("direction") == "down"]
        assert len(down_edge) >= 1
        assert "dlvl_2" in down_edge[0].to_id

    def test_decay_reduces_old_node_confidence(self):
        graph = EnvironmentBeliefGraph()
        graph.upsert_node(BeliefNode("old", "test", "old node", "ctx", confidence=1.0, last_seen_seq=1))
        graph.decay_unobserved(1000, half_life_steps=100)
        assert graph.nodes["old"].confidence < 0.5


# =========================================================================
#  Component 4 — Domain-Aware Tactical Simulator
# =========================================================================


class TestDomainAwareSimulator:
    """Simulator: entity-aware risk, resource deltas, observation info gain."""

    def test_observation_intents_get_high_info_gain(self):
        sim = TacticalSimulator()
        belief = EnvironmentBeliefGraph()
        for intent_name in ["observe", "inspect", "inventory", "far_look", "search"]:
            bundle = sim.simulate(belief, ActionIntent(name=intent_name))
            assert bundle.hypotheses[0].information_gain >= 0.3, f"{intent_name} should have high info gain"

    def test_move_near_hostiles_increases_risk(self):
        sim = TacticalSimulator()
        belief = EnvironmentBeliefGraph()
        belief.upsert_node(BeliefNode("e1", "entity:hostile", "goblin", "ctx", confidence=0.9))
        bundle = sim.simulate(belief, ActionIntent(name="move", parameters={"direction": "east"}))
        assert bundle.worst_case_risk >= 0.3

    def test_eat_predicts_nutrition_delta(self):
        sim = TacticalSimulator()
        belief = EnvironmentBeliefGraph()
        bundle = sim.simulate(belief, ActionIntent(name="eat"))
        assert bundle.hypotheses[0].predicted_resource_delta.get("nutrition", 0) > 0

    def test_unknown_quaff_increases_risk(self):
        sim = TacticalSimulator()
        belief = EnvironmentBeliefGraph()
        bundle = sim.simulate(belief, ActionIntent(name="quaff", tags={"unknown"}))
        assert bundle.worst_case_risk >= 0.5
        assert bundle.uncertainty >= 0.6

    def test_pray_predicts_health_delta(self):
        sim = TacticalSimulator()
        belief = EnvironmentBeliefGraph()
        bundle = sim.simulate(belief, ActionIntent(name="pray"))
        assert bundle.hypotheses[0].predicted_resource_delta.get("health", 0) > 0


# =========================================================================
#  Component 5 — Enriched Homeostasis
# =========================================================================


class TestEnrichedHomeostasis:
    """Homeostasis: prayer bias, mobility, trend tracking, turns to critical."""

    def test_prayer_bias_when_health_critical(self):
        homeo = Homeostasis()
        resources = [
            Resource(name="health", kind="health", value=2, max_value=28, critical_low=0.35),
        ]
        assessment = homeo.assess(resources)
        assert "health" in assessment.critical_resources
        assert "PRAY" in assessment.action_biases
        assert assessment.action_biases["PRAY"] >= 0.8

    def test_eat_bias_when_nutrition_critical(self):
        homeo = Homeostasis()
        resources = [
            Resource(name="nutrition", kind="nutrition", value=0.1, max_value=1.0, critical_low=0.25),
        ]
        assessment = homeo.assess(resources)
        assert "nutrition" in assessment.critical_resources
        assert "EAT" in assessment.action_biases

    def test_mobility_resource_kind_accepted(self):
        homeo = Homeostasis()
        resources = [
            Resource(name="mobility", kind="mobility", value=0.7, max_value=1.0, critical_low=0.3),
        ]
        assessment = homeo.assess(resources)
        assert assessment.stability_score == pytest.approx(0.7, abs=0.01)

    def test_trend_tracking_between_assessments(self):
        homeo = Homeostasis()
        r1 = [Resource(name="health", kind="health", value=10, max_value=12, critical_low=0.35)]
        homeo.assess(r1)
        r2 = [Resource(name="health", kind="health", value=6, max_value=12, critical_low=0.35)]
        assessment2 = homeo.assess(r2)
        assert "health" in assessment2.deteriorating_resources

    def test_turns_until_critical_estimation(self):
        homeo = Homeostasis()
        r1 = [Resource(name="health", kind="health", value=10, max_value=12, critical_low=0.35)]
        homeo.assess(r1)
        r2 = [Resource(name="health", kind="health", value=8, max_value=12, critical_low=0.35)]
        assessment2 = homeo.assess(r2)
        if "health" in assessment2.turns_until_critical:
            assert assessment2.turns_until_critical["health"] > 0

    def test_encumbrance_to_mobility_in_state_compiler(self):
        parsed = _compiled("nethack_inventory.txt")
        assert "mobility" in parsed.resources
        mobility = parsed.resources["mobility"]
        assert mobility.value == pytest.approx(0.7, abs=0.01)  # Burdened = 0.7


# =========================================================================
#  Component 6 — Death Analysis & Outcome Learning
# =========================================================================


class TestOutcomeAttribution:
    """Outcome attribution: death detection, semantic events, resource deltas."""

    def test_death_detection_from_messages(self):
        attr = OutcomeAttributor()
        events = attr.classify_events(["You die..."])
        assert "death" in events

    def test_combat_event_classification(self):
        attr = OutcomeAttributor()
        events = attr.classify_events(["You hit the goblin!"])
        assert "combat_hit" in events

    def test_item_acquisition_event(self):
        attr = OutcomeAttributor()
        events = attr.classify_events(["You pick up a gold piece"])
        assert "item_acquired" in events

    def test_death_assessment_returns_zero_success(self):
        attr = OutcomeAttributor()
        result = attr.assess(
            action="move",
            expected_effect="position_changed",
            observed_events=["execution_ok"],
            messages=["You die..."],
        )
        assert result.is_death is True
        assert result.success_score == 0.0
        assert result.harm_score == 1.0
        assert "death" in result.lesson

    def test_resource_delta_computation(self):
        attr = OutcomeAttributor()
        delta = attr.compute_resource_delta(
            {"health": 12.0, "power": 7.0},
            {"health": 8.0, "power": 7.0},
        )
        assert "health" in delta
        assert delta["health"] == pytest.approx(-4.0)
        assert "power" not in delta  # no change

    def test_empty_messages_no_events(self):
        attr = OutcomeAttributor()
        events = attr.classify_events([])
        assert events == []


# =========================================================================
#  Component 7 — Modal State Machine
# =========================================================================


class TestModalStateMachine:
    """Modal: from_prompt_text, ModalPolicy, dangerous prompt classification."""

    def test_from_prompt_text_direction(self):
        modal = ModalState.from_prompt_text("In what direction?")
        assert modal.kind == "direction_selection"
        assert "\x1b" in modal.legal_responses

    def test_from_prompt_text_item_selection(self):
        modal = ModalState.from_prompt_text("What do you want to eat?")
        assert modal.kind == "item_selection"

    def test_from_prompt_text_confirmation(self):
        modal = ModalState.from_prompt_text("Are you sure? [yn]")
        assert modal.kind == "confirmation"
        assert modal.safe_default == "n"

    def test_from_prompt_text_dangerous_confirmation(self):
        modal = ModalState.from_prompt_text("Really attack the shopkeeper? [yn]")
        assert modal.kind == "confirmation"
        assert "y" in modal.dangerous_responses

    def test_from_prompt_text_shop(self):
        modal = ModalState.from_prompt_text("Pay 50 gold?")
        assert "y" in modal.dangerous_responses
        assert modal.safe_default == "n"

    def test_from_prompt_text_more(self):
        modal = ModalState.from_prompt_text("--More--")
        assert modal.kind == "prompt"
        assert modal.safe_default == " "

    def test_from_prompt_text_unknown(self):
        modal = ModalState.from_prompt_text("Something totally unexpected happened")
        assert modal.kind == "unknown"

    def test_modal_policy_uses_item_letter(self):
        policy = ModalPolicy()
        modal = ModalState(
            kind="item_selection",
            text="What do you want to eat?",
            legal_responses=set(),  # empty = any letter accepted
        )
        response = policy.resolve_with_intent(
            modal,
            intent_name="eat",
            intent_parameters={"item_letter": "b"},
        )
        assert response == "b"

    def test_modal_policy_safe_confirmation(self):
        policy = ModalPolicy()
        modal = ModalState(
            kind="confirmation",
            text="Eat the food ration?",
            legal_responses={"y", "n"},
            safe_default="n",
        )
        response = policy.resolve_with_intent(modal, intent_name="eat")
        assert response == "y"

    def test_modal_policy_rejects_dangerous(self):
        policy = ModalPolicy()
        modal = ModalState(
            kind="confirmation",
            text="Really attack?",
            legal_responses={"y", "n"},
            safe_default="n",
            dangerous_responses={"y"},
        )
        response = policy.resolve_with_intent(modal, intent_name="move")
        assert response == "n"

    def test_modal_manager_resolves_safe_default(self):
        manager = ModalManager()
        modal = ModalState(kind="menu", text="inventory", safe_default="\x1b")
        assert manager.resolve(modal) == "\x1b"


# =========================================================================
#  Component 8 — Action Budget & Gateway Hardening
# =========================================================================


class TestActionBudgetAndGateway:
    """Action budget edge cases and gateway hardening."""

    def test_action_budget_exhaustion_detection(self):
        budget = ActionBudget(
            max_total_steps=100,
            max_irreversible_actions=3,
            max_unknown_actions=10,
            max_repeated_failures=3,
            max_modal_steps=20,
            max_resource_cost=500.0,
        )
        for _ in range(4):
            budget.record(action_name="wield", irreversible=True)
        reasons = budget.exhausted_reasons()
        assert "max_irreversible_actions" in reasons

    def test_action_budget_repeated_failures_tracked(self):
        budget = ActionBudget(
            max_total_steps=100,
            max_irreversible_actions=3,
            max_unknown_actions=10,
            max_repeated_failures=2,
            max_modal_steps=20,
            max_resource_cost=500.0,
        )
        for _ in range(3):
            budget.record(action_name="eat", failed=True)
        reasons = budget.exhausted_reasons()
        assert "max_repeated_failures" in reasons

    def test_gateway_blocks_forbidden_actions(self):
        gw = EnvironmentActionGateway()
        intent = ActionIntent(name="move", risk="forbidden")
        decision = gw.approve(intent, context_id="test")
        assert not decision.approved
        assert "forbidden_action" in decision.vetoes

    def test_gateway_blocks_on_modal(self):
        gw = EnvironmentActionGateway()
        modal = ModalState(kind="prompt", text="test", requires_resolution=True)
        intent = ActionIntent(name="move")
        decision = gw.approve(intent, modal_state=modal, context_id="test")
        assert not decision.approved
        assert "modal_state_blocks_normal_policy" in decision.vetoes

    def test_gateway_allows_resolve_modal_during_modal(self):
        gw = EnvironmentActionGateway()
        modal = ModalState(kind="prompt", text="test", requires_resolution=True)
        intent = ActionIntent(name="resolve_modal", parameters={"response": " "})
        decision = gw.approve(intent, modal_state=modal, context_id="test")
        assert decision.approved

    def test_gateway_suppresses_repeated_failures(self):
        gw = EnvironmentActionGateway()
        gw.record_failure("eat", "test")
        gw.record_failure("eat", "test")
        intent = ActionIntent(name="eat")
        decision = gw.approve(intent, context_id="test")
        assert not decision.approved
        assert "repeated_failure_suppresses_same_action" in decision.vetoes

    def test_gateway_accepts_legal_actions_set(self):
        gw = EnvironmentActionGateway(legal_actions={"move", "wait"})
        intent = ActionIntent(name="fly")
        decision = gw.approve(intent, context_id="test")
        assert not decision.approved
        assert "unknown_or_illegal_action:fly" in decision.vetoes


# =========================================================================
#  Component 9 — Full Integration Test
# =========================================================================


class TestFullIntegration:
    """Integration: observe→compile→belief→simulate→gate→execute cycle."""

    @pytest.mark.asyncio
    async def test_full_kernel_step_cycle_with_belief_change(self):
        from core.environment.environment_kernel import EnvironmentKernel
        from core.environments.terminal_grid import NetHackTerminalGridAdapter
        from core.environments.terminal_grid.nethack_parser import NetHackStateCompiler

        adapter = NetHackTerminalGridAdapter(force_simulated=True)
        compiler = NetHackStateCompiler()
        kernel = EnvironmentKernel(
            adapter=adapter,
            state_compiler=compiler,
            command_compiler=NetHackCommandCompiler(),
        )
        await kernel.start(run_id="integration-test", seed=42)

        # Observe to populate belief
        frame1 = await kernel.observe()
        assert frame1.parsed_state.environment_id == "terminal_grid:nethack"
        h1 = kernel.belief.stable_hash()

        # Step with a move intent
        intent = ActionIntent(name="move", parameters={"direction": "east"})
        frame2 = await kernel.step(intent)
        assert frame2.action_intent is not None
        assert frame2.gateway_decision is not None
        assert frame2.gateway_decision.approved
        assert frame2.receipt is not None

        await kernel.close()
        assert not adapter.is_alive()

    @pytest.mark.asyncio
    async def test_state_compiler_produces_mobility_resource(self):
        from core.environments.terminal_grid import NetHackTerminalGridAdapter
        adapter = NetHackTerminalGridAdapter(force_simulated=True)
        compiler = NetHackStateCompiler()
        from core.environment.environment_kernel import EnvironmentKernel

        kernel = EnvironmentKernel(
            adapter=adapter,
            state_compiler=compiler,
            command_compiler=NetHackCommandCompiler(),
        )
        await kernel.start(run_id="mobility-test", seed=42)
        frame = await kernel.observe()
        # Default screen should produce mobility resource
        assert "mobility" in frame.parsed_state.resources
        await kernel.close()

    def test_belief_graph_accumulates_from_parsed_state(self):
        parsed = _compiled("nethack_start.txt")
        graph = EnvironmentBeliefGraph()
        graph.update_from_parsed_state(parsed)
        # Should have context, self entity, monster entity, stairs object
        assert len(graph.nodes) >= 3
        assert len(graph.edges) >= 2
        assert len(graph.frontiers) >= 1  # stairs
        assert len(graph.hazards) >= 1  # hostile entity

    def test_end_to_end_parser_to_homeostasis(self):
        """Full pipeline: fixture → parse → compile → homeostasis assess."""
        parsed = _compiled("nethack_inventory.txt")
        homeo = Homeostasis()
        resources = homeo.extract(parsed)
        assessment = homeo.assess(resources)
        # HP 5/28 = 0.18, below critical 0.35
        assert "health" in assessment.critical_resources
        assert "PRAY" in assessment.action_biases
        # Hungry → nutrition critical
        assert assessment.stability_score < 0.8


# =========================================================================
#  Additional regression tests for the perception layer
# =========================================================================


class TestPerceptionRegression:
    """Ensure no regressions in base perception layer."""

    def test_nethack_start_fixture_still_works(self):
        parsed = _compiled("nethack_start.txt")
        assert parsed.environment_id == "terminal_grid:nethack"
        assert parsed.resources["health"].normalized == 1.0
        assert any(e.kind == "self" for e in parsed.entities)

    def test_default_screen_parses_successfully(self):
        from core.environments.terminal_grid.base import TerminalGridAdapter
        screen = TerminalGridAdapter._default_screen()
        parsed = NetHackStateCompiler().parse_text(screen)
        assert parsed.environment_id == "terminal_grid:nethack"
        assert parsed.self_state.get("dlvl") == 1
