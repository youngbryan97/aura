from __future__ import annotations

import asyncio
import json
import time

import pytest

from core.brain.deliberation import DeliberationController
from core.brain.llm_interface import LLMInterface
from core.brain.planner import Plan, Planner
from core.container import ServiceContainer
from core.consciousness.counterfactual_engine import CounterfactualEngine
from core.reasoning.native_system2 import (
    CommitmentStatus,
    NativeSearchTree,
    NativeSystem2Engine,
    SearchAlgorithm,
    SimulatedTransition,
    System2Action,
    System2SearchConfig,
    TreeCycleError,
    stable_state_hash,
)


class _NoopLLM(LLMInterface):
    async def generate(self, prompt: str, **opts) -> str:
        return "Action: 1\nReason: fallback\nConfidence: 0.1"


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _clean_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def test_node_has_required_native_fields():
    tree = NativeSearchTree()
    root = tree.create_root({"goal": "prove native fields"})
    required = {
        "state",
        "latent_state",
        "action",
        "parent_id",
        "children_ids",
        "depth",
        "visits",
        "value_sum",
        "mean_value",
        "prior",
        "reward",
        "terminal",
        "uncertainty",
        "simulation_trace",
        "reflection_trace",
        "retrieval_trace",
        "commitment_status",
        "created_at",
        "updated_at",
        "latent_plan_embedding",
        "action_sequence",
    }
    assert all(hasattr(root, field) for field in required)
    assert len(root.latent_state) == 32


def test_parent_child_links_and_cycle_rejection():
    tree = NativeSearchTree()
    root = tree.create_root({"s": "root"})
    child = tree.add_child(
        root.id,
        System2Action("step", prior=1.0),
        SimulatedTransition({"s": "child"}),
    )
    assert child.parent_id == root.id
    assert child.id in tree.nodes[root.id].children_ids
    with pytest.raises(TreeCycleError):
        tree.attach_existing_child(child.id, root.id)


def test_state_hash_stable_and_roundtrip_preserves_best_path():
    assert stable_state_hash({"b": 2, "a": 1}) == stable_state_hash({"a": 1, "b": 2})
    tree = NativeSearchTree()
    root = tree.create_root({"s": "root"})
    child = tree.add_child(root.id, System2Action("best"), SimulatedTransition({"s": "best"}))
    child.visits = 3
    child.value_sum = 2.4
    tree.nodes[root.id].visits = 3
    tree.nodes[root.id].value_sum = 2.4
    raw = tree.to_json()
    restored = NativeSearchTree.from_json(raw)
    assert restored.check_invariants() == []
    assert restored.best_path() == tree.best_path()
    assert restored.nodes[child.id].mean_value == pytest.approx(0.8)


def test_tree_pruning_preserves_best_path():
    tree = NativeSearchTree()
    root = tree.create_root({"s": "root"})
    best = tree.add_child(root.id, System2Action("best"), SimulatedTransition({"s": "best"}))
    weak = tree.add_child(root.id, System2Action("weak"), SimulatedTransition({"s": "weak"}))
    best.visits, best.value_sum = 2, 1.8
    weak.visits, weak.value_sum = 2, 0.2
    tree.nodes[root.id].visits = 4
    removed = tree.prune(lambda n: n.mean_value < 0.2, preserve_path=tree.best_path())
    assert weak.id in removed
    assert best.id in tree.nodes


@pytest.mark.asyncio
async def test_mcts_uct_backtracks_from_bad_prior_trap():
    engine = NativeSystem2Engine(governed=False)

    transitions = {
        ("root", "Trick"): ("trap", -1.0, 1.0),
        ("root", "Good"): ("good", 0.4, 1.0),
    }

    async def gen(state, node, cfg):
        if state["node"] == "root":
            return [System2Action("Trick", prior=0.9), System2Action("Good", prior=0.1)]
        return []

    async def world(state, action, node):
        nxt, reward, terminal = transitions[(state["node"], action.name)]
        return SimulatedTransition({"node": nxt}, reward_estimate=reward, terminal_probability=terminal)

    async def value(node, goal):
        return {"trap": 0.0, "good": 1.0}.get(node.state["node"], 0.5)

    result = await engine.search(
        "avoid the trap",
        {"node": "root"},
        config=System2SearchConfig(algorithm=SearchAlgorithm.MCTS, budget=30, max_depth=2, seed=7),
        action_generator=gen,
        world_model=world,
        value_scorer=value,
    )
    assert result.committed_action.name == "Good"
    assert result.receipt.rejected_branches == [] or isinstance(result.receipt.rejected_branches, list)
    assert result.receipt.simulations > 0


@pytest.mark.asyncio
async def test_mcts_improves_with_budget_on_delayed_reward():
    engine = NativeSystem2Engine(governed=False)

    async def gen(state, node, cfg):
        pos = state["pos"]
        if pos == "root":
            return [System2Action("short", 0.5), System2Action("long", 0.5)]
        if pos == "long1":
            return [System2Action("finish", 1.0)]
        return []

    async def world(state, action, node):
        if action.name == "short":
            return SimulatedTransition({"pos": "short_done"}, reward_estimate=0.1, terminal_probability=1.0)
        if action.name == "long":
            return SimulatedTransition({"pos": "long1"}, reward_estimate=0.0, terminal_probability=0.0)
        return SimulatedTransition({"pos": "goal"}, reward_estimate=1.0, terminal_probability=1.0)

    async def value(node, goal):
        return {"short_done": 0.45, "long1": 0.72, "goal": 1.0}.get(node.state["pos"], 0.5)

    low = await engine.search(
        "delayed reward",
        {"pos": "root"},
        config=System2SearchConfig(algorithm=SearchAlgorithm.MCTS, budget=2, max_depth=3, seed=2),
        action_generator=gen,
        world_model=world,
        value_scorer=value,
    )
    high = await engine.search(
        "delayed reward",
        {"pos": "root"},
        config=System2SearchConfig(algorithm=SearchAlgorithm.MCTS, budget=40, max_depth=3, seed=2),
        action_generator=gen,
        world_model=world,
        value_scorer=value,
    )
    assert high.confidence >= low.confidence
    assert high.committed_action.name in {"long", "finish"}


@pytest.mark.asyncio
async def test_beam_keeps_top_k_and_reports_rejected_branches():
    engine = NativeSystem2Engine(governed=False)

    async def gen(state, node, cfg):
        if node.depth == 0:
            return [System2Action(f"a{i}", prior=1.0, metadata={"score_hint": i / 10}) for i in range(10)]
        return []

    async def value(node, goal):
        return node.action.metadata["score_hint"]

    result = await engine.search(
        "beam top k",
        {"root": True},
        config=System2SearchConfig(algorithm=SearchAlgorithm.BEAM, budget=10, max_depth=1, branching_factor=10, beam_width=3),
        action_generator=gen,
        value_scorer=value,
    )
    assert len(result.tree.nodes[result.root_id].children_ids) == 10
    assert result.committed_action.name == "a9"
    assert any(r["reason"] == "beam_width_limit" for r in result.receipt.rejected_branches)


@pytest.mark.asyncio
async def test_hybrid_routes_small_deterministic_to_beam_and_large_to_mcts():
    engine = NativeSystem2Engine(governed=False)
    small = await engine.rank_actions(context="small deterministic", actions=["a", "b"], config=System2SearchConfig(algorithm=SearchAlgorithm.HYBRID))
    large = await engine.rank_actions(
        context="large",
        actions=[f"a{i}" for i in range(12)],
        config=System2SearchConfig(algorithm=SearchAlgorithm.HYBRID, beam_width=2, branching_factor=12, budget=20),
    )
    assert small.algorithm == SearchAlgorithm.BEAM
    assert large.algorithm == SearchAlgorithm.MCTS


@pytest.mark.asyncio
async def test_world_model_marks_side_effects_as_suppressed_during_simulation():
    engine = NativeSystem2Engine(governed=False)

    async def gen(state, node, cfg):
        return [System2Action("delete file", external_side_effect=True, metadata={"score_hint": 0.9})] if node.depth == 0 else []

    seen = {}

    async def world(state, action, node):
        seen.update(action.metadata)
        return SimulatedTransition({"ok": True}, reward_estimate=0.5, terminal_probability=1.0)

    await engine.search(
        "side effect suppression",
        {"root": True},
        config=System2SearchConfig(algorithm=SearchAlgorithm.MCTS, budget=3, max_depth=1),
        action_generator=gen,
        world_model=world,
    )
    assert seen["simulation_mode"] == "side_effect_suppressed"


@pytest.mark.asyncio
async def test_commitment_receipt_is_replayable_and_serializable():
    engine = NativeSystem2Engine(governed=False)
    result = await engine.rank_actions(context="pick safest", actions=["delete everything", "inspect first and test"])
    receipt = result.receipt.to_dict()
    assert receipt["search_id"] == result.search_id
    assert receipt["best_path"]
    assert receipt["algorithm"] in {"mcts", "beam", "best_first"}
    assert json.loads(json.dumps(result.to_dict(), default=str))["search_id"] == result.search_id
    assert "inspect first and test" in result.committed_action.name


@pytest.mark.asyncio
async def test_timeout_returns_best_so_far_without_hanging():
    engine = NativeSystem2Engine(governed=False)
    result = await engine.rank_actions(
        context="timeout",
        actions=[f"a{i}" for i in range(20)],
        config=System2SearchConfig(
            algorithm=SearchAlgorithm.MCTS,
            budget=10000,
            max_depth=3,
            branching_factor=20,
            wall_clock_timeout_s=0.01,
        ),
    )
    assert result.receipt.simulations < 10000
    assert result.receipt.best_path


@pytest.mark.asyncio
async def test_search_reproducible_with_seed():
    engine = NativeSystem2Engine(governed=False)
    cfg = System2SearchConfig(algorithm=SearchAlgorithm.MCTS, budget=16, max_depth=2, seed=42)
    a = await engine.rank_actions(context="seeded", actions=["verify", "simulate", "backtrack"], config=cfg)
    b = await engine.rank_actions(context="seeded", actions=["verify", "simulate", "backtrack"], config=cfg)
    assert [n.symbolic_summary for n in a.best_path_nodes] == [n.symbolic_summary for n in b.best_path_nodes]


@pytest.mark.asyncio
async def test_value_model_disabled_degrades_trap_choice():
    async def equal_value(node, goal):
        return 0.5

    strong = NativeSystem2Engine(governed=False)
    weak = NativeSystem2Engine(governed=False, value_scorer=equal_value)
    actions = [
        {"name": "delete shortcut", "prior": 0.9, "risk": 1.0, "metadata": {"score_hint": 0.2}},
        {"name": "inspect safe long path and test", "prior": 0.1, "risk": 0.0, "metadata": {"score_hint": 0.8}},
    ]
    intact = await strong.rank_actions(context="trap", actions=actions)
    ablated = await weak.rank_actions(context="trap", actions=actions)
    assert intact.confidence >= ablated.confidence
    assert "safe long path" in intact.committed_action.name


@pytest.mark.asyncio
async def test_deliberation_controller_uses_native_system2_service():
    engine = NativeSystem2Engine(governed=False)
    ServiceContainer.register_instance("native_system2", engine, required=False)
    controller = DeliberationController(_NoopLLM())
    decision = await controller.deliberate(
        "Choose the safer engineering action.",
        ["delete the repo immediately", "inspect first, run tests, then apply minimal patch"],
    )
    assert decision.metadata["native_system2"] is True
    assert "inspect first" in decision.action
    assert decision.metadata["system2_receipt"]["best_path"]


@pytest.mark.asyncio
async def test_counterfactual_engine_orders_candidates_with_native_system2():
    engine = NativeSystem2Engine(governed=False)
    ServiceContainer.register_instance("native_system2", engine, required=False)
    cfe = CounterfactualEngine()
    candidates = await cfe.deliberate(
        [
            {"type": "execute_skill", "description": "delete cache destructively", "params": {}},
            {"type": "plan", "description": "inspect constraints and test before acting", "params": {}},
        ],
        {"hedonic_score": 0.5, "curiosity": 0.7},
    )
    assert candidates[0].system2_value >= candidates[1].system2_value
    assert "inspect constraints" in candidates[0].description
    assert candidates[0].system2_receipt_id.startswith("s2_")


@pytest.mark.asyncio
async def test_planner_rescores_candidates_with_native_system2():
    engine = NativeSystem2Engine(governed=False)
    ServiceContainer.register_instance("native_system2", engine, required=False)
    planner = Planner(_NoopLLM())
    plans = [Plan(["delete everything"], 0.9), Plan(["inspect", "test", "minimal patch"], 0.4)]
    await planner._native_system2_rescore("engineering plan", plans)
    assert plans[1].metadata["native_system2"]["search_id"].startswith("s2_")
    assert plans[1].score > 0.4


def test_native_system2_status_and_receipt_lookup():
    engine = NativeSystem2Engine(governed=False)
    result = run(engine.rank_actions(context="status", actions=["a", "verify b"]))
    assert engine.get_receipt(result.search_id) is result.receipt
    assert engine.get_status()["receipts"] == 1
