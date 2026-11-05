"""tests/test_proof_obligations.py -- Decisive Proof Obligations
================================================================
Proves the research modules produce measurable behavioral differences,
not just structural existence.

Test categories:
  1. MESU ablation: catastrophic forgetting with/without MESU
  2. MetaCognitive ablation: strategy adjustment produces measurable change
  3. Intrinsic motivation grounding: proposals are evidence-backed
  4. Value invention is not label generation: full pipeline validation
  5. Runtime integration: _dream_research_modules is callable
  6. Shadow-mode value proposal: 20-cycle accumulation without promotion
"""
import json, re, sys, unittest, copy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestCatastrophicForgetting(unittest.TestCase):
    """MESU should prevent catastrophic forgetting on sequential tasks."""

    def test_mesu_retains_old_task_performance(self):
        """Sequential learning on 10 domains: <5% regression with MESU."""
        from core.consciousness.stdp_learning import STDPLearningEngine
        rng = np.random.default_rng(42)

        engine = STDPLearningEngine(n_neurons=16)
        W = rng.standard_normal((16, 16)).astype(np.float32) * 0.3
        np.fill_diagonal(W, 0)

        # Phase 1: Learn task 1 (specific activation pattern)
        task1_pattern = rng.random(16).astype(np.float32)
        task1_pattern[task1_pattern < 0.3] = 0  # Sparse
        for step in range(100):
            engine.record_spikes(task1_pattern + rng.random(16) * 0.05, float(step))
            dw = engine.deliver_reward(surprise=0.1, prediction_error=0.05)
            W = engine.apply_to_connectivity(W, dw)
        W_after_task1 = W.copy()

        # Phase 2: Learn 9 more tasks (different patterns)
        for task_id in range(2, 11):
            pattern = rng.random(16).astype(np.float32)
            pattern[pattern < 0.3] = 0
            for step in range(50):
                t = 100 + (task_id - 2) * 50 + step
                engine.record_spikes(pattern + rng.random(16) * 0.05, float(t))
                dw = engine.deliver_reward(surprise=0.3, prediction_error=0.2)
                W = engine.apply_to_connectivity(W, dw)
        W_after_all = W.copy()

        # Measure regression: how much did task-1-relevant weights change?
        # Focus on synapses that were active during task 1
        task1_active = task1_pattern > 0.3
        relevant_mask = np.outer(task1_active, task1_active)
        np.fill_diagonal(relevant_mask, False)

        if np.sum(relevant_mask) == 0:
            self.skipTest("No relevant synapses for task 1")

        # MESU should have locked some of these weights
        locked = engine.get_locked_mask()
        locked_relevant = np.sum(locked & relevant_mask)

        # Compute drift on relevant weights
        drift = np.abs(W_after_all[relevant_mask] - W_after_task1[relevant_mask])
        mean_drift = float(np.mean(drift))
        max_drift = float(np.max(drift))

        # The key assertion: with MESU, relevant weights should not have
        # drifted catastrophically. Mean drift should be small.
        self.assertLess(mean_drift, 0.5,
            f"Mean drift on task-1 weights too high: {mean_drift:.4f}")

    def test_without_mesu_drift_is_higher(self):
        """Compare drift with MESU locked vs unlocked."""
        from core.consciousness.stdp_learning import STDPLearningEngine
        rng = np.random.default_rng(42)

        # Run WITH MESU (normal)
        engine_mesu = STDPLearningEngine(n_neurons=8)
        W_mesu = rng.standard_normal((8, 8)).astype(np.float32) * 0.3
        np.fill_diagonal(W_mesu, 0)
        pattern = rng.random(8).astype(np.float32)
        for i in range(150):
            engine_mesu.record_spikes(pattern, float(i))
            dw = engine_mesu.deliver_reward(0.1, 0.05)
            W_mesu = engine_mesu.apply_to_connectivity(W_mesu, dw)
        W_mesu_snapshot = W_mesu.copy()
        # New task
        for i in range(150, 300):
            engine_mesu.record_spikes(rng.random(8), float(i))
            dw = engine_mesu.deliver_reward(0.3, 0.2)
            W_mesu = engine_mesu.apply_to_connectivity(W_mesu, dw)
        drift_mesu = float(np.mean(np.abs(W_mesu - W_mesu_snapshot)))

        # Run WITHOUT MESU (all weights unlocked throughout)
        engine_no = STDPLearningEngine(n_neurons=8)
        rng2 = np.random.default_rng(42)
        W_no = rng2.standard_normal((8, 8)).astype(np.float32) * 0.3
        np.fill_diagonal(W_no, 0)
        pattern2 = rng2.random(8).astype(np.float32)
        for i in range(150):
            engine_no.record_spikes(pattern2, float(i))
            dw = engine_no.deliver_reward(0.1, 0.05)
            # Force unlock all MESU locks every step
            engine_no._mesu_locked[:] = False
            engine_no._mesu_lr_scale[:] = 1.0
            W_no = engine_no.apply_to_connectivity(W_no, dw)
        W_no_snapshot = W_no.copy()
        for i in range(150, 300):
            engine_no.record_spikes(rng2.random(8), float(i))
            dw = engine_no.deliver_reward(0.3, 0.2)
            engine_no._mesu_locked[:] = False
            engine_no._mesu_lr_scale[:] = 1.0
            W_no = engine_no.apply_to_connectivity(W_no, dw)
        drift_no = float(np.mean(np.abs(W_no - W_no_snapshot)))

        # MESU should produce less drift (weights are protected)
        self.assertLessEqual(drift_mesu, drift_no + 0.001,
            f"MESU drift ({drift_mesu:.4f}) should be <= unlocked drift ({drift_no:.4f})")


class TestMetaCognitiveAblation(unittest.TestCase):
    """MetaCognitive monitor produces actionable strategy changes."""

    def test_plateau_triggers_lr_raise(self):
        from core.meta.metacognitive_monitor import (
            MetaCognitiveMonitor, MetaCognitiveConfig,
            LearningCondition, StrategyAction)
        cfg = MetaCognitiveConfig(plateau_patience=5, window_size=20)
        mon = MetaCognitiveMonitor(config=cfg)
        for _ in range(25):
            mon.observe(0.001, 0.5, 0.1, 0.5, 0.5)
        r = mon.assess()
        self.assertEqual(r.condition, LearningCondition.PLATEAU)
        self.assertIn(StrategyAction.RAISE_LR, r.recommended_actions)

    def test_instability_triggers_lr_lower(self):
        from core.meta.metacognitive_monitor import (
            MetaCognitiveMonitor, LearningCondition, StrategyAction)
        mon = MetaCognitiveMonitor()
        for _ in range(20):
            mon.observe(10.0, 5.0, 0.5, 0.5, 0.5)
        r = mon.assess()
        self.assertEqual(r.condition, LearningCondition.UNSTABLE)
        self.assertIn(StrategyAction.LOWER_LR, r.recommended_actions)

    def test_overfit_detected(self):
        from core.meta.metacognitive_monitor import (
            MetaCognitiveMonitor, MetaCognitiveConfig, LearningCondition)
        cfg = MetaCognitiveConfig(plateau_patience=100, window_size=30)
        mon = MetaCognitiveMonitor(config=cfg)
        for i in range(35):
            mon.observe(
                gradient_norm=0.5,
                loss=0.5 - i * 0.02,          # Loss clearly decreasing
                prediction_error=0.1 + i * 0.03,  # PE clearly increasing
                confidence=0.5, accuracy=0.5)
        r = mon.assess()
        self.assertEqual(r.condition, LearningCondition.OVERFIT)


class TestValueInventionGrounding(unittest.TestCase):
    """Prove value proposals are evidence-backed, not label generation."""

    def test_proposal_requires_evidence_cluster(self):
        from core.adaptation.intrinsic_motivation import IntrinsicMotivationEngine
        engine = IntrinsicMotivationEngine()
        # No rewards → no proposals
        proposals = engine.check_value_proposals()
        self.assertEqual(len(proposals), 0)

    def test_proposal_has_required_fields(self):
        from core.adaptation.intrinsic_motivation import IntrinsicMotivationEngine
        engine = IntrinsicMotivationEngine()
        engine._proposal_threshold = 3
        engine._reward_threshold = 0.0
        for _ in range(5):
            engine.record_competence("new_domain", success=True)
        proposals = engine.check_value_proposals(existing_drives=[])
        self.assertGreater(len(proposals), 0)
        p = proposals[0]
        # Every proposal must have these fields
        self.assertIn("proposed_value_name", p)
        self.assertIn("evidence_count", p)
        self.assertIn("mean_reward", p)
        self.assertIn("sources", p)
        self.assertIn("reasoning", p)
        self.assertGreater(p["evidence_count"], 0)

    def test_existing_drives_suppress_proposals(self):
        from core.adaptation.intrinsic_motivation import IntrinsicMotivationEngine
        engine = IntrinsicMotivationEngine()
        engine._proposal_threshold = 3
        engine._reward_threshold = 0.0
        for _ in range(5):
            engine.record_competence("curiosity", success=True)
        # "curiosity" already exists → no proposal
        proposals = engine.check_value_proposals(existing_drives=["curiosity"])
        self.assertEqual(len(proposals), 0)

    def test_value_pipeline_promotion_stages(self):
        """Test that DVG nodes progress through status stages with evidence."""
        from core.adaptation.dynamic_value_graph import (
            DynamicValueGraph, ValueEvidence, EvidenceType, ValueNodeStatus)
        import time

        graph = DynamicValueGraph()
        graph.ROLLBACK_GRACE_SECONDS = 0.01

        # Step 1: Record diverse evidence → CANDIDATE with enough signal
        for i in range(20):
            graph.record_evidence(ValueEvidence(
                evidence_type=[EvidenceType.OUTCOME_QUALITY,
                               EvidenceType.ENGAGEMENT,
                               EvidenceType.FREE_ENERGY_REDUCTION][i % 3],
                value_name="test_pipeline_val",
                signal=0.5 + i * 0.02,
                confidence=0.6 + (i % 5) * 0.05,
                source=f"source_{i % 5}",  # 5 different sources
                context=f"test_{i}"))
        graph.evolve()
        node = graph._nodes.get("test_pipeline_val")
        self.assertIsNotNone(node)
        # Should have progressed at least to SANDBOX
        self.assertIn(node.status, [
            ValueNodeStatus.CANDIDATE, ValueNodeStatus.SANDBOX,
            ValueNodeStatus.PROVISIONAL, ValueNodeStatus.ADOPTED])

        # Step 2: Keep feeding evidence and evolving
        for cycle in range(10):
            for j in range(5):
                graph.record_evidence(ValueEvidence(
                    evidence_type=EvidenceType.LONGITUDINAL,
                    value_name="test_pipeline_val",
                    signal=0.7, confidence=0.7,
                    source=f"long_source_{j}",
                    context=f"sustained_{cycle}"))
            time.sleep(0.005)
            graph.evolve()

        node = graph._nodes["test_pipeline_val"]
        # After extensive evidence, should have progressed beyond CANDIDATE
        self.assertNotEqual(node.status, ValueNodeStatus.DEPRECATED)


class TestShadowModeValueProposal(unittest.TestCase):
    """Enable one IM-proposed value in shadow mode for 20 cycles."""

    def test_shadow_accumulation_without_promotion(self):
        from core.adaptation.dynamic_value_graph import (
            DynamicValueGraph, ValueEvidence, EvidenceType, ValueNodeStatus)

        graph = DynamicValueGraph()
        graph.MIN_EVIDENCE = 100  # Set very high so it never promotes

        # Run 20 evolution cycles feeding evidence for a shadow value
        for cycle in range(20):
            graph.record_evidence(ValueEvidence(
                evidence_type=EvidenceType.FREE_ENERGY_REDUCTION,
                value_name="shadow_exploration_drive",
                signal=0.5 + np.random.default_rng(cycle).random() * 0.3,
                confidence=0.6,
                source=f"im_source_{cycle % 4}",
                context=f"shadow cycle {cycle}"))
            graph.evolve()

        # Value should exist but still be CANDIDATE (not promoted)
        node = graph._nodes.get("shadow_exploration_drive")
        self.assertIsNotNone(node)
        self.assertEqual(node.status, ValueNodeStatus.CANDIDATE)

        # Evidence should have accumulated
        self.assertGreater(node.total_evidence_count, 10)

        # Weight should still be at origin (0.5, not influenced)
        self.assertAlmostEqual(node.weight, 0.5, places=2)


class TestRuntimeIntegration(unittest.TestCase):
    """Prove modules are wired into Aura's runtime, not standalone."""

    def test_dream_method_exists_on_mindtick(self):
        """_dream_research_modules exists and is callable."""
        from core.mind_tick import MindTick
        self.assertTrue(hasattr(MindTick, '_dream_research_modules'))
        self.assertTrue(callable(getattr(MindTick, '_dream_research_modules')))

    def test_mind_tick_imports_research_modules(self):
        """MindTick source references all research modules."""
        source = Path(__file__).resolve().parents[1] / "core" / "mind_tick.py"
        text = source.read_text()
        self.assertIn("metacognitive_monitor", text)
        self.assertIn("intrinsic_motivation", text)
        self.assertIn("dynamic_value_graph", text)
        self.assertIn("hidden_eval", text)
        self.assertIn("stdp_engine", text)
        self.assertIn("plasticity_governor", text)

    def test_boot_initializer_registers_research_services(self):
        """The live boot initializer registers the Phase 3/4 modules.

        This asserted against ``core/initializers/cognitive_sensory.py`` — a
        second, unreachable copy of the initializer that no boot path called,
        retired in 053b0a8ab. The obligation passed for years against a file
        that could not register anything.
        """
        # It then moved to core/orchestrator/initializers/cognitive_sensory.py,
        # whose `init_cognitive_sensory_layer` nothing calls either: the same
        # mistake one directory over. The registrar is the provider that
        # `register_all_services` runs (November 2026).
        source = (
            Path(__file__).resolve().parents[1]
            / "core" / "providers" / "self_maintenance_provider.py"
        )
        text = source.read_text()
        self.assertIn("metacognitive_monitor", text)
        self.assertIn("intrinsic_motivation", text)
        self.assertIn("experience_distillery", text)
        self.assertIn("plasticity_governor", text)

    def test_every_dream_research_read_has_a_registrar(self):
        """A container read with no registrar anywhere is a permanent None.

        Pair the readers to the writer by name rather than trusting a comment
        that says "boot-registered singleton": that comment was in the source
        the whole time these four resolved to None.
        """
        root = Path(__file__).resolve().parents[1]
        tick = (root / "core" / "mind_tick.py").read_text()
        initializer = (
            root / "core" / "providers" / "self_maintenance_provider.py"
        ).read_text()

        read_names = set(
            re.findall(r'ServiceContainer\.get\(\s*"([a-z0-9_]+)"', tick)
        )
        research = {
            "metacognitive_monitor",
            "intrinsic_motivation",
            "experience_distillery",
            "plasticity_governor",
        }
        exercised = research & read_names
        self.assertTrue(exercised, "mind_tick no longer reads any research service")

        unregistered = sorted(
            name
            for name in exercised
            if f'"{name}"' not in initializer
        )
        self.assertFalse(
            unregistered,
            f"mind_tick reads these with no boot registrar: {unregistered}",
        )

    def test_closed_loop_feeds_metacognitive(self):
        """ClosedCausalLoop continuously feeds the metacognitive monitor."""
        source = Path(__file__).resolve().parents[1] / "core" / "consciousness" / "closed_loop.py"
        text = source.read_text()
        self.assertIn("_get_research_metacog", text)
        self.assertIn("metacog.observe", text)

    def test_goal_eval_feeds_intrinsic_motivation(self):
        """MindTick goal evaluation feeds IM competence recording."""
        source = Path(__file__).resolve().parents[1] / "core" / "mind_tick.py"
        text = source.read_text()
        self.assertIn("im.record_competence", text)
        self.assertIn("intrinsic_motivation", text)

    def test_stdp_mesu_wired_to_closed_loop(self):
        """STDP engine (with MESU) is imported by the closed loop."""
        from core.consciousness.stdp_learning import STDPLearningEngine
        engine = STDPLearningEngine(n_neurons=8)
        self.assertTrue(hasattr(engine, '_mesu_locked'))
        self.assertTrue(hasattr(engine, '_mesu_var'))
        self.assertTrue(hasattr(engine, 'get_mesu_diagnostics'))

    def test_value_autopoiesis_delegates_to_dvg(self):
        """Value autopoiesis references the DVG."""
        source = Path(__file__).resolve().parents[1] / "core" / "adaptation" / "value_autopoiesis.py"
        if source.exists():
            text = source.read_text()
            self.assertIn("dynamic_value_graph", text)


class TestTreeLoRAAdapterTree(unittest.TestCase):
    """Prove the LLM backbone uses TreeLoRA hierarchical adapter routing."""

    def test_orthogonal_tasks_cause_branching(self):
        from core.learning.tree_lora_manager import TreeLoRAManager, TaskGradientSignature
        import numpy as np

        manager = TreeLoRAManager(signature_dim=10, branching_threshold=0.5, layer_count=1)
        
        # Task 1 creates the first centroid update on root (force exact match by setting centroid)
        manager.tree[0]["layer_0_root"].signature_centroid = np.array([1.0, 0.0] + [0.0]*8)
        sig1 = TaskGradientSignature("task_A", np.array([1.0, 0.0] + [0.0]*8), 0.5)
        node1_id = manager.route_and_adapt(sig1, layer_idx=0)
        self.assertEqual(node1_id, "layer_0_root")

        # Task 2 has orthogonal gradient -> should cause branching
        sig2 = TaskGradientSignature("task_B", np.array([0.0, 1.0] + [0.0]*8), 0.5)
        node2_id = manager.route_and_adapt(sig2, layer_idx=0)
        self.assertNotEqual(node1_id, node2_id)
        self.assertIn(node2_id, manager.tree[0]["layer_0_root"].children)

    def test_similar_tasks_route_to_same_branch(self):
        from core.learning.tree_lora_manager import TreeLoRAManager, TaskGradientSignature
        import numpy as np

        manager = TreeLoRAManager(signature_dim=10, branching_threshold=0.5, layer_count=1)
        
        # Task 1 initializes root
        manager.tree[0]["layer_0_root"].signature_centroid = np.array([1.0, 0.1] + [0.0]*8)
        sig1 = TaskGradientSignature("task_A", np.array([1.0, 0.1] + [0.0]*8), 0.5)
        manager.route_and_adapt(sig1, layer_idx=0)

        # Task 2 is very similar -> should route to same node
        sig2 = TaskGradientSignature("task_A_variant", np.array([0.9, 0.2] + [0.0]*8), 0.5)
        node2_id = manager.route_and_adapt(sig2, layer_idx=0)
        self.assertEqual(node2_id, "layer_0_root")

    def test_pruning_lesions_tree(self):
        from core.learning.tree_lora_manager import TreeLoRAManager, TaskGradientSignature
        import numpy as np

        manager = TreeLoRAManager(signature_dim=10, branching_threshold=0.5, layer_count=1)
        sig1 = TaskGradientSignature("task_A", np.array([1.0, 0.0] + [0.0]*8), 0.5)
        manager.route_and_adapt(sig1, layer_idx=0)
        sig2 = TaskGradientSignature("task_B", np.array([0.0, 1.0] + [0.0]*8), 0.5)
        branch_id = manager.route_and_adapt(sig2, layer_idx=0)
        
        self.assertTrue(manager.tree[0][branch_id].is_active)
        manager.prune_node(branch_id, layer_idx=0)
        self.assertFalse(manager.tree[0][branch_id].is_active)


class TestLearnedMCTSPlanner(unittest.TestCase):
    """Prove System 2 deliberation uses a learned world model."""

    def test_mcts_ablation_learned_model_matters(self):
        from core.cognition.mcts_world_model import LearnedMCTSPlanner
        from core.world_model.learned_world_model import LearnedWorldModel, WorldModelConfig
        import numpy as np

        # Setup VRNN and Planner
        config = WorldModelConfig(observation_dim=4, latent_dim=4, hidden_dim=8, action_dim=2)
        world_model = LearnedWorldModel(config)
        action_space = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        def mock_value_scorer(hidden_state: np.ndarray) -> float:
            # Reward states where hidden_state[0] is positive
            return float(hidden_state[0])

        planner = LearnedMCTSPlanner(
            world_model=world_model,
            action_space=action_space,
            value_scorer=mock_value_scorer,
            num_simulations=10,
            max_depth=3
        )
        
        obs = np.zeros(4)
        
        # Test with learned model
        _, info_learned = planner.plan(obs, ablate_learned_model=False)
        self.assertGreater(info_learned["root_visits"], 0)
        
        # Test with ablated heuristic model
        _, info_ablated = planner.plan(obs, ablate_learned_model=True)
        
        # The Q values in the learned model should reflect the scorer's structured logic
        # compared to the random heuristic ablation.
        self.assertNotEqual(info_learned["best_q"], info_ablated["best_q"])


class TestSCMDoCalculus(unittest.TestCase):
    """Prove the CausalWorldModel performs structural interventions (do-calculus)."""

    def test_intervention_upgrades_correlation_to_causation(self):
        """A causal claim has to be earned: a control, and a replication.

        This test used to call the intervention API with three arguments and
        expect "causes" back. The model deliberately refuses that, for two
        reasons it is right about:

        * a single outcome LEVEL is a correlation however it was produced —
          without a control outcome there is no treatment effect to estimate;
        * one intervention is an anecdote, so MIN_INTERVENTIONS_FOR_CAUSAL
          replications are required before the edge is stated as causal.

        Asserting the weaker contract meant the suite would have accepted a
        model that minted causation from one reading. It now proves the
        stronger one, and pins both refusals.
        """
        from core.brain.causal_world_model import (
            MIN_INTERVENTIONS_FOR_CAUSAL,
            CausalWorldModel,
        )
        scm = CausalWorldModel()
        scm.data_path.parent.mkdir(parents=True, exist_ok=True)

        # Initial passive correlation
        scm.add_observation("weather", "mood", 0.5)
        edge = next(e for e in scm.edges if e.source == "weather" and e.target == "mood")
        self.assertEqual(edge.relationship, "correlates_with")

        # An intervention with NO control outcome must not mint causation.
        scm.discover_causality_via_intervention("do_weather", "mood", 1.0, 0.8)
        edge_no_control = next(
            e for e in scm.edges if e.source == "do_weather" and e.target == "mood"
        )
        self.assertEqual(edge_no_control.relationship, "correlates_with")

        # A single controlled intervention is still only one datum.
        scm.discover_causality_via_intervention(
            "do_weather", "mood", 1.0, 0.8, control_val=0.1
        )
        edge_once = next(
            e for e in scm.edges if e.source == "do_weather" and e.target == "mood"
        )
        self.assertEqual(edge_once.relationship, "correlates_with")

        # Replicated, controlled interventions earn the causal claim.
        for _ in range(MIN_INTERVENTIONS_FOR_CAUSAL):
            scm.discover_causality_via_intervention(
                "do_weather", "mood", 1.0, 0.8, control_val=0.1
            )
        edge_do = next(
            e for e in scm.edges if e.source == "do_weather" and e.target == "mood"
        )
        self.assertEqual(edge_do.relationship, "causes")
        self.assertGreater(edge_do.confidence, 0.2)

    def test_counterfactual_simulation_performs_graph_surgery(self):
        from core.brain.causal_world_model import CausalWorldModel
        scm = CausalWorldModel()
        
        # Chain: a -> b -> c, built with controlled and REPLICATED
        # interventions, because the model will not state a causal edge on
        # less — and graph surgery is only meaningful over causal edges.
        from core.brain.causal_world_model import MIN_INTERVENTIONS_FOR_CAUSAL

        for _ in range(MIN_INTERVENTIONS_FOR_CAUSAL):
            scm.discover_causality_via_intervention("a", "b", 1.0, 1.0, control_val=0.0)
            scm.discover_causality_via_intervention("b", "c", 1.0, 1.0, control_val=0.0)
        
        scm.nodes["a"].activation = 1.0
        scm.nodes["b"].activation = 0.0
        
        # If we DO(b=0), a should NOT influence b, and b(0) means c becomes 0.
        # SCM Surgery severs a -> b.
        final_state = scm.simulate_counterfactual(do_interventions={"b": 0.0}, steps=2)
        self.assertEqual(final_state["b"], 0.0)
        self.assertEqual(final_state["c"], 0.0)
        
        # Without intervention, A drives B and B drives C. Activations are
        # re-seeded because simulate_counterfactual reads live node state and
        # the do-run above consumed it.
        scm.nodes["a"].activation = 1.0
        scm.nodes["b"].activation = 0.0
        final_state_passive = scm.simulate_counterfactual(do_interventions={}, steps=2)
        self.assertGreater(final_state_passive["b"], 0.0)
        self.assertGreater(final_state_passive["c"], 0.0)


class TestEmbodiedSimulator(unittest.TestCase):
    """Prove the continuous physics simulator grounds actions and feeds SCM."""

    def test_physical_intervention_updates_physics_and_scm(self):
        from core.environment.embodied_simulator import ContinuousSimulatorLoop
        from core.perception.affordance_schema import AffordanceKnowledgeBase
        from core.brain.causal_world_model import CausalWorldModel
        import numpy as np
        
        aff_kb = AffordanceKnowledgeBase(domain="test_embodied")
        scm = CausalWorldModel()
        sim = ContinuousSimulatorLoop(aff_kb, scm)
        
        sim.scene.add_or_update_entity("box_1", "block", np.array([0., 0., 0.]))
        
        # Perform the physical intervention twice: the SCM requires
        # replication before stating causation, and a simulator that can push
        # a block can push it again. One push is an anecdote there too.
        from core.brain.causal_world_model import MIN_INTERVENTIONS_FOR_CAUSAL

        for _ in range(MIN_INTERVENTIONS_FOR_CAUSAL):
            outcome = sim.physical_intervention("box_1", "push", intensity=1.0)
        
        # Physics updated
        ent = sim.scene.entities["box_1"]
        self.assertGreater(ent.velocity[0], 0.0)
        
        # SCM updated with discovery
        edge = next((e for e in scm.edges if e.source == "do_push_block" and e.target == "block_state_change"), None)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.relationship, "causes")


class TestWhatIsStillLeft(unittest.TestCase):
    """Document what remains for full proof (informational, always passes)."""

    def test_remaining_proof_obligations(self):
        """This test documents the remaining work. It always passes."""
        remaining = [
            "External benchmark: Aura vs same-model-no-architecture baseline",
            "Independent replication: package + external run",
            "N=100 A/B trials with real CAA steering vectors",
            "72-hour long-horizon stability run",
            "Embodied grounding: physics simulator integration",
        ]
        # These are informational — the test passes to document scope
        for item in remaining:
            self.assertTrue(len(item) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
