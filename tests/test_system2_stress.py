"""tests/test_system2_stress.py -- Stress and Edge Case Tests for AGI Modules
========================================================================
Tests edge cases, nuances, and scale limits of the TreeLoRA manager
and the MCTS VRNN planner.
"""
import unittest
import numpy as np

from core.learning.tree_lora_manager import TreeLoRAManager, TaskGradientSignature
from core.cognition.mcts_world_model import LearnedMCTSPlanner
from core.world_model.learned_world_model import LearnedWorldModel, WorldModelConfig

class TestTreeLoRAEdgeCases(unittest.TestCase):
    def setUp(self):
        self.manager = TreeLoRAManager(signature_dim=16, branching_threshold=0.5, layer_count=1)
        self.manager.tree[0]["layer_0_root"].signature_centroid = np.ones(16) / np.linalg.norm(np.ones(16))

    def test_degenerate_signatures_handled_gracefully(self):
        """Test how routing handles all-zero and NaN gradient signatures."""
        sig_zeros = TaskGradientSignature("task_zero", np.zeros(16), 0.5)
        node_id_zeros = self.manager.route_and_adapt(sig_zeros, layer_idx=0)
        self.assertIsNotNone(node_id_zeros)

        # NaNs should not crash the router
        nan_array = np.full(16, np.nan)
        sig_nan = TaskGradientSignature("task_nan", nan_array, 0.5)
        # We expect a fallback or safe handling, not a crash
        # For our mock, cosine_similarity handles NaNs by returning NaN, which might fail > comparisons.
        # We just test it doesn't throw a fatal exception.
        try:
            node_id_nan = self.manager.route_and_adapt(sig_nan, layer_idx=0)
            self.assertIsNotNone(node_id_nan)
        except Exception as e:
            self.fail(f"NaN signature caused an exception: {e}")

    def test_tree_width_explosion_stress(self):
        """Stress test: 1,000 orthogonal tasks to ensure tree handles width scaling."""
        import random
        random.seed(42)
        np.random.seed(42)
        
        branch_count = 0
        for i in range(1000):
            # Generate random orthogonal-ish vectors
            vec = np.random.randn(16)
            vec /= (np.linalg.norm(vec) + 1e-8)
            sig = TaskGradientSignature(f"task_{i}", vec, 0.5)
            node_id = self.manager.route_and_adapt(sig, layer_idx=0)
            if "branch" in node_id:
                branch_count += 1
                
        # With 1000 random tasks, we expect many branches to form to accommodate the diversity.
        self.assertGreater(branch_count, 100)
        
        # Ensure composition still works for a deep/wide node
        composed = self.manager.compose_adapters(node_id, layer_idx=0)
        self.assertIsNotNone(composed["A"])

    def test_prune_root_protection(self):
        """Ensure the root node cannot be pruned (which would destroy the layer)."""
        root_id = "layer_0_root"
        self.manager.prune_node(root_id, layer_idx=0)
        self.assertTrue(self.manager.tree[0][root_id].is_active)


class TestMCTSPlannerEdgeCases(unittest.TestCase):
    def setUp(self):
        config = WorldModelConfig(observation_dim=4, latent_dim=4, hidden_dim=8, action_dim=2)
        self.world_model = LearnedWorldModel(config)
        self.action_space = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        self.scorer = lambda h: float(np.sum(h))

    def test_zero_simulations(self):
        """MCTS should gracefully handle 0 simulations by picking an arbitrary valid action."""
        planner = LearnedMCTSPlanner(
            self.world_model, self.action_space, self.scorer, num_simulations=0
        )
        obs = np.zeros(4)
        action, info = planner.plan(obs)
        self.assertIn(action.tolist(), [a.tolist() for a in self.action_space])
        self.assertEqual(info["root_visits"], 0)

    def test_max_depth_zero(self):
        """MCTS should return immediate heuristic values if max depth is 0."""
        planner = LearnedMCTSPlanner(
            self.world_model, self.action_space, self.scorer, num_simulations=10, max_depth=0
        )
        obs = np.zeros(4)
        action, info = planner.plan(obs)
        # It expands root, but children depth is > max_depth (0), so they just get scored immediately.
        self.assertEqual(info["max_depth_reached"], 1) # root is 0, expands children to depth 1

    def test_high_uncertainty_exploration(self):
        """Ensure high latent variance drives UCB exploration across siblings."""
        planner = LearnedMCTSPlanner(
            self.world_model, self.action_space, self.scorer, num_simulations=20, max_depth=3
        )
        # Manually inflate uncertainty on one branch to guarantee it gets explored more
        # We will mock the _expand method to test PUCT logic directly.
        original_expand = planner._expand
        
        def mock_expand(node, ablate=False):
            original_expand(node, ablate)
            # Make action 0 highly uncertain
            if 0 in node.children:
                node.children[0].uncertainty = 1000.0 
                
        planner._expand = mock_expand
        
        obs = np.zeros(4)
        action, info = planner.plan(obs)
        
        # We expect action 0 to be heavily favored due to massive uncertainty bonus in PUCT
        self.assertTrue(np.array_equal(action, self.action_space[0]))


if __name__ == "__main__":
    unittest.main()
