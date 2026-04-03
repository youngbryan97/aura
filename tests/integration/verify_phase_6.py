import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure project root is in path
sys.path.append("/Users/bryan/.gemini/antigravity/scratch/autonomy_engine")

from core.agency_core import AgencyCore
from core.brain.identity import IdentityService
from core.memory.knowledge_graph import PersistentKnowledgeGraph

class TestPhase6GoalGenesis(unittest.IsolatedAsyncioTestCase):
    async def test_sparse_node_detection(self):
        print("\nTesting Sparse Node Detection...")
        # Use a temporary DB for testing
        db_path = "/tmp/test_knowledge_phase6.db"
        if os.path.exists(db_path):
            os.remove(db_path)
            
        kg = PersistentKnowledgeGraph(db_path=db_path)
        
        # Add some nodes
        id1 = kg.add_knowledge("Neural Networks", type="concept")
        id2 = kg.add_knowledge("Quantum Computing", type="concept")
        id3 = kg.add_knowledge("Identity", type="concept")
        
        # Add relationships for id1 and id3, leave id2 sparse
        kg.add_relationship(id1, id3, "related_to")
        
        sparse = kg.get_sparse_nodes(limit=1)
        print(f"Sparse nodes: {sparse}")
        self.assertIn("Quantum Computing", sparse)
        print("✓ Sparse node detection identified lowest relationship density.")
        
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_identity_goal_scoring(self):
        print("\nTesting Identity Goal Scoring...")
        identity = IdentityService()
        identity.state.values = ["Agency & Sovereignty", "Intellectual Curiosity"]
        
        goal1 = "Researching advanced sovereignty frameworks"
        goal2 = "Buying groceries"
        
        score1 = identity.score_goal(goal1)
        score2 = identity.score_goal(goal2)
        
        print(f"Score for '{goal1}': {score1}")
        print(f"Score for '{goal2}': {score2}")
        
        self.assertGreater(score1, score2)
        print("✓ IdentityService correctly scores goals based on alignment.")

    async def test_goal_genesis_pathway(self):
        print("\nTesting Goal Genesis Pathway...")
        core = AgencyCore()
        
        # Mocking services
        mock_identity = MagicMock()
        mock_identity.score_goal.return_value = 0.8
        
        mock_kg = MagicMock()
        mock_kg.get_sparse_nodes.return_value = ["Swarm Intelligence"]
        
        with patch('core.container.ServiceContainer.get') as mock_get:
            mock_get.side_effect = lambda name, default=None: {
                "identity": mock_identity,
                "knowledge_graph": mock_kg
            }.get(name, default)
            
            # Mock swarm
            core.swarm = MagicMock()
            core.swarm.spawn_shard = MagicMock(return_value=asyncio.Future())
            core.swarm.spawn_shard.return_value.set_result(True)
            
            # Prepare state
            core.state.curiosity_pressure = 0.9
            core.state.last_goal_genesis_time = 0
            core.state.pending_goals = []
            
            action = await core._pathway_goal_genesis(now=time.time(), idle_seconds=1000.0)
            
            if action:
                print(f"Genesis Action: {action.get('topic')}")
                self.assertEqual(action.get('type'), "genesis_goal")
                self.assertIn("Swarm Intelligence", action.get('topic'))
                self.assertEqual(len(core.state.pending_goals), 1)
                mock_identity.add_long_term_goal.assert_called_once()
                core.swarm.spawn_shard.assert_called_once()
                print("✓ Goal genesis pathway forms, scores, and persists new goals.")
            else:
                self.fail("Goal genesis pathway returned None despite high curiosity.")

if __name__ == "__main__":
    unittest.main()

