################################################################################

"""tests/test_federated_query.py
Unit test for Federated Query logic in BeliefGraph.
"""
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
from core.world_model.belief_graph import BeliefGraph
from core.container import ServiceContainer

class TestFederatedQuery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Clear container
        ServiceContainer._services = {}
        
        self.graph = BeliefGraph(persist_path=":memory:")
        self.graph.update_belief("User", "is", "Happy", 0.9)
        
        self.sync_service = AsyncMock()
        ServiceContainer.register_instance("belief_sync", self.sync_service)

    async def test_merged_query(self):
        # Mock remote response
        remote_beliefs = [
            {"source": "User", "relation": "likes", "target": "Coffee", "confidence": 0.9}
        ]
        self.sync_service.query_peers.return_value = remote_beliefs
        
        # Execute federated query
        results = await self.graph.query_federated("User")
        
        # Verify merge
        sources = [b["source"] for b in results]
        targets = [b["target"] for b in results]
        
        self.assertIn("User", sources)
        self.assertIn("Happy", targets)
        self.assertIn("Coffee", targets)
        
        # Check discount on remote belief (0.9 * 0.8 = 0.72)
        coffee_belief = next(b for b in results if b["target"] == "Coffee")
        self.assertAlmostEqual(coffee_belief["confidence"], 0.72)

if __name__ == '__main__':
    unittest.main()


##
