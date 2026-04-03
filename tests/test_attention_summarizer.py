################################################################################

"""tests/test_attention_summarizer.py
Unit test for AttentionSummarizer context compression.
"""
import asyncio
import json
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from core.memory.attention import AttentionSummarizer
from core.consciousness.global_workspace import GlobalWorkspace, BroadcastRecord, CognitiveCandidate
from core.world_model.belief_graph import BeliefGraph
from core.container import ServiceContainer

class TestAttentionSummarizer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        self.orchestrator = MagicMock()
        self.brain = AsyncMock()
        self.orchestrator.cognitive_engine = self.brain
        
        self.workspace = GlobalWorkspace()
        ServiceContainer.register_instance("global_workspace", self.workspace)
        
        self.graph = BeliefGraph(persist_path=":memory:")
        ServiceContainer.register_instance("belief_graph", self.graph)
        
        self.summarizer = AttentionSummarizer(self.orchestrator)
        # Low triggers for testing
        self.summarizer.compression_interval = 0.1
        self.summarizer.history_trigger = 5

    async def test_summarization_flow(self):
        # 0. Check container
        ws_reg = ServiceContainer.get("global_workspace")
        self.assertEqual(ws_reg, self.workspace)
        
        # 1. Fill workspace with history
        import time
        for i in range(10):
            candidate = CognitiveCandidate(content=f"event-{i}", source="test-source", priority=0.5)
            record = BroadcastRecord(winner=candidate, losers=[], timestamp=time.time())
            self.workspace.history.append(record)
        
        self.assertEqual(len(self.workspace.history), 10)
        
        # 2. Mock brain response
        mock_response = MagicMock()
        mock_response.content = "Summary: User had coffee and discussed Phase 16."
        self.brain.think = AsyncMock(return_value=mock_response)
        
        # 3. Start summarizer and let it run
        print("\nStarting summarizer...")
        await self.summarizer.start()
        
        # Give it plenty of time to run multiple cycles
        for _ in range(10):
            await asyncio.sleep(0.1)
            if len(self.workspace.history) < 10:
                break
        
        print(f"History size after sleep: {len(self.workspace.history)}")
        await self.summarizer.stop()
        
        # 4. Verify results
        seeds = self.graph.get_strong_beliefs(0.1)
        seed_targets = [b["target"] for b in seeds if b["relation"] == "latent_seed_thought"]
        
        print(f"Found seed targets: {seed_targets}")
        
        self.assertIn("Summary: User had coffee and discussed Phase 16.", seed_targets)
        self.assertLess(len(self.workspace.history), 10)

if __name__ == '__main__':
    unittest.main()


##
