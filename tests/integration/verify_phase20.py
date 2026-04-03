################################################################################

import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import sys
import os

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.volition import VolitionEngine
from core.curiosity_engine import CuriosityEngine, CuriosityTopic
from core.ops.singularity_monitor import SingularityMonitor

class TestPhase20(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.orchestrator = MagicMock()
        self.orchestrator.cognitive_engine = MagicMock()
        self.orchestrator.knowledge_graph = MagicMock()
        self.orchestrator.is_busy = False

    async def test_roadmap_awareness(self):
        """Verify that VolitionEngine can scan the brain directory for phases."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.glob", return_value=[Path("/tmp/brain/phase19/task.md"), Path("/tmp/brain/phase20/task.md")]):
                # Mock file reading
                mock_content = {
                    Path("/tmp/brain/phase19/task.md"): "# Phase 19: The Hall of Mirrors",
                    Path("/tmp/brain/phase20/task.md"): "# Phase 20: Singularity Prep"
                }

                def mock_open(path, mode="r"):
                    m = MagicMock()
                    m.__enter__.return_value.read.return_value = mock_content.get(Path(path), "")
                    return m

                with patch("builtins.open", mock_open):
                    volition = VolitionEngine(self.orchestrator)
                    milestones = volition._scan_roadmap()
                    self.assertIn("Phase 19: The Hall of Mirrors", milestones)
                    self.assertIn("Phase 20: Singularity Prep", milestones)
                    
                    # Verify selected goal objective contains current phase
                    goal = volition._check_roadmap()
                    # We might need to force the random roll or call it until it hits
                    found = False
                    for _ in range(100):
                        g = volition._check_roadmap()
                        if g:
                            self.assertIn("Phase 20: Singularity Prep", g["objective"])
                            found = True
                            break
                    self.assertTrue(found, "Roadmap goal should eventually fire")

    async def test_kg_driven_curiosity(self):
        """Verify that CuriosityEngine targets sparse nodes in the KG."""
        self.orchestrator.knowledge_graph.get_sparse_nodes.return_value = ["Quantum Entanglement"]
        
        curiosity = CuriosityEngine(self.orchestrator, MagicMock())
        # Clear queue to trigger novelty search
        curiosity.curiosity_queue.clear()
        
        topic = curiosity._get_next()
        self.assertIsNotNone(topic)
        self.assertIn("Quantum Entanglement", topic.topic)
        self.assertEqual(topic.reason, "knowledge graph novelty search")

    async def test_singularity_heartbeat(self):
        """Verify that SingularityMonitor enables acceleration."""
        self.orchestrator.metacognition.mirror.get_audit_summary.return_value = {
            "health_score": 0.95
        }
        
        monitor = SingularityMonitor(self.orchestrator)
        monitor.improvement_rate = 0.05 # Pre-set to trigger acceleration
        
        monitor.pulse()
        self.assertTrue(monitor.is_accelerated)
        self.assertEqual(monitor.acceleration_factor, 1.5)
        self.assertEqual(self.orchestrator.cognitive_engine.singularity_factor, 1.5)

if __name__ == "__main__":
    unittest.main()


##
