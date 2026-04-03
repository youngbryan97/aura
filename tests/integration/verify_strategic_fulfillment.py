################################################################################

"""tests/verify_strategic_fulfillment.py
Simulation of a multi-step project fulfillment with error recovery.
"""
import asyncio
import unittest
from unittest.mock import MagicMock
from core.data.project_store import ProjectStore
from core.strategic_planner import StrategicPlanner
from core.container import ServiceContainer

class TestStrategicFulfillment(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        self.db_path = "/tmp/test_strategic_fulfillment.db"
        import os
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        self.store = ProjectStore(self.db_path)
        self.brain = MagicMock()
        self.planner = StrategicPlanner(self.brain, self.store)
        ServiceContainer.register_instance("strategic_planner", self.planner)
        ServiceContainer.register_instance("neural_feed", MagicMock())

    async def test_full_fulfillment_cycle(self):
        # 1. Manually create a 5-step project
        project = self.store.create_project("Project Singularity", "Achieve world peace")
        tasks = [
            "Gather global data",
            "Identify conflict points",
            "Propose solutions",
            "Negotiate treaties",
            "Final verification"
        ]
        for i, desc in enumerate(tasks):
            self.store.add_task(project.id, desc, priority=len(tasks)-i)
        
        # 2. Simulate completion of first 2 tasks
        t1 = self.planner.get_next_task(project.id)
        self.assertEqual(t1.description, "Gather global data")
        self.planner.mark_task_complete(t1.id, "Data gathered successfully.")
        
        t2 = self.planner.get_next_task(project.id)
        self.assertEqual(t2.description, "Identify conflict points")
        self.planner.mark_task_complete(t2.id, "Conflicts indexed.")
        
        # 3. Simulate failure on 3rd task
        t3 = self.planner.get_next_task(project.id)
        self.assertEqual(t3.description, "Propose solutions")
        self.planner.mark_task_failed(t3.id, "Algorithm deadlock: No solutions found.")
        
        # 4. Trigger Reflection Loop (Replanning)
        # Mock brain response for replanning
        import json
        replan_json = {
            "project_name": "Project Singularity",
            "tasks": [
                { "description": "Consult external advisors", "priority": 15 },
                { "description": "Alternative algorithm trial", "priority": 14 }
            ]
        }
        
        mock_thought = MagicMock()
        mock_thought.content = f"```json\n{json.dumps(replan_json)}\n```"
        
        # Async mock setup
        self.brain.think.return_value = asyncio.Future()
        self.brain.think.return_value.set_result(mock_thought)
        
        success = await self.planner.replan_project(project.id, "Algorithm deadlock")
        self.assertTrue(success)
        
        # 5. Verify next task is the NEW one
        next_t = self.planner.get_next_task(project.id)
        self.assertEqual(next_t.description, "Consult external advisors")
        
        # 6. Verify status report
        report = await self.planner.get_project_status_report(project.id)
        print("\n--- Project Status Report ---\n", report)
        self.assertIn("Consult external advisors", report)
        self.assertIn("Gather global data", report)

if __name__ == '__main__':
    unittest.main()


##
