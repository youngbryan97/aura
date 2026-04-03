import pytest
################################################################################

import asyncio
import os
import sys
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data.project_store import ProjectStore
from core.strategic_planner import StrategicPlanner
from core.container import ServiceContainer

class MockBrain:
    async def think(self, prompt, mode="deep"):
        return type('Thought', (), {
            'content': '{"project_name": "Test project", "tasks": [{"description": "Step 1", "priority": 10}, {"description": "Step 2", "priority": 5}]}',
            'confidence': 0.9,
            'expectation': 'success'
        })

@pytest.mark.asyncio
async def test_flow():
    db_path = "tests/test_projects.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    store = ProjectStore(db_path)
    brain = MockBrain()
    planner = StrategicPlanner(brain, store)
    
    # 1. Test Project Creation
    print("--- Testing Project Creation ---")
    project = await planner.analyze_and_plan("Build a Mars rover")
    assert project is not None
    assert project.name == "Test project"
    print(f"✓ Project created: {project.name}")
    
    # 2. Test Task Retrieval
    print("--- Testing Task Retrieval ---")
    task = planner.get_next_task(project.id)
    assert task is not None
    assert task.description == "Step 1"
    print(f"✓ Next task: {task.description}")
    
    # 3. Test Task Completion
    print("--- Testing Task Completion ---")
    planner.mark_task_complete(task.id, "Step 1 is done")
    task2 = planner.get_next_task(project.id)
    assert task2.description == "Step 2"
    print(f"✓ Task 1 completed, Next task: {task2.description}")
    
    # 4. Test Persistence
    print("--- Testing Persistence ---")
    new_store = ProjectStore(db_path)
    new_planner = StrategicPlanner(brain, new_store)
    projects = new_store.get_active_projects()
    assert len(projects) == 1
    assert projects[0].name == "Test project"
    print(f"✓ Persistence verified: Project recovered from DB")
    
    # 5. Test Status Report
    print("--- Testing Status Report ---")
    report = await planner.get_project_status_report(project.id)
    print(report)
    assert "Test project" in report
    assert "✅ Step 1" in report
    print("✓ Status report generated correctly")

    print("\n✅ STRATEGIC PLANNER VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    asyncio.run(test_flow())


##
