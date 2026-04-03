import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from core.data.project_store import ProjectStore, Project, StrategicTask
from core.container import ServiceContainer

logger = logging.getLogger("Aura.StrategicPlanner")

class StrategicPlanner:
    """The high-level mind that manages projects and multi-stage goals."""
    
    def __init__(self, cognitive_engine, project_store: ProjectStore):
        self.brain = cognitive_engine
        self.store = project_store
        self.active_project_id: Optional[str] = None
        
    async def analyze_and_plan(self, goal_text: str) -> Optional[Project]:
        """Analyze a mega-goal and create a persistent project with tasks."""
        logger.info("Decomposing mega-goal: %s", goal_text)
        
        # 1. Ask the brain for a decomposition
        prompt = f"""You are Aura's Strategic Planner. 
Analyze the following high-level goal and break it down into a logical sequence of 4-6 major tasks.
Each task should be a concrete step that contributes to the final goal.

HIGH-LEVEL GOAL: {goal_text}

Return the response as a valid JSON object with the following structure:
{{
  "project_name": "A short, descriptive name",
  "tasks": [
    {{ "description": "Step 1 description", "priority": 1 }},
    {{ "description": "Step 2 description", "priority": 2 }}
  ]
}}
"""
        thought = await self.brain.think(prompt, mode="SLOW")
        if not thought or not thought.content:
            logger.error("Failed to generate strategic plan")
            return None
            
        try:
            # Simple JSON extraction from LLM content
            content = thought.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            
            # Robust parsing (handle quotes and trailing commas)
            plan_data = json.loads(content)
            
            # 2. Create Project and tasks atomically
            with self.store.transaction() as conn:
                project = self.store.create_project(
                    name=plan_data.get("project_name", "New Project"),
                    goal=goal_text,
                    conn=conn
                )
                self.active_project_id = project.id
                
                # 3. Add Tasks with calculated priorities
                tasks = plan_data.get("tasks", [])
                for i, task_info in enumerate(tasks):
                    # Ensure priority is descending unless explicitly provided
                    base_priority = task_info.get("priority")
                    if base_priority is None:
                        base_priority = len(tasks) - i
                    
                    self.store.add_task(
                        project_id=project.id,
                        description=task_info.get("description", "Untitled Task"),
                        priority=base_priority,
                        metadata={"index": i, "strategic_intent": True},
                        conn=conn
                    )
                
            logger.info("🎯 Strategic Plan Accepted: '%s' (%d components)", project.name, len(tasks))
            
            # 4. Integrate with Neural Feed (Phase 17.1)
            feed = ServiceContainer.get("neural_feed", default=None)
            if feed:
                feed.push(f"STRATEGIC_PLAN: Decomposed mega-goal into project '{project.name}' with {len(tasks)} tasks.", 
                          category="STRATEGY")

            return project
            
        except Exception as e:
            logger.error("🚫 Strategic scaling failure: %s", e)
            return None

    def get_next_task(self, project_id: Optional[str] = None) -> Optional[StrategicTask]:
        """Retrieve the next pending task for the specified project (or first active)."""
        p_id = project_id or self.active_project_id
        if not p_id:
            active = self.store.get_active_projects()
            if not active:
                return None
            p_id = active[0].id
            
        tasks = self.store.get_tasks_for_project(p_id)
        for task in tasks:
            if task.status == "pending":
                return task
        return None

    def mark_task_complete(self, task_id: str, result_summary: str):
        """Record success for a task and move on."""
        self.store.update_task_status(task_id, "completed", {"result": result_summary})
        
    def mark_task_failed(self, task_id: str, error_msg: str):
        """Record failure for a task."""
        self.store.update_task_status(task_id, "failed", {"error": error_msg})

    async def replan_project(self, project_id: str, reason: str) -> bool:
        """Analyze a failure and redistribute remaining tasks."""
        project = self.store.get_project(project_id)
        if not project: return False
        
        logger.warning("🔄 Initiating Reflection Loop for project '%s': %s", project.name, reason)
        
        tasks = self.store.get_tasks_for_project(project_id)
        pending = [t for t in tasks if t.status == "pending"]
        completed = [t for t in tasks if t.status == "completed"]
        
        prompt = f"""You are Aura's Strategic Planner. A failure has occurred in project '{project.name}'.
GOAL: {project.goal}
FAILURE REASON: {reason}

COMPLETED TASKS:
{[t.description for t in completed]}

PENDING TASKS (to be revised):
{[t.description for t in pending]}

Analyze why the previous plan failed and provide a REVISED list of 3-5 tasks to complete the goal.
Return as JSON as before:
{{ "project_name": "{project.name}", "tasks": [...] }}
"""
        thought = await self.brain.think(prompt, mode="SLOW")
        if not thought or not thought.content:
            return False
            
        try:
            content = thought.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "{" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            
            plan_data = json.loads(content)
            
            # Wrapper for atomic updates
            with self.store.transaction() as conn:
                # 1. Archive previous pending tasks
                for t in pending:
                    self.store.update_task_status(t.id, "archived", {"reason": "Replanned due to failure"}, conn=conn)
                
                # 2. Add new tasks
                new_tasks = plan_data.get("tasks", [])
                for i, task_info in enumerate(new_tasks):
                    self.store.add_task(
                        project_id=project_id,
                        description=task_info.get("description"),
                        priority=len(new_tasks) - i + 10, # Give them higher priority than original
                        metadata={"replanned": True, "reason": reason},
                        conn=conn
                    )
            
            feed = ServiceContainer.get("neural_feed", default=None)
            if feed:
                feed.push(f"REFLECTION_LOOP: Replanned project '{project.name}' after failure: {reason}", category="STRATEGY")
            
            return True
        except Exception as e:
            logger.error("Failed to replan project: %s", e, exc_info=True)
            return False

    async def get_project_status_report(self, project_id: str) -> str:
        """Generate a human-readable status report for a project."""
        project = self.store.get_project(project_id)
        if not project:
            return "Project not found."
            
        tasks = self.store.get_tasks_for_project(project_id)
        completed = sum(1 for t in tasks if t.status == "completed")
        total = len(tasks)
        
        report = f"### Project: {project.name}\n"
        report += f"Goal: {project.goal}\n"
        report += f"Progress: {completed}/{total} tasks complete.\n\n"
        
        for task in tasks:
            status_icon = "✅" if task.status == "completed" else "⏳" if task.status == "pending" else "❌" if task.status == "failed" else "⚙️"
            report += f"- {status_icon} {task.description} ({task.status})\n"
            
        return report