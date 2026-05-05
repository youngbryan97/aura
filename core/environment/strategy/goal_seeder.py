"""Goal initialization for Aura's HTN Planner.

Injects intrinsic drives based on Aura's personality and evolutionary goals,
then maps environment-specific milestones to those intrinsic drives.
"""
from __future__ import annotations

import logging
from .htn_planner import HTNPlanner

logger = logging.getLogger("Aura.HTN.GoalSeeder")


def seed_aura_goals(planner: HTNPlanner, environment_id: str) -> None:
    """Seeds the HTN planner with Aura's intrinsic drives and domain goals."""
    
    # 1. Base Root Goal (Aura's Prime Directive)
    root_id = planner.set_root_goal(name="achieve_homeostasis_and_evolve")

    # 2. Base Survival Drive (Priority 1.0 - Highest)
    survive_id = planner.add_subtask(
        parent_id=root_id,
        name="ensure_survival",
        priority=1.0,
    )

    # 3. Evolutionary / Knowledge Drive (Priority 0.8)
    knowledge_id = planner.add_subtask(
        parent_id=root_id,
        name="expand_world_knowledge",
        priority=0.8,
    )

    planner.add_subtask(parent_id=knowledge_id, name="explore_frontier", priority=0.8)
    planner.add_subtask(parent_id=knowledge_id, name="identify_unknown_objects", priority=0.8)
    planner.add_subtask(parent_id=knowledge_id, name="map_topology", priority=0.8)

    # 4. Capability-family objectives mapped to drives. These are intentionally
    # not task- or game-specific; adapters provide typed transitions, resources,
    # prompts, and terminal success/failure events.
    if "terminal_grid" in environment_id or "grid" in environment_id:
        logger.info("Seeding bounded-grid exploration milestones into HTN.")
        planner.add_milestone("transition_discovered")
        planner.add_milestone("context_changed")
        planner.add_milestone("primary_objective_discovered")

        grid_id = planner.add_subtask(
            parent_id=knowledge_id,
            name="solve_bounded_grid_environment",
            priority=0.9,
        )
        planner.add_subtask(parent_id=grid_id, name="discover_transition", priority=0.9)
        planner.add_subtask(parent_id=grid_id, name="change_context_when_safe", priority=0.9)
        planner.add_subtask(parent_id=grid_id, name="locate_primary_objective", priority=0.9)

    elif "browser" in environment_id:
        logger.info("Seeding document-navigation milestones into HTN.")
        planner.add_milestone("page_loaded")
        planner.add_milestone("target_element_found")
        
        br_id = planner.add_subtask(parent_id=knowledge_id, name="navigate_information_surface", priority=0.9)
        planner.add_subtask(parent_id=br_id, name="load_url")
        planner.add_subtask(parent_id=br_id, name="find_element")
        planner.add_subtask(parent_id=br_id, name="read_content")

    elif "os" in environment_id or "desktop" in environment_id:
        logger.info("Seeding desktop-interaction milestones into HTN.")
        os_id = planner.add_subtask(parent_id=knowledge_id, name="complete_external_tool_task", priority=0.9)
        planner.add_subtask(parent_id=os_id, name="parse_request")
        planner.add_subtask(parent_id=os_id, name="locate_application")
        planner.add_subtask(parent_id=os_id, name="perform_interaction")

__all__ = ["seed_aura_goals"]
