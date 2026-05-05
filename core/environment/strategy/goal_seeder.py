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

    # 4. Environment-Specific Objectives mapped to drives
    if "nethack" in environment_id or "terminal_grid" in environment_id:
        logger.info("Seeding NetHack-specific milestones into HTN.")
        
        # Domain milestones
        planner.add_milestone("found_stairs")
        planner.add_milestone("descended_level")
        planner.add_milestone("acquired_amulet")
        
        # Add domain sub-task to the knowledge drive
        nh_id = planner.add_subtask(
            parent_id=knowledge_id,
            name="solve_nethack_dungeon",
            priority=0.9,
        )
        planner.add_subtask(parent_id=nh_id, name="find_stairs", priority=0.9)
        planner.add_subtask(parent_id=nh_id, name="descend", priority=0.9)
        planner.add_subtask(parent_id=nh_id, name="locate_amulet", priority=0.9)
        
    elif "browser" in environment_id:
        logger.info("Seeding Browser-specific milestones into HTN.")
        planner.add_milestone("page_loaded")
        planner.add_milestone("target_element_found")
        
        br_id = planner.add_subtask(parent_id=knowledge_id, name="navigate_and_extract", priority=0.9)
        planner.add_subtask(parent_id=br_id, name="load_url")
        planner.add_subtask(parent_id=br_id, name="find_element")
        planner.add_subtask(parent_id=br_id, name="read_content")

    elif "os" in environment_id or "desktop" in environment_id:
        logger.info("Seeding OS-specific milestones into HTN.")
        os_id = planner.add_subtask(parent_id=knowledge_id, name="execute_user_request", priority=0.9)
        planner.add_subtask(parent_id=os_id, name="parse_request")
        planner.add_subtask(parent_id=os_id, name="locate_application")
        planner.add_subtask(parent_id=os_id, name="perform_interaction")

__all__ = ["seed_aura_goals"]
