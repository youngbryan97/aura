"""Default durable goals for overt Aura agency.

These are not marketing copy. They are the boot-seeded, tool-attached goals
that keep the initiative funnel from collapsing into "do nothing" when no user
task is currently active.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from core.runtime.errors import record_degradation


DEFAULT_AUTONOMY_GOALS = (
    {
        "name": "Maintain and repair Aura",
        "objective": "Continuously detect runtime failures, propose safe repairs, run targeted tests, and hold verified patches for promotion.",
        "priority": 0.92,
        "required_tools": ["shell", "self_repair", "test_generator"],
        "required_skills": ["self_repair", "auto_refactor", "self_evolution"],
        "success_criteria": "Every repair proposal carries failing evidence, patch lineage, targeted tests, and a Will receipt.",
    },
    {
        "name": "Keep proof bundle current",
        "objective": "Canonicalize proof artifacts, keep failures visible, compare against prompt-only and agent baselines, and refresh behavioral proof evidence.",
        "priority": 0.86,
        "required_tools": ["shell", "proof_bundle", "pytest"],
        "required_skills": ["coding", "self_improvement"],
        "success_criteria": "artifacts/proof_bundle/latest/MANIFEST.json and CANONICAL_PROOF_BUNDLE.json reflect the current runtime.",
    },
    {
        "name": "Ground substrate in live sensors",
        "objective": "Keep camera, screen, microphone, and sensory summaries coupled into the substrate input vector when governed capability tokens allow it.",
        "priority": 0.78,
        "required_tools": ["camera", "microphone", "screen"],
        "required_skills": ["computer_use", "listen"],
        "success_criteria": "The inner-state proof surface reports recent sensorimotor grounding observations.",
    },
    {
        "name": "Improve architecture through ASA",
        "objective": "Use the architecture governor on messy non-toy refactors, preserve tests, and keep docs aligned with the actual running architecture.",
        "priority": 0.74,
        "required_tools": ["shell", "filesystem", "pytest"],
        "required_skills": ["auto_refactor", "self_evolution"],
        "success_criteria": "Architecture changes include tests, receipts, docs, and rollback context.",
    },
)


async def seed_default_autonomy_goals(goal_engine: Any | None = None) -> list[dict[str, Any]]:
    """Seed durable IN_PROGRESS goals once per boot without duplicating them."""
    if os.getenv("AURA_SEED_DEFAULT_GOALS", "1").strip().lower() in {"0", "false", "off", "no"}:
        return []
    if goal_engine is None:
        from core.container import ServiceContainer

        goal_engine = ServiceContainer.get("goal_engine", default=None)
    if goal_engine is None or not hasattr(goal_engine, "add_goal"):
        return []

    seeded: list[dict[str, Any]] = []
    for spec in DEFAULT_AUTONOMY_GOALS:
        try:
            record = await goal_engine.add_goal(
                spec["name"],
                objective=spec["objective"],
                status="in_progress",
                horizon="long_term",
                source="boot_default",
                priority=spec["priority"],
                required_tools=spec["required_tools"],
                required_skills=spec["required_skills"],
                success_criteria=spec["success_criteria"],
                metadata={"boot_seeded": True, "default_goal": True},
            )
            seeded.append(record)
        except Exception as exc:
            record_degradation("default_goals", exc)
    return seeded


def seed_default_autonomy_goals_sync(goal_engine: Any | None = None) -> list[dict[str, Any]]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(seed_default_autonomy_goals(goal_engine))
    raise RuntimeError("seed_default_autonomy_goals_sync called inside a running event loop")


__all__ = ["DEFAULT_AUTONOMY_GOALS", "seed_default_autonomy_goals"]
