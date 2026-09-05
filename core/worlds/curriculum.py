"""core/worlds/curriculum.py
─────────────────────────
Embodied practice: generated tasks with objective scores.

Simulation-heavy training needs three things the wishlist named:
reproducible task environments, an agent that attempts them, and honest
measurement. A PracticeTask is generated from a seed (deterministic
world + objective), executed by the embodied agent's actual competence
(A* navigation, grasping, carrying), and scored on outcome + efficiency.
Results append to a governed practice ledger so learning systems can
read performance trends instead of anecdotes.

Task kinds:
- navigate: reach a target point across generated terrain.
- fetch: reach an object, grasp it, carry it to a drop zone.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get_config
from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.worlds.embodied import EmbodiedAgent
from core.worlds.generation import generate_world
from core.worlds.physics import Body, PhysicsError

logger = logging.getLogger("Aura.Worlds.Curriculum")

TASK_KINDS = ("navigate", "fetch")
_LEDGER_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


@dataclass
class PracticeTask:
    kind: str
    seed: int
    world_size: int
    theme: str
    target: tuple[float, float]
    fetch_object: str | None = None
    drop_zone: tuple[float, float] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "world_size": self.world_size,
            "theme": self.theme,
            "target": list(self.target),
            "fetch_object": self.fetch_object,
            "drop_zone": list(self.drop_zone) if self.drop_zone else None,
        }


def generate_task(seed: int, kind: str = "navigate", *, size: int = 20) -> PracticeTask:
    """Deterministic task from a seed: same seed → same world, same goal."""
    if kind not in TASK_KINDS:
        raise PhysicsError(f"task kind must be one of {TASK_KINDS}")
    rng = np.random.default_rng(seed ^ 0x5EED)
    reach = size / 2.0 - 2.0
    target = (float(rng.uniform(-reach, reach)), float(rng.uniform(-reach, reach)))
    if kind == "navigate":
        return PracticeTask(kind, seed, size, "plains", target)
    drop = (float(rng.uniform(-reach, reach)), float(rng.uniform(-reach, reach)))
    return PracticeTask(kind, seed, size, "plains", target,
                        fetch_object="practice_ball", drop_zone=drop)


def run_task(task: PracticeTask, *, max_ticks: int = 24000) -> dict[str, Any]:
    """Execute the task with the embodied agent's real competence and
    return an objective score. No LLM in the loop; physics decides."""
    blueprint = generate_world(task.seed, size=task.world_size, theme=task.theme)
    world = blueprint.to_physics_world()
    if task.kind == "fetch":
        surface_clearance = 0.35
        world.add_body(Body(
            body_id=task.fetch_object, shape="sphere",
            position=(task.target[0], task.target[1], surface_clearance + 3.0),
            velocity=(0.0, 0.0, 0.0), mass=1.5, radius=0.3,
            restitution=0.2, friction=0.6, rolling_resistance=0.03,
        ))
        world.step(600)  # let the object land and settle before the run
    agent = EmbodiedAgent.spawn(world, blueprint)
    started = world.tick

    if task.kind == "navigate":
        outcome = agent.navigate_to(task.target, tolerance=1.2, max_ticks=max_ticks)
        success = outcome["status"] == "reached"
        detail: dict[str, Any] = {"navigation": outcome}
    else:
        leg_budget = max_ticks // 2
        approach = agent.navigate_to(task.target, tolerance=1.4, max_ticks=leg_budget)
        grabbed = False
        delivered = False
        detail = {"approach": approach}
        if approach["status"] == "reached":
            grabbed = agent.grasp(task.fetch_object)
            detail["grasped"] = grabbed
            if grabbed:
                delivery = agent.navigate_to(
                    task.drop_zone, tolerance=1.4, max_ticks=leg_budget)
                detail["delivery"] = delivery
                if delivery["status"] == "reached":
                    agent.throw(speed=1.0, pitch=0.1)
                    dropped = world.body(task.fetch_object)
                    distance = float(np.linalg.norm(
                        dropped.position[:2] - np.array(task.drop_zone)))
                    detail["drop_distance"] = round(distance, 3)
                    delivered = distance <= 2.5
        success = delivered

    ticks_used = world.tick - started
    # Efficiency: straight-line distance over ticks actually spent walking.
    straight = float(np.linalg.norm(np.array(task.target) - np.array(
        blueprint.spawn_point[:2])))
    efficiency = min(1.0, (straight / 3.0 * 120.0) / max(1, ticks_used)) if success else 0.0
    score = round((0.7 if success else 0.0) + 0.3 * efficiency, 4)
    return {
        "task": task.describe(),
        "success": success,
        "score": score,
        "ticks_used": ticks_used,
        "efficiency": round(efficiency, 4),
        "detail": detail,
        "at": time.time(),
    }


# ── practice ledger ──────────────────────────────────────────────

def ledger_path() -> Path:
    return Path(get_config().paths.data_dir) / "worlds" / "practice_ledger.jsonl"


async def record_practice(result: dict[str, Any]) -> None:
    try:
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "worlds.curriculum",
            domain="file_write",
            receipt_prefix="world-practice",
        ):
            await gateway.ensure_directory_async(
                ledger_path().parent, source="worlds.curriculum")
            await gateway.append_text_async(
                ledger_path(),
                json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
                source="worlds.curriculum",
            )
    except _LEDGER_ERRORS as exc:
        record_degradation("worlds.curriculum.ledger", exc)
        logger.error("Failed to record practice result: %s", exc)
    _feed_learning_loop(result)


def _feed_learning_loop(result: dict[str, Any]) -> None:
    """Embodied practice is trainable signal, not just measurement: each
    scored attempt flows into the PracticeDirector so the compounding
    learning loop can weigh embodied domains against everything else."""
    try:
        from core.learning.deliberate_practice import get_practice_director

        kind = str(result.get("task", {}).get("kind", "unknown"))
        get_practice_director().observe(
            domain=f"embodied.{kind}",
            attempts=1,
            correct=1 if result.get("success") else 0,
            source="world_curriculum",
            receipt=f"practice_ledger:{result.get('at')}",
        )
    except _LEDGER_ERRORS + (ImportError, AttributeError) as exc:
        record_degradation("worlds.curriculum.learning_feed", exc)


def practice_summary(limit: int = 200) -> dict[str, Any]:
    """Read the ledger tail and report the honest trend."""
    path = ledger_path()
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-max(1, limit):]:
                rows.append(json.loads(line))
        except _LEDGER_ERRORS as exc:
            record_degradation("worlds.curriculum.summary", exc)
    if not rows:
        return {"attempts": 0, "success_rate": None, "mean_score": None}
    successes = sum(1 for row in rows if row.get("success"))
    half = len(rows) // 2
    recent = rows[half:]
    return {
        "attempts": len(rows),
        "success_rate": round(successes / len(rows), 4),
        "mean_score": round(sum(r.get("score", 0.0) for r in rows) / len(rows), 4),
        "recent_success_rate": round(
            sum(1 for r in recent if r.get("success")) / max(1, len(recent)), 4),
        "by_kind": {
            kind: sum(1 for r in rows if r.get("task", {}).get("kind") == kind)
            for kind in TASK_KINDS
        },
    }
