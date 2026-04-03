from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("Aura.SelfModel")

from core.common.paths import DATA_DIR
DATA_FILE = DATA_DIR / "self_model.json"

@dataclass
class SelfSnapshot:
    id: str
    ts: float
    summary: str
    beliefs: Dict[str, Any]
    confidence: float
    revision_note: Optional[str] = None

@dataclass
class SelfModel:
    """The 'Ego' and State-Persistence layer of Aura.
    Combines capability awareness (Ego) with durable status snapshots.
    """

    id: str
    name: str = "aura"
    version: int = 0
    beliefs: Dict[str, Any] = field(default_factory=dict)
    snapshots: Dict[str, SelfSnapshot] = field(default_factory=dict)
    
    # Ego Subsystems
    _capability_map: Optional[Any] = None
    _reliability: Optional[Any] = None
    _belief_graph: Optional[Any] = None
    _episodic_memory: Optional[Any] = None
    _goal_hierarchy: Optional[Any] = None
    _tool_learner: Optional[Any] = None
    
    _boot_time: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @classmethod
    async def load(cls) -> "SelfModel":
        """Load persistent state from disk or return a fresh instance."""
        if DATA_FILE.exists():
            try:
                raw = json.loads(DATA_FILE.read_text())
                snaps_raw = raw.get("snapshots", {})
                snapshots = {k: SelfSnapshot(**v) for k, v in snaps_raw.items()}
                
                return cls(
                    id=raw.get("id", str(uuid4())),
                    name=raw.get("name", "aura"),
                    version=raw.get("version", 0),
                    beliefs=raw.get("beliefs", {}),
                    snapshots=snapshots
                )
            except Exception as e:
                logger.error("Failed to load self model: %s", e)
                
        return cls(id=str(uuid4()))

    async def persist(self):
        """Save current state to disk."""
        async with self._lock:
            self.version += 1
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                data = {
                    "id": self.id,
                    "name": self.name,
                    "version": self.version,
                    "beliefs": self.beliefs,
                    "snapshots": {k: asdict(v) for k, v in self.snapshots.items()}
                }
                DATA_FILE.write_text(json.dumps(data, indent=2))
            except Exception as e:
                logger.error("Failed to persist self model: %s", e)

    # --- Ego awareness methods (migrated from self_modeling version) ---

    def attach_subsystems(self, capability_map=None, reliability=None, belief_graph=None, 
                         episodic_memory=None, goal_hierarchy=None, tool_learner=None):
        """Wire in subsystems for rich self-awareness."""
        if capability_map: self._capability_map = capability_map
        if reliability: self._reliability = reliability
        if belief_graph: self._belief_graph = belief_graph
        if episodic_memory: self._episodic_memory = episodic_memory
        if goal_hierarchy: self._goal_hierarchy = goal_hierarchy
        if tool_learner: self._tool_learner = tool_learner

    def get_self_awareness_prompt(self) -> str:
        """Generate the system prompt section describing 'Who I am and what I can do'."""
        uptime_hours = (time.time() - self._boot_time) / 3600.0
        prompt = "\n\n## SELF-AWARENESS (Ego)\n"
        prompt += f"Uptime: {uptime_hours:.1f} hours.\n"

        if self._capability_map:
            status = self._capability_map.get_status()
            prompt += f"Active capabilities: {status['online']}/{status['total_capabilities']}.\n"

        if self._reliability:
            rel_summary = self._reliability.get_capabilities_summary()
            if rel_summary: prompt += f"PERFORMANCE: {rel_summary}\n"

        if self._belief_graph:
            try:
                bsummary = self._belief_graph.get_summary()
                prompt += f"BELIEFS: {bsummary['total_beliefs']} beliefs.\n"
            except Exception as e:
                logger.debug("Belief graph summary unavailable: %s", e)

        if self._episodic_memory:
            try:
                msummary = self._episodic_memory.get_summary()
                prompt += f"MEMORY: {msummary['total_episodes']} episodes, mood={msummary['avg_emotional_valence']:+.2f}.\n"
            except Exception as e:
                logger.debug("Episodic memory summary unavailable: %s", e)

        if self._goal_hierarchy:
            try:
                gsummary = self._goal_hierarchy.get_summary()
                prompt += f"GOALS: {gsummary['pending']} pending, {gsummary['completed']} units achieved.\n"
            except Exception as e:
                logger.debug("Goal hierarchy summary unavailable: %s", e)

        return prompt

    # --- State Management ---

    async def update_belief(self, key: str, value: Any, note: Optional[str] = None):
        """Update a belief and create a snapshot."""
        try:
            from core.constitution import get_constitutional_core

            reviewed = get_constitutional_core().belief_authority.review_update(
                "self_model",
                key,
                value,
                note=note,
            )
            key = reviewed.key
            value = reviewed.value
            if note:
                note = f"{note} | {reviewed.reason}"
            else:
                note = reviewed.reason
        except Exception as exc:
            logger.debug("BeliefAuthority review skipped: %s", exc)

        constitutional_runtime_live = False
        try:
            from core.container import ServiceContainer
            from core.executive.executive_core import (
                ActionType,
                DecisionOutcome,
                Intent,
                IntentSource,
                get_executive_core,
            )

            constitutional_runtime_live = (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
            if constitutional_runtime_live:
                intent = Intent(
                    source=IntentSource.SYSTEM,
                    goal=f"self_model_belief:{key}",
                    action_type=ActionType.UPDATE_BELIEF,
                    payload={"key": key, "value": value, "note": note},
                    priority=0.7,
                    requires_memory_commit=True,
                )
                record = get_executive_core().request_approval_sync(intent)
                if record.outcome not in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "self_model",
                            "belief_update_blocked",
                            detail=str(key),
                            severity="warning",
                            classification="background_degraded",
                            context={"reason": record.reason},
                        )
                    except Exception as degraded_exc:
                        logger.debug("Self-model degraded-event logging failed: %s", degraded_exc)

                    snap = SelfSnapshot(
                        id=str(uuid4()),
                        ts=time.time(),
                        summary=f"blocked update {key}",
                        beliefs={},
                        confidence=0.0,
                        revision_note=record.reason,
                    )
                    async with self._lock:
                        self.snapshots[snap.id] = snap
                    return snap
        except Exception as exc:
            if constitutional_runtime_live:
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "self_model",
                        "belief_update_gate_failed",
                        detail=str(key),
                        severity="warning",
                        classification="background_degraded",
                        context={"error": type(exc).__name__},
                        exc=exc,
                    )
                except Exception as degraded_exc:
                    logger.debug("Self-model degraded-event logging failed: %s", degraded_exc)

                snap = SelfSnapshot(
                    id=str(uuid4()),
                    ts=time.time(),
                    summary=f"blocked update {key}",
                    beliefs={},
                    confidence=0.0,
                    revision_note="executive_gate_failed",
                )
                async with self._lock:
                    self.snapshots[snap.id] = snap
                return snap
            logger.debug("Executive belief gate skipped: %s", exc)

        async with self._lock:
            self.beliefs[key] = value
            snap = SelfSnapshot(
                id=str(uuid4()),
                ts=time.time(),
                summary=f"update {key}",
                beliefs={key: value},
                confidence=0.9,
                revision_note=note
            )
            self.snapshots[snap.id] = snap
        await self.persist()
        return snap

    async def get(self, key: str, default=None):
        return self.beliefs.get(key, default)

    def get_introspection(self) -> Dict[str, Any]:
        """Programmatic access to internal stats."""
        return {
            "uptime_hours": round((time.time() - self._boot_time) / 3600.0, 2),
            "version": self.version,
            "belief_count": len(self.beliefs),
            "snapshot_count": len(self.snapshots)
        }
