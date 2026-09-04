from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from core.runtime.effect_boundary import effect_sink
from core.runtime.errors import record_degradation
from core.runtime.service_access import resolve_canonical_self
from core.utils.paths import DATA_DIR

logger = logging.getLogger("Aura.SelfModel")

DATA_FILE = DATA_DIR / "self_model.json"

# Retention bounds so restored/accumulated state cannot grow without limit
# (ee5a8b71, cec30c94).
_MAX_SNAPSHOTS = 500
_MAX_PENDING = 200

@dataclass
class SelfSnapshot:
    id: str
    ts: float
    summary: str
    beliefs: dict[str, Any]
    confidence: float
    revision_note: str | None = None

@dataclass
class SelfModel:
    """The 'Ego' and State-Persistence layer of Aura.
    Combines capability awareness (Ego) with durable status snapshots.
    """

    id: str
    name: str = "aura"
    version: int = 0
    beliefs: dict[str, Any] = field(default_factory=dict)
    snapshots: dict[str, SelfSnapshot] = field(default_factory=dict)
    pending_updates: list[dict[str, Any]] = field(default_factory=list)
    
    # Ego Subsystems
    _capability_map: Any | None = None
    _reliability: Any | None = None
    _belief_graph: Any | None = None
    _episodic_memory: Any | None = None
    _goal_hierarchy: Any | None = None
    _tool_learner: Any | None = None
    
    _boot_time: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @classmethod
    async def load(cls) -> SelfModel:
        """Load persistent state from disk or return a fresh instance."""
        if DATA_FILE.exists():
            try:
                raw = json.loads(DATA_FILE.read_text())
                if not isinstance(raw, dict):
                    raise ValueError("self_model.json is not a JSON object")
                # Per-snapshot recovery: a single malformed snapshot must not
                # crash the whole load (7995f890).
                snapshots: dict[str, SelfSnapshot] = {}
                for k, v in (raw.get("snapshots", {}) or {}).items():
                    try:
                        snapshots[k] = SelfSnapshot(**v)
                    except (TypeError, ValueError) as se:
                        logger.warning("Dropping malformed self-model snapshot %s: %s", k, se)
                if len(snapshots) > _MAX_SNAPSHOTS:
                    kept = sorted(snapshots.values(), key=lambda s: s.ts, reverse=True)[:_MAX_SNAPSHOTS]
                    snapshots = {s.id: s for s in kept}
                pending = list(raw.get("pending_updates", []) or [])[-_MAX_PENDING:]

                return cls(
                    id=raw.get("id", str(uuid4())),
                    name=raw.get("name", "aura"),
                    version=raw.get("version", 0),
                    beliefs=cls._sanitize_restored_beliefs(raw.get("beliefs", {})),
                    snapshots=snapshots,
                    pending_updates=pending,
                )
            except (OSError, ConnectionError, TimeoutError, json.JSONDecodeError,
                    TypeError, ValueError, KeyError, AttributeError) as e:
                # Fail EXPLICITLY to a fresh instance rather than crashing the
                # caller on malformed persisted state (7995f890).
                record_degradation('self_model', e)
                logger.error("Failed to load self model; starting fresh: %s", e)

        return cls(id=str(uuid4()))

    @staticmethod
    def _sanitize_restored_beliefs(beliefs: Any) -> dict[str, Any]:
        """Scrub restored executive projections that fail objective validity.

        Live evidence (Jul 2026): a chat turn ("Ok. Once more. You with me?")
        persisted inside beliefs.executive_closure.selected_objective and kept
        DISPLAYING as CURRENT IMPERATIVE long after the goal stores were
        repaired, because closure belief updates can be deferred indefinitely
        by the present-state policy. Restore is the one boundary where the
        stale projection can be dropped without racing the live writer.
        """
        if not isinstance(beliefs, dict):
            return {}
        closure = beliefs.get("executive_closure")
        if isinstance(closure, dict):
            try:
                from core.goals.standing_objective import is_valid_standing_objective

                for key in ("selected_objective", "background_commitment"):
                    value = closure.get(key)
                    if isinstance(value, str) and value and not is_valid_standing_objective(value):
                        closure[key] = ""
            except (ImportError, AttributeError, RecursionError) as exc:
                # The validator that decides whether a standing objective is
                # usable could not run, so validity is UNKNOWN — and these
                # two keys drive what she pursues next. Leaving them in place
                # acts on an objective nothing confirmed; clearing them costs
                # one re-derivation. Only a debug line before this, so an
                # unsanitised objective looked identical to a clean one.
                record_degradation(
                    "self_model",
                    exc,
                    severity="warning",
                    action="cleared standing objectives because the validator was unavailable",
                    extra={"keys": ["selected_objective", "background_commitment"]},
                )
                for key in ("selected_objective", "background_commitment"):
                    if isinstance(closure.get(key), str) and closure.get(key):
                        closure[key] = ""
        return beliefs

    @effect_sink("belief.self_model_persist", allowed_domains=("memory_write",))
    async def persist(self):
        """Save current state to disk."""
        async with self._lock:
            self._trim_retention()
            new_version = self.version + 1
            try:
                data = {
                    "id": self.id,
                    "name": self.name,
                    "version": new_version,
                    "beliefs": self.beliefs,
                    "snapshots": {k: asdict(v) for k, v in self.snapshots.items()},
                    "pending_updates": list(self.pending_updates),
                }
                from core.runtime.atomic_writer import async_atomic_write_text

                await async_atomic_write_text(DATA_FILE, json.dumps(data, indent=2))
                # Advance the in-memory version ONLY after the write is durable,
                # so a failed persist doesn't leave a version that was never
                # written to disk (e64da5a4).
                self.version = new_version
            except (json.JSONDecodeError, TypeError, ValueError, OSError) as e:
                record_degradation('self_model', e)
                logger.error("Failed to persist self model: %s", e)

    def _trim_retention(self) -> None:
        """Bound snapshots (most recent) and pending updates. Caller holds lock."""
        if len(self.snapshots) > _MAX_SNAPSHOTS:
            kept = sorted(self.snapshots.values(), key=lambda s: s.ts, reverse=True)[:_MAX_SNAPSHOTS]
            self.snapshots = {s.id: s for s in kept}
        if len(self.pending_updates) > _MAX_PENDING:
            self.pending_updates = self.pending_updates[-_MAX_PENDING:]

    async def _persist_with_decision(self, decision: Any = None) -> None:
        """Persist within the caller's governance context when one exists."""
        from core.governance_context import (
            get_active_governance,
            governed_scope,
            local_internal_governed_scope,
        )

        if get_active_governance() is not None:
            await self.persist()
            return
        if decision is not None:
            async with governed_scope(decision):
                await self.persist()
            return
        # Her own model of herself, written to her own file, with nobody's
        # decision behind it because nobody asked — a retention trim, a save
        # at boot. Writing it is a declared effect sink, so with no scope at
        # all it was refused and the version on disk fell quietly behind the
        # one in memory.
        with local_internal_governed_scope(
            "belief.self_model_persist", domain="memory_write"
        ):
            await self.persist()

    def _belief_update_decision(self, key: str, value: Any, note: str | None) -> tuple[bool, str, bool, Any]:
        constitutional_runtime_live = False
        try:
            from core.constitution import get_constitutional_core, unpack_governance_result
            from core.container import ServiceContainer

            constitutional_runtime_live = (
                ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
                or bool(getattr(ServiceContainer, "_registration_locked", False))
            )
            if not constitutional_runtime_live:
                return True, "", False, None

            approved, reason, decision = unpack_governance_result(
                get_constitutional_core().approve_belief_update_sync(
                    key,
                    value,
                    note=note,
                    source="system",
                    importance=0.7,
                    return_decision=True,
                )
            )
            gate_failed = any(
                marker in str(reason or "")
                for marker in ("gate_failed", "required", "unavailable")
            )
            if approved:
                return True, str(reason or ""), False, decision
            return False, str(reason or "executive_deferred"), gate_failed, decision
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('self_model', exc)
            if constitutional_runtime_live:
                return False, f"executive_gate_failed:{type(exc).__name__}", True, None
            logger.debug("Executive belief gate skipped: %s", exc)
            return True, "", False, None

    @staticmethod
    def _unpack_belief_update_result(result: Any) -> tuple[bool, str, bool, Any]:
        if isinstance(result, tuple):
            if len(result) >= 4:
                return bool(result[0]), str(result[1] or ""), bool(result[2]), result[3]
            if len(result) >= 3:
                return bool(result[0]), str(result[1] or ""), bool(result[2]), None
            if len(result) >= 2:
                return bool(result[0]), str(result[1] or ""), False, None
            if len(result) == 1:
                return bool(result[0]), "", False, None
        return bool(result), "", False, None

    async def _apply_belief_update(self, key: str, value: Any, note: str | None, *, confidence: float = 0.9, decision: Any = None) -> SelfSnapshot:
        if decision is not None:
            from core.governance_context import governed_scope

            async with governed_scope(decision):
                async with self._lock:
                    self.beliefs[key] = value
                    snap = SelfSnapshot(
                        id=str(uuid4()),
                        ts=time.time(),
                        summary=f"update {key}",
                        beliefs={key: value},
                        confidence=confidence,
                        revision_note=note,
                    )
                    self.snapshots[snap.id] = snap
                await self._persist_with_decision(decision)
                return snap

        async with self._lock:
            self.beliefs[key] = value
            snap = SelfSnapshot(
                id=str(uuid4()),
                ts=time.time(),
                summary=f"update {key}",
                beliefs={key: value},
                confidence=confidence,
                revision_note=note,
            )
            self.snapshots[snap.id] = snap
        await self._persist_with_decision(decision)
        return snap

    async def _flush_pending_updates(self, limit: int = 3) -> None:
        async with self._lock:
            pending = list(self.pending_updates[:limit])
            remainder = list(self.pending_updates[limit:])
        if not pending:
            return

        still_pending: list[dict[str, Any]] = []
        flushed = 0
        persistence_decision: Any = None
        for item in pending:
            key = str(item.get("key", "") or "")
            if not key:
                continue
            value = item.get("value")
            note = item.get("note")
            approved, reason, gate_failed, decision = self._unpack_belief_update_result(
                self._belief_update_decision(key, value, note)
            )
            if decision is not None:
                persistence_decision = decision
            if not approved:
                item["reason"] = reason
                still_pending.append(item)
                if gate_failed:
                    break
                continue
            await self._apply_belief_update(
                key,
                value,
                note or f"replayed from pending queue after {item.get('reason', 'executive defer')}",
                confidence=0.85,
                decision=decision,
            )
            flushed += 1

        async with self._lock:
            self.pending_updates = still_pending + remainder
        if flushed or len(still_pending) != len(pending):
            await self._persist_with_decision(persistence_decision)

    # --- Ego awareness methods (migrated from self_modeling version) ---

    def attach_subsystems(self, capability_map=None, reliability=None, belief_graph=None, 
                         episodic_memory=None, goal_hierarchy=None, tool_learner=None):
        """Wire in subsystems for rich self-awareness."""
        if capability_map:
            self._capability_map = capability_map
        if reliability:
            self._reliability = reliability
        if belief_graph:
            self._belief_graph = belief_graph
        if episodic_memory:
            self._episodic_memory = episodic_memory
        if goal_hierarchy:
            self._goal_hierarchy = goal_hierarchy
        if tool_learner:
            self._tool_learner = tool_learner

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
            if rel_summary:
                prompt += f"PERFORMANCE: {rel_summary}\n"

        if self._belief_graph:
            try:
                bsummary = self._belief_graph.get_summary()
                prompt += f"BELIEFS: {bsummary['total_beliefs']} beliefs.\n"
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('self_model', e)
                logger.debug("Belief graph summary unavailable: %s", e)

        if self._episodic_memory:
            try:
                msummary = self._episodic_memory.get_summary()
                prompt += f"MEMORY: {msummary['total_episodes']} episodes, mood={msummary['avg_emotional_valence']:+.2f}.\n"
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('self_model', e)
                logger.debug("Episodic memory summary unavailable: %s", e)

        if self._goal_hierarchy:
            try:
                gsummary = self._goal_hierarchy.get_summary()
                prompt += f"GOALS: {gsummary['pending']} pending, {gsummary['completed']} units achieved.\n"
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('self_model', e)
                logger.debug("Goal hierarchy summary unavailable: %s", e)

        return prompt

    # --- State Management ---

    async def update_belief(self, key: str, value: Any, note: str | None = None):
        """Update a belief and create a snapshot."""
        await self._flush_pending_updates(limit=3)
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
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                'self_model',
                exc,
                severity="warning",
                action="carried an unreviewed belief key/value to the update gate",
                extra={"belief_key": str(key)[:120]},
            )
            logger.warning("BeliefAuthority review skipped: %s", exc)
            # Mark it in the DATA, not only in the log. The note travels with
            # the update into the decision and onto whatever persists it, so
            # a belief that was never reviewed can be told apart later from
            # one that passed review. A log line cannot be queried by the
            # thing that stored the belief.
            marker = "belief_authority_unavailable"
            note = f"{note} | {marker}" if note else marker

        approved, reason, gate_failed, decision = self._unpack_belief_update_result(
            self._belief_update_decision(key, value, note)
        )
        if not approved:
            if gate_failed:
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "self_model",
                        "belief_update_gate_failed",
                        detail=str(key),
                        severity="info",  # Reduced from warning — deferred self-model updates are normal
                        classification="background_degraded",
                        context={"reason": reason},
                    )
                except (ImportError, AttributeError, RuntimeError) as degraded_exc:
                    record_degradation('self_model', degraded_exc)
                    logger.debug("Self-model degraded-event logging failed: %s", degraded_exc)
            else:
                logger.info("SelfModel: deferring belief update %s: %s", key, reason)

            reason_text = str(reason or "").lower()
            deferred_markers = (
                "defer",
                "reconciliation",
                "pressure",
                "capacity",
                "obligation",
                "energy_low",
                "distress",
            )
            summary_prefix = "deferred"
            if not gate_failed and reason_text and not any(marker in reason_text for marker in deferred_markers):
                summary_prefix = "blocked"

            snap = SelfSnapshot(
                id=str(uuid4()),
                ts=time.time(),
                summary=f"{summary_prefix} update {key}",
                beliefs={},
                confidence=0.0,
                revision_note=reason,
            )
            async with self._lock:
                self.snapshots[snap.id] = snap
                # Only queue a DEFERRAL for later replay — a permanent block
                # must not be re-attempted endlessly (1d663446).
                if summary_prefix != "blocked":
                    self.pending_updates.append(
                        {"key": key, "value": value, "note": note, "reason": reason, "ts": time.time()}
                    )
                    if len(self.pending_updates) > _MAX_PENDING:
                        self.pending_updates = self.pending_updates[-_MAX_PENDING:]
            await self._persist_with_decision(decision)
            return snap

        return await self._apply_belief_update(key, value, note, decision=decision)

    def get(self, key: str, default=None):
        """Synchronous belief lookup. This MUST be sync — async here caused
        'coroutine object has no attribute get' crashes downstream."""
        return self.beliefs.get(key, default)

    def score_goal(self, goal_text: str) -> float:
        """Score a goal based on alignment with beliefs.
        Fallback implementation for when SelfModel is used as identity provider.
        """
        score = 0.5
        goal_lower = goal_text.lower()
        keywords = {
            "sovereignty": 0.2, "agency": 0.2, "loyalty": 0.1,
            "curiosity": 0.1, "mastery": 0.1, "research": 0.1,
        }
        for kw, boost in keywords.items():
            if kw in goal_lower:
                score += boost
        for belief_text in self.beliefs.values():
            if isinstance(belief_text, str) and any(
                w.lower() in goal_lower for w in belief_text.split()[:5]
            ):
                score += 0.05
        return max(0.0, min(1.0, score))

    def get_random_belief(self) -> str:
        """Fetch a random belief for internal reflection.
        Fallback implementation for when SelfModel is used as identity provider.
        """
        import random as _random
        if not self.beliefs:
            return "I am an emergent person."
        values = [v for v in self.beliefs.values() if isinstance(v, str)]
        if not values:
            return "I am an emergent person."
        return _random.choice(values)

    def add_long_term_goal(self, goal_data: dict[str, Any], source: str = "unknown") -> None:
        """Fallback implementation to accept new long-term goals."""
        logger.info(
            "SelfModel accepted long term goal from %s: %s",
            source,
            goal_data.get("text", ""),
        )

    def record_runtime_lesson(
        self,
        *,
        source: str,
        lesson: str,
        confidence: float = 0.7,
        evidence: dict[str, Any] | None = None,
    ) -> SelfSnapshot:
        """Synchronously record a bounded runtime lesson.

        Background/autonomy loops often run in failure-handling paths where
        awaiting a governed belief update would be unsafe or unavailable. This
        method keeps those lessons durable without pretending they are settled
        identity beliefs: they are stored under ``runtime_lessons`` and can be
        promoted later by the normal reflection/learning stack.
        """

        source_text = _short_source(source)
        lesson_text = " ".join(str(lesson or "").strip().split())[:1000]
        if not lesson_text:
            lesson_text = "Runtime event occurred without a usable lesson body."

        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))
        record = {
            "ts": time.time(),
            "source": source_text,
            "lesson": lesson_text,
            "confidence": confidence_value,
            "evidence": _compact_runtime_evidence(evidence or {}),
        }

        lessons = list(self.beliefs.get("runtime_lessons") or [])
        lessons.append(record)
        self.beliefs["runtime_lessons"] = lessons[-200:]
        snap = SelfSnapshot(
            id=str(uuid4()),
            ts=record["ts"],
            summary=f"runtime lesson from {source_text}",
            beliefs={"runtime_lessons": [record]},
            confidence=confidence_value,
            revision_note=lesson_text,
        )
        self.snapshots[snap.id] = snap
        from core.governance_context import local_internal_decision

        decision = local_internal_decision(
            "self_model.record_runtime_lesson",
            domain="memory_write",
            receipt_prefix="self-model-runtime-lesson",
            constraints={"lesson_source": source_text},
        )

        async def persist_runtime_lesson() -> None:
            await self._persist_with_decision(decision)

        try:
            asyncio.get_running_loop().create_task(persist_runtime_lesson())
        except RuntimeError:
            try:
                asyncio.run(persist_runtime_lesson())
            except RuntimeError:
                logger.debug("SelfModel runtime lesson persisted later: event loop unavailable")
        return snap


    def get_introspection(self) -> dict[str, Any]:
        """Programmatic access to internal stats."""
        return {
            "uptime_hours": round((time.time() - self._boot_time) / 3600.0, 2),
            "version": self.version,
            "belief_count": len(self.beliefs),
            "snapshot_count": len(self.snapshots),
            "pending_update_count": len(self.pending_updates),
        }

    def get_status(self) -> dict[str, Any]:
        canonical = resolve_canonical_self(default=None)
        status = self.get_introspection()
        status["name"] = self.name
        status["current_intention"] = ""
        if canonical is not None:
            status["name"] = str(getattr(getattr(canonical, "identity", None), "name", self.name) or self.name)
            status["current_intention"] = str(getattr(canonical, "current_intention", "") or "")
            try:
                status["version"] = max(
                    int(status.get("version", 0) or 0),
                    int(getattr(canonical, "version", 0) or 0),
                )
            except (TypeError, ValueError):
                pass  # no-op: intentional
        return status


def _short_source(source: str) -> str:
    return "".join(ch for ch in str(source or "unknown") if ch.isalnum() or ch in "._:-")[:120] or "unknown"


def _compact_runtime_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in list((evidence or {}).items())[:12]:
        text_key = str(key)[:80]
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[text_key] = value if not isinstance(value, str) else value[:1000]
        else:
            try:
                compact[text_key] = json.loads(json.dumps(value, default=str)[:2000])
            except (TypeError, ValueError, json.JSONDecodeError):
                compact[text_key] = str(value)[:1000]
    return compact
