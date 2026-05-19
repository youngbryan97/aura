"""Sleep / Dream Skill — Semantic Memory Consolidation.

Consolidates episodic memories from the day into semantic knowledge,
evolves the identity's "evolved context" layer, and prunes stale data.

Implements an 'Idea Immune System' — the base identity (core directives)
is immutable; only the evolved context layer can change.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import Severity, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Sleep")

_SLEEP_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    OSError,
    ConnectionError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_sleep_degradation(
    error: BaseException,
    *,
    phase: str,
    action: str,
    severity: Severity = "warning",
) -> None:
    record_degradation(
        "sleep",
        error,
        severity=severity,
        action=action,
        extra={"phase": phase},
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _set_phase(phases: dict[str, dict[str, Any]], phase: str, status: str, **fields: Any) -> None:
    payload = {"status": status}
    payload.update({key: value for key, value in fields.items() if value is not None})
    phases[phase] = payload


def _error_summary(error: BaseException) -> str:
    return f"{type(error).__qualname__}: {error}"[:240]


def _heuristic_knowledge(mem_text: str, *, limit: int = 5) -> str:
    """Deterministic fallback consolidation when the reflective LLM path is unavailable."""
    facts: list[str] = []
    seen: set[str] = set()
    for raw_line in mem_text.splitlines():
        line = " ".join(str(raw_line).strip().split())
        if ":" in line:
            role, content = line.split(":", 1)
            if role.lower() in {"user", "assistant", "system", "thought"}:
                line = content.strip()
        if len(line) < 12:
            continue
        key = line.lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        facts.append(line[:220])
        if len(facts) >= limit:
            break
    if not facts:
        return ""
    return "\n".join(f"- {fact}" for fact in facts)


def _get_sleep_service(
    name: str,
    *,
    phases: dict[str, dict[str, Any]],
    phase: str,
    action: str,
    severity: Severity = "warning",
) -> Any | None:
    try:
        return ServiceContainer.get(name, default=None)
    except _SLEEP_RECOVERABLE_ERRORS as exc:
        _record_sleep_degradation(exc, phase=phase, action=action, severity=severity)
        _set_phase(phases, phase, "failed", error=_error_summary(exc), action=action)
        return None


class SleepSkill(BaseSkill):
    name = "dream_sleep"
    description = "Consolidates memories and evolves identity during downtime."
    timeout_seconds = 120.0
    metabolic_cost = 3

    async def execute(self, params: Any = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
        logger.info("Aura is entering REM sleep (Neural Consolidation)...")
        context = context or {}
        steps_completed: list[str] = []
        phases: dict[str, dict[str, Any]] = {}

        # ── 1. Gather recent memories ────────────────────────────────
        mem_text = ""
        memory = _get_sleep_service(
            "memory_facade",
            phases=phases,
            phase="memory_recall",
            action="Skipped memory recall and will try conversation-history fallback",
            severity="warning",
        )
        if memory and hasattr(memory, "recall"):
            try:
                recent = await _maybe_await(memory.recall("Today's important lessons", limit=20))
                if recent:
                    mem_text = "\n".join(str(m) for m in recent)
                    steps_completed.append("memory_recall")
                    _set_phase(phases, "memory_recall", "completed", items=len(recent))
                else:
                    _set_phase(phases, "memory_recall", "skipped", reason="no_recent_memories")
            except _SLEEP_RECOVERABLE_ERRORS as exc:
                action = "Memory recall failed; falling back to conversation history for consolidation seed"
                _record_sleep_degradation(exc, phase="memory_recall", action=action, severity="warning")
                _set_phase(phases, "memory_recall", "failed", error=_error_summary(exc), action=action)
        elif "memory_recall" not in phases:
            _set_phase(phases, "memory_recall", "skipped", reason="memory_facade_unavailable")

        if not mem_text:
            orch = _get_sleep_service(
                "orchestrator",
                phases=phases,
                phase="conversation_fallback",
                action="Conversation-history fallback unavailable; sleep may rest without consolidation",
                severity="debug",
            )
            if orch and hasattr(orch, "conversation_history") and orch.conversation_history:
                try:
                    recent_turns = orch.conversation_history[-30:]
                    mem_text = "\n".join(
                        f"{m.get('role', '?')}: {str(m.get('content', ''))[:200]}"
                        for m in recent_turns
                        if isinstance(m, dict)
                    )
                    if mem_text:
                        steps_completed.append("conversation_fallback")
                        _set_phase(
                            phases,
                            "conversation_fallback",
                            "completed",
                            turns=len(recent_turns),
                        )
                except _SLEEP_RECOVERABLE_ERRORS as exc:
                    action = "Conversation fallback failed; sleep will rest without consolidation seed"
                    _record_sleep_degradation(exc, phase="conversation_fallback", action=action, severity="warning")
                    _set_phase(phases, "conversation_fallback", "failed", error=_error_summary(exc), action=action)
            elif "conversation_fallback" not in phases:
                _set_phase(phases, "conversation_fallback", "skipped", reason="no_conversation_history")

        if not mem_text:
            return {
                "ok": True,
                "summary": "No memories to consolidate — Aura rested quietly.",
                "steps": steps_completed,
                "phases": phases,
                "degraded_steps": [
                    phase for phase, payload in phases.items() if payload.get("status") == "failed"
                ],
            }

        # ── 2. Dream: Extract semantic facts via LLM, with deterministic fallback ──
        derived_knowledge = ""
        brain = _get_sleep_service(
            "cognitive_engine",
            phases=phases,
            phase="dream_synthesis",
            action="Reflective brain unavailable; using deterministic heuristic consolidation",
            severity="debug",
        )
        if brain and hasattr(brain, "think"):
            try:
                from core.brain.types import ThinkingMode

                dream_prompt = (
                    f"NEW EXPERIENCES:\n{mem_text[:3000]}\n\n"
                    "TASK: Consolidate these episodic experiences into distinct semantic facts or lessons. "
                    "Focus on: what was learned, what changed, what matters for tomorrow. "
                    "Output a bulleted list of 'Derived Knowledge'."
                )
                result = await _maybe_await(brain.think(dream_prompt, mode=ThinkingMode.REFLECTIVE))
                derived_knowledge = str(getattr(result, "content", result) or "").strip()
                if derived_knowledge:
                    steps_completed.append("dream_synthesis")
                    _set_phase(phases, "dream_synthesis", "completed", method="reflective_llm")
                    logger.info("Dream synthesis complete: knowledge extracted.")
            except _SLEEP_RECOVERABLE_ERRORS as exc:
                action = "Reflective dream synthesis failed; using deterministic heuristic consolidation"
                _record_sleep_degradation(exc, phase="dream_synthesis", action=action, severity="warning")
                _set_phase(phases, "dream_synthesis", "failed", error=_error_summary(exc), action=action)
        elif "dream_synthesis" not in phases:
            _set_phase(phases, "dream_synthesis", "skipped", reason="cognitive_engine_unavailable")

        if not derived_knowledge:
            derived_knowledge = _heuristic_knowledge(mem_text)
            if derived_knowledge:
                steps_completed.append("heuristic_synthesis")
                _set_phase(
                    phases,
                    "heuristic_synthesis",
                    "completed",
                    method="deterministic_memory_summary",
                )
            else:
                _set_phase(phases, "heuristic_synthesis", "skipped", reason="no_extractable_memory_lines")

        # ── 3. Dream journal entry ───────────────────────────────────
        dream_journal = _get_sleep_service(
            "dream_journal",
            phases=phases,
            phase="dream_journal",
            action="Skipped dream journal write because service lookup failed",
            severity="warning",
        )
        if dream_journal and hasattr(dream_journal, "record"):
            try:
                await _maybe_await(
                    dream_journal.record(
                        content=derived_knowledge or mem_text[:500],
                        dream_type="consolidation",
                    )
                )
                steps_completed.append("dream_journal")
                _set_phase(phases, "dream_journal", "completed")
            except _SLEEP_RECOVERABLE_ERRORS as exc:
                action = "Dream journal write failed; preserved derived knowledge in skill result"
                _record_sleep_degradation(exc, phase="dream_journal", action=action, severity="warning")
                _set_phase(phases, "dream_journal", "failed", error=_error_summary(exc), action=action)
        elif "dream_journal" not in phases:
            _set_phase(phases, "dream_journal", "skipped", reason="dream_journal_unavailable")

        # ── 4. Identity evolution (immune system) ────────────────────
        identity_evolved = False
        if derived_knowledge:
            canonical_engine = _get_sleep_service(
                "canonical_self_engine",
                phases=phases,
                phase="identity_evolution",
                action="Skipped identity evolution because canonical self engine lookup failed",
                severity="warning",
            )
            if canonical_engine and hasattr(canonical_engine, "evolve_from_dream"):
                try:
                    identity_evolved = bool(
                        await _maybe_await(canonical_engine.evolve_from_dream(derived_knowledge))
                    )
                    if identity_evolved:
                        steps_completed.append("identity_evolution")
                    _set_phase(phases, "identity_evolution", "completed", evolved=identity_evolved)
                except _SLEEP_RECOVERABLE_ERRORS as exc:
                    action = "Identity evolution failed; kept immutable base identity and retained derived knowledge"
                    _record_sleep_degradation(exc, phase="identity_evolution", action=action, severity="warning")
                    _set_phase(phases, "identity_evolution", "failed", error=_error_summary(exc), action=action)
            elif "identity_evolution" not in phases:
                _set_phase(phases, "identity_evolution", "skipped", reason="canonical_self_engine_unavailable")
        else:
            _set_phase(phases, "identity_evolution", "skipped", reason="no_derived_knowledge")

        # ── 5. Memory compaction ─────────────────────────────────────
        compressor = _get_sleep_service(
            "knowledge_compression",
            phases=phases,
            phase="memory_compaction",
            action="Skipped memory compaction because compressor lookup failed",
            severity="warning",
        )
        if compressor and hasattr(compressor, "compact"):
            try:
                await _maybe_await(compressor.compact())
                steps_completed.append("memory_compaction")
                _set_phase(phases, "memory_compaction", "completed")
            except _SLEEP_RECOVERABLE_ERRORS as exc:
                action = "Memory compaction failed; kept consolidated knowledge result and continued restoration"
                _record_sleep_degradation(exc, phase="memory_compaction", action=action, severity="warning")
                _set_phase(phases, "memory_compaction", "failed", error=_error_summary(exc), action=action)
        elif "memory_compaction" not in phases:
            _set_phase(phases, "memory_compaction", "skipped", reason="knowledge_compression_unavailable")

        # ── 6. Satisfy drives (rest restores energy) ─────────────────
        drive = _get_sleep_service(
            "drive_engine",
            phases=phases,
            phase="drive_restoration",
            action="Skipped drive restoration because drive engine lookup failed",
            severity="warning",
        )
        if drive and hasattr(drive, "satisfy"):
            restored: list[str] = []
            failed: list[str] = []
            for drive_name, amount in (("energy", 30.0), ("competence", 5.0)):
                try:
                    await _maybe_await(drive.satisfy(drive_name, amount))
                    restored.append(drive_name)
                except _SLEEP_RECOVERABLE_ERRORS as exc:
                    failed.append(drive_name)
                    action = f"Drive restoration for {drive_name} failed; continued remaining restoration lanes"
                    _record_sleep_degradation(exc, phase="drive_restoration", action=action, severity="warning")
            if restored:
                steps_completed.append("drive_restoration")
            status = "completed" if restored and not failed else "degraded" if restored else "failed"
            _set_phase(phases, "drive_restoration", status, restored=restored, failed=failed)
        elif "drive_restoration" not in phases:
            _set_phase(phases, "drive_restoration", "skipped", reason="drive_engine_unavailable")

        # ── 7. Record in WorldState ──────────────────────────────────
        try:
            from core.world_state import get_world_state

            ws = get_world_state()
            ws.record_event(
                f"Dream cycle completed: {len(steps_completed)} steps",
                source="sleep_skill",
                salience=0.3,
                ttl=28800,  # 8 hours
            )
            _set_phase(phases, "world_state", "completed")
        except _SLEEP_RECOVERABLE_ERRORS as exc:
            action = "World-state sleep event write failed; returning phase receipts in skill result"
            _record_sleep_degradation(exc, phase="world_state", action=action, severity="debug")
            _set_phase(phases, "world_state", "failed", error=_error_summary(exc), action=action)

        degraded_steps = [
            phase for phase, payload in phases.items() if payload.get("status") in {"failed", "degraded"}
        ]
        summary_parts = [f"Dream cycle completed ({len(steps_completed)} steps)."]
        if degraded_steps:
            summary_parts.append(f"Degraded phases: {', '.join(degraded_steps[:4])}.")
        if derived_knowledge:
            summary_parts.append(f"Learned: {derived_knowledge[:200]}")
        if identity_evolved:
            summary_parts.append("Identity evolved from dream insights.")

        return {
            "ok": True,
            "summary": " ".join(summary_parts),
            "steps": steps_completed,
            "phases": phases,
            "degraded_steps": degraded_steps,
            "knowledge": derived_knowledge[:500] if derived_knowledge else "",
            "identity_evolved": identity_evolved,
        }
