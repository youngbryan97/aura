"""Reasoning self-improvement — the internal STaR/RLVR bootstrap.

The honest mechanism for making a fixed local model genuinely better at reasoning,
with NO external teacher:

    1. The amplifier solves a hard problem and the local truth-engines verify it.
    2. That verifier-clean (problem -> answer) pair is captured as a training trace.
    3. When enough high-quality traces accumulate, they are fed into the EXISTING
       governed fine-tune pipe (Will-gated, scheduler-validated, promoted only if it
       measurably improves). The model learns to reach those answers in fewer steps.
    4. A stronger model solves harder problems -> more traces -> repeat.

This is self-play with a sound verifier (the lawful "bootstrap"): the loop's strength
equals the verifier's soundness, and it only bootstraps where verification is sound —
so we curate **only verifier-clean, source-independent** task types (math/code/logic),
never repo/architecture/factual where a captured "truth" could later be false.

Boundaries (deliberate, honest):
  * Capture is cheap and on by default. Training is NOT triggered here — traces are
    *fed* to the governed pipe, which validates and promotes (or rejects) under the
    existing scheduler. No unsupervised weight overwrite, no unbounded loop.
  * Bounded dataset (max traces, dedup by problem key). No infinite growth.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.brain.reasoning_solved_cache import (
    DEFAULT_CACHEABLE_TASK_TYPES,
    _problem_key,
)
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ReasoningSelfImprovement")

_DEFAULT_MAX_TRACES = 5000
_DEFAULT_MIN_CONFIDENCE = 0.7   # only train on high-confidence verifier-clean wins
_DEFAULT_MIN_TRACES_TO_TRAIN = 64


def _flag_on(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class ReasoningTrace:
    objective: str
    answer: str
    task_type: str
    confidence: float
    mode: str
    captured_at: float = field(default_factory=time.time)
    fed: bool = False  # already handed to the training pipe

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "answer": self.answer,
            "task_type": self.task_type,
            "confidence": round(float(self.confidence), 4),
            "mode": self.mode,
            "captured_at": self.captured_at,
            "fed": self.fed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningTrace":
        return cls(
            objective=str(data.get("objective", "")),
            answer=str(data.get("answer", "")),
            task_type=str(data.get("task_type", "")),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            mode=str(data.get("mode", "")),
            captured_at=float(data.get("captured_at", time.time()) or time.time()),
            fed=bool(data.get("fed", False)),
        )


def _feed_was_accepted(result: Any, *, expected: int) -> tuple[bool, str]:
    """Whether a training feed positively confirmed it took the traces.

    Marking a trace fed retires it permanently, so this requires a positive
    statement rather than the absence of an exception (CP126 2aaf46cd).
    A feed that returns nothing meaningful has not confirmed anything.
    """
    if result is None:
        return False, "feed returned None"
    if result is False:
        return False, "feed returned False"
    if isinstance(result, Mapping):
        if result.get("ok") is False or result.get("error"):
            return False, f"feed reported an error: {str(result.get('error'))[:120]}"
        accepted = result.get("accepted", result.get("count"))
        if isinstance(accepted, int) and not isinstance(accepted, bool):
            if accepted < expected:
                return False, f"feed accepted {accepted} of {expected}"
        return True, ""
    # A non-mapping, non-falsey result (including True) is taken as
    # acceptance: several governed pipes return only a truthy handle.
    return True, ""


class ReasoningSelfImprovement:
    """Collect verifier-clean reasoning wins; feed them to the governed train pipe."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError)

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_traces: int = _DEFAULT_MAX_TRACES,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        cacheable_task_types: frozenset[str] = DEFAULT_CACHEABLE_TASK_TYPES,
    ) -> None:
        self._path = Path(
            path or str(state_root() / "data/runtime/reasoning_traces.json")
        )
        self._max_traces = max(64, int(max_traces))
        self._min_confidence = float(min_confidence)
        self._cacheable = frozenset(cacheable_task_types)
        self._lock = threading.RLock()
        self._traces: dict[str, ReasoningTrace] = {}
        self._stats = {"captured": 0, "skipped": 0, "fed": 0, "evicted": 0}
        self._load()

    # ── capture ───────────────────────────────────────────────────────────
    def record_win(
        self,
        objective: str,
        task_type: str,
        *,
        answer: str,
        confidence: float,
        mode: str,
        verified: bool,
    ) -> bool:
        """Capture a verifier-clean, source-independent win as a training trace."""
        if not verified or not _flag_on("AURA_REASONING_SELF_IMPROVEMENT"):
            self._stats["skipped"] += 1
            return False
        tt = str(task_type or "").strip().lower()
        if tt not in self._cacheable:
            self._stats["skipped"] += 1
            return False
        clean = str(answer or "").strip()
        if not clean or float(confidence) < self._min_confidence:
            self._stats["skipped"] += 1
            return False
        if not self._domain_admitted(tt):
            # Verifier Foundry admission gate (frontier-general P1): a win may
            # only become TRAINING DATA when this domain's verification has
            # measured reliability (or a seed admission that evidence hasn't
            # revoked). Self-training on weakly-verified domains is how a
            # model amplifies its own garbage — the gate is the ceiling-mover.
            self._stats["skipped"] += 1
            self._stats["unadmitted"] = self._stats.get("unadmitted", 0) + 1
            return False
        key = _problem_key(objective, tt)
        with self._lock:
            self._traces[key] = ReasoningTrace(
                objective=str(objective or "").strip(),
                answer=clean,
                task_type=tt,
                confidence=float(confidence),
                mode=str(mode or ""),
            )
            self._stats["captured"] += 1
            self._evict_if_needed()
            self._persist()
        return True

    @staticmethod
    def _domain_admitted(task_type: str) -> bool:
        """Consult the Verifier Foundry and fail closed without its evidence.

        Self-improvement is a durable mutation path. Availability is not
        evidence of verifier reliability, so an absent or failed Foundry may
        delay learning but can never admit self-generated training data.
        """
        try:
            from core.runtime.service_access import optional_service

            foundry = optional_service("verifier_foundry", default=None)
            if foundry is None:
                record_degradation(
                    "reasoning_self_improvement",
                    RuntimeError("verifier_foundry_unavailable"),
                    severity="warning",
                    action="refused self-training admission without verifier reliability evidence",
                )
                return False
            return bool(foundry.domain_admitted(task_type).admitted)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "reasoning_self_improvement",
                exc,
                severity="warning",
                action="refused self-training admission after verifier reliability check failed",
            )
            return False

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._traces.values() if not t.fed)

    def export_training_examples(self) -> list[dict[str, str]]:
        """Curated (prompt, completion) pairs ready to feed a trainer."""
        with self._lock:
            return [
                {"prompt": t.objective, "completion": t.answer, "task_type": t.task_type}
                for t in self._traces.values()
                if not t.fed
            ]

    # ── feed the governed training pipe (no heavy op here) ──────────────────
    async def maybe_improve(
        self,
        *,
        min_traces: int = _DEFAULT_MIN_TRACES_TO_TRAIN,
        feed_fn: Callable[[list[dict[str, str]]], Any] | None = None,
    ) -> dict[str, Any]:
        """When enough verified traces accumulate, hand them to the governed pipe.

        This does NOT train directly — it feeds the existing Will-gated, scheduler-
        validated fine-tune path, which promotes only a measurably-better adapter.
        Returns a status dict; safe to call on a timer.
        """
        if not _flag_on("AURA_REASONING_SELF_IMPROVEMENT"):
            return {"status": "disabled"}
        pending = self.export_training_examples()
        if len(pending) < int(min_traces):
            return {"status": "insufficient_traces", "pending": len(pending), "need": int(min_traces)}

        # Never compete with an in-flight LoRA run.
        #
        # CP126 237ae4c2. This used to record a degradation and CONTINUE
        # feeding when the governor was missing or its check raised — so
        # governance infrastructure being absent or broken EXPANDED training
        # authority instead of restraining it. Not knowing whether a LoRA
        # run is active is not permission to start another; it is precisely
        # the state in which to wait.
        try:
            from core.adaptation.online_lora_governor import get_online_lora_governor

            if get_online_lora_governor().active_lora_processes():
                return {"status": "blocked_existing_training", "pending": len(pending)}
        except (ImportError, RuntimeError, AttributeError) as exc:
            record_degradation(
                "reasoning_self_improvement_governor",
                exc,
                severity="warning",
                action="refused to feed self-training while LoRA activity was unobservable",
            )
            return {
                "status": "blocked_governor_unavailable",
                "pending": len(pending),
                "error": f"{type(exc).__name__}: {exc}",
            }

        fed_fn = feed_fn or self._default_feed
        try:
            result = fed_fn(pending)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment]
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("reasoning_self_improvement_feed", exc)
            return {"status": "feed_failed", "error": f"{type(exc).__name__}: {exc}"}

        # CP126 2aaf46cd. Any result that did not RAISE marked every trace
        # fed and reported status "fed" — so False, None, an error envelope
        # or a partial count all retired the traces. A trace marked fed is
        # never offered again, so a feed that silently did nothing discarded
        # the work permanently while reporting success.
        accepted, reason = _feed_was_accepted(result, expected=len(pending))
        if not accepted:
            record_degradation(
                "reasoning_self_improvement_feed",
                RuntimeError(f"training feed not confirmed: {reason}"),
                severity="warning",
                action="left traces unfed after an unconfirmed training feed",
            )
            return {
                "status": "feed_unconfirmed",
                "reason": reason,
                "pending": len(pending),
                "feed_result": result,
            }

        with self._lock:
            for t in self._traces.values():
                t.fed = True
            self._stats["fed"] += len(pending)
            self._persist()
        logger.info("🧠 [SelfImprove] fed %d verifier-clean traces to the governed train pipe.", len(pending))
        return {"status": "fed", "count": len(pending), "feed_result": result}

    async def _default_feed(self, examples: list[dict[str, str]]) -> dict[str, Any]:
        """Feed traces into the existing governed fine-tune pipe (collect-only)."""
        from core.adaptation.finetune_pipe import get_finetune_pipe

        pipe = get_finetune_pipe()
        for ex in examples:
            await pipe.register_success(
                task_description=ex.get("prompt", "")[:800],
                context="",
                reasoning="verifier-clean reasoning win (STaR self-improvement)",
                final_action=ex.get("completion", "")[:2000],
                quality_score=0.85,
            )
        await pipe.flush()
        return {"ok": True, "fed": len(examples), "dataset_path": str(getattr(pipe, "dataset_path", ""))}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {**self._stats, "traces": len(self._traces), "pending": self.pending_count()}

    # ── internals ───────────────────────────────────────────────────────────
    def _evict_if_needed(self) -> None:
        # Caller holds lock. Prefer evicting already-fed, then oldest.
        while len(self._traces) > self._max_traces:
            fed_keys = [k for k, t in self._traces.items() if t.fed]
            victim = (
                min(fed_keys, key=lambda k: self._traces[k].captured_at)
                if fed_keys
                else min(self._traces.items(), key=lambda kv: kv[1].captured_at)[0]
            )
            self._traces.pop(victim, None)
            self._stats["evicted"] += 1

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            items = raw.get("traces", {}) if isinstance(raw, dict) else {}
            with self._lock:
                for key, data in items.items():
                    try:
                        self._traces[key] = ReasoningTrace.from_dict(data)
                    except self._ERRORS:
                        continue
                self._evict_if_needed()
        except self._ERRORS as exc:
            record_degradation("reasoning_self_improvement_load", exc)

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "saved_at": time.time(),
                "traces": {k: t.to_dict() for k, t in self._traces.items()},
            }
            fd, tmp = tempfile.mkstemp(prefix=".traces_", suffix=".json", dir=str(self._path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                os.replace(tmp, self._path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except self._ERRORS as exc:
            record_degradation("reasoning_self_improvement_persist", exc)


_singleton: ReasoningSelfImprovement | None = None
_singleton_lock = threading.Lock()


def get_reasoning_self_improvement() -> ReasoningSelfImprovement:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ReasoningSelfImprovement()
    return _singleton


def reset_reasoning_self_improvement() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None
