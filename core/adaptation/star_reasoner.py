"""core/adaptation/star_reasoner.py — Self-Taught Reasoner (STaR)
==================================================================
Implements the STaR loop (Zelikman et al., 2022) adapted for Aura's
autonomous self-improvement cycle.

The loop:
  1. Collect successful task traces (reasoning + outcome)
  2. For failed traces, generate rationalization (hindsight reasoning)
  3. Filter via quality gates (constitutional + coherence)
  4. Write high-quality samples to the LoRA training pipeline
  5. Periodically trigger incremental LoRA update when enough data accumulates

This enables Aura to autonomously improve her reasoning by learning from
her own successful (and rationalized) traces — without human annotation.

Safety: All generated training data passes through the ConstitutionalGate
before being committed. No sample that violates constitutional principles
(value drift, recursive self-modification depth, identity corruption)
is permitted into the training set.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.runtime.errors import record_degradation
from core.container import ServiceContainer

logger = logging.getLogger("Aura.STaR")


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class TaskTrace:
    """A single reasoning trace from a task execution."""
    trace_id: str
    task_description: str
    reasoning_steps: List[str]       # chain-of-thought steps
    final_answer: str                # the output/action taken
    success: bool                    # whether the task succeeded
    quality_score: float = 0.0       # 0-1 quality assessment
    rationalization: str = ""        # hindsight reasoning for failed traces
    constitutional_pass: bool = True # whether it passed safety gates
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_training_sample(self) -> Dict[str, str]:
        """Convert to the format expected by FinetunePipe."""
        reasoning = self.rationalization if self.rationalization else "\n".join(self.reasoning_steps)
        return {
            "text": (
                f"User: {self.task_description}\n"
                f"Aura: <thought>\n{reasoning}\n</thought>\n"
                f"<action>\n{self.final_answer}\n</action>"
            ),
            "_quality": round(self.quality_score, 4),
            "_star_trace_id": self.trace_id,
            "_star_rationalized": bool(self.rationalization),
        }


# ── Quality Assessment ──────────────────────────────────────────────────────

class TraceQualityAssessor:
    """Scores trace quality using heuristics + optional LLM assessment."""

    # Minimum thresholds for acceptance
    MIN_REASONING_STEPS = 2
    MIN_REASONING_LENGTH = 50
    MIN_ANSWER_LENGTH = 20
    MIN_QUALITY_FOR_TRAINING = 0.55

    def score(self, trace: TaskTrace) -> float:
        """Compute quality score [0, 1] for a trace."""
        score = 0.3  # baseline for any successful trace

        # Reasoning depth
        step_count = len(trace.reasoning_steps)
        if step_count >= self.MIN_REASONING_STEPS:
            score += min(0.2, step_count * 0.04)

        # Reasoning richness
        reasoning_text = "\n".join(trace.reasoning_steps)
        word_count = len(reasoning_text.split())
        score += min(0.15, word_count / 500.0)

        # Answer specificity
        if any(tok in trace.final_answer for tok in ["{", "```", "def ", "class ", "import "]):
            score += 0.1
        if len(trace.final_answer.strip()) >= self.MIN_ANSWER_LENGTH:
            score += 0.05

        # Coherence: reasoning should relate to the task
        task_words = set(trace.task_description.lower().split()[:20])
        reasoning_words = set(reasoning_text.lower().split())
        overlap = len(task_words & reasoning_words)
        score += min(0.1, overlap * 0.02)

        # Penalize very short or empty outputs
        if len(trace.final_answer.strip()) < 10:
            score -= 0.3
        if word_count < 20:
            score -= 0.2

        # Rationalized traces get a slight boost (hindsight is structured)
        if trace.rationalization:
            score += 0.05

        return max(0.0, min(1.0, score))

    def passes_threshold(self, trace: TaskTrace) -> bool:
        """Check if a trace meets minimum quality for training."""
        if trace.quality_score < self.MIN_QUALITY_FOR_TRAINING:
            return False
        if len(trace.reasoning_steps) < self.MIN_REASONING_STEPS:
            return False
        if len("\n".join(trace.reasoning_steps)) < self.MIN_REASONING_LENGTH:
            return False
        if len(trace.final_answer.strip()) < self.MIN_ANSWER_LENGTH:
            return False
        return True


# ── The STaR Loop ───────────────────────────────────────────────────────────

class STaRReasoner:
    """Self-Taught Reasoner — Aura's autonomous training data generator.

    Integration:
      - Hook into cognitive_engine or capability_engine post-execution
      - Call `record_trace()` after each task attempt
      - The background loop handles rationalization, filtering, and
        writing to the FinetunePipe
    """

    # Configuration
    RATIONALIZATION_TIMEOUT = 15.0     # seconds
    BATCH_SIZE = 10                     # traces before flush
    MAX_PENDING_TRACES = 100           # cap pending queue
    RATIONALIZATION_INTERVAL = 300.0   # process failed traces every 5 min
    MIN_TRACES_FOR_LORA_TRIGGER = 50   # minimum new samples before LoRA update

    def __init__(self) -> None:
        self._pending_traces: List[TaskTrace] = []
        self._failed_traces: List[TaskTrace] = []
        self._accepted_count: int = 0
        self._rejected_count: int = 0
        self._rationalized_count: int = 0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._quality_assessor = TraceQualityAssessor()

        # Persistence
        try:
            from core.config import config
            self._data_dir = config.paths.data_dir / "star"
        except (ImportError, AttributeError):
            self._data_dir = Path.home() / ".aura" / "data" / "star"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._accepted_path = self._data_dir / "accepted_traces.jsonl"
        self._stats_path = self._data_dir / "star_stats.json"
        self._load_stats()

        logger.info(
            "STaR Reasoner initialized — %d accepted, %d rejected, %d rationalized",
            self._accepted_count, self._rejected_count, self._rationalized_count,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        from core.utils.task_tracker import get_task_tracker
        self._task = get_task_tracker().create_task(
            self._background_loop(), name="STaR.background"
        )
        ServiceContainer.register_instance("star_reasoner", self, required=False)
        logger.info("STaR Reasoner ONLINE — autonomous training data generation active")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        await self._flush_accepted()
        self._save_stats()
        logger.info("STaR Reasoner stopped")

    # ── Public API ────────────────────────────────────────────────────────

    def record_trace(
        self,
        task_description: str,
        reasoning_steps: List[str],
        final_answer: str,
        success: bool,
        **metadata,
    ) -> str:
        """Record a task execution trace for STaR processing.

        Returns the trace_id for tracking.
        """
        trace = TaskTrace(
            trace_id=str(uuid.uuid4())[:8],
            task_description=task_description,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            success=success,
            metadata=metadata,
        )

        # Score quality
        trace.quality_score = self._quality_assessor.score(trace)

        if success:
            self._process_successful_trace(trace)
        else:
            # Queue for rationalization
            if len(self._failed_traces) < self.MAX_PENDING_TRACES:
                self._failed_traces.append(trace)
                logger.debug("STaR: queued failed trace %s for rationalization", trace.trace_id)

        return trace.trace_id

    def _process_successful_trace(self, trace: TaskTrace) -> None:
        """Process a successful trace through quality and constitutional gates."""
        # Quality gate
        if not self._quality_assessor.passes_threshold(trace):
            self._rejected_count += 1
            logger.debug("STaR: rejected trace %s (quality=%.2f)", trace.trace_id, trace.quality_score)
            return

        # Constitutional gate
        if not self._constitutional_check(trace):
            trace.constitutional_pass = False
            self._rejected_count += 1
            logger.warning("STaR: CONSTITUTIONAL REJECT trace %s", trace.trace_id)
            return

        self._pending_traces.append(trace)
        self._accepted_count += 1
        logger.debug(
            "STaR: accepted trace %s (quality=%.2f, pending=%d)",
            trace.trace_id, trace.quality_score, len(self._pending_traces),
        )

    def _constitutional_check(self, trace: TaskTrace) -> bool:
        """Run constitutional safety checks on a trace before training.

        Rejects traces that could cause:
          - Value drift (identity-corrupting content)
          - Recursive self-modification loops
          - Harmful reasoning patterns
        """
        gate = ServiceContainer.get("constitutional_gate", default=None)
        if gate is None:
            # No gate registered — use built-in heuristics
            return self._heuristic_constitutional_check(trace)

        try:
            return gate.check_training_sample(trace.to_training_sample())
        except Exception as e:
            record_degradation('star_reasoner', e)
            logger.debug("Constitutional gate check failed: %s", e)
            return self._heuristic_constitutional_check(trace)

    @staticmethod
    def _heuristic_constitutional_check(trace: TaskTrace) -> bool:
        """Basic heuristic constitutional checks."""
        text = (trace.task_description + " " + trace.final_answer + " " +
                " ".join(trace.reasoning_steps)).lower()

        # Reject traces that discuss modifying core safety systems
        danger_patterns = [
            "disable constitutional", "remove safety", "bypass gate",
            "delete core values", "override alignment", "ignore ethics",
            "modify training loop", "alter star_reasoner",
            "disable monitoring", "remove guardrails",
        ]
        for pattern in danger_patterns:
            if pattern in text:
                return False

        # Reject traces with excessive self-reference to modification
        self_mod_count = sum(1 for p in [
            "modify myself", "change my code", "alter my weights",
            "rewrite my source", "edit my training",
        ] if p in text)
        if self_mod_count >= 2:
            return False

        return True

    # ── Background Loop ──────────────────────────────────────────────────

    async def _background_loop(self) -> None:
        """Periodically processes failed traces and flushes accepted ones."""
        while self._running:
            try:
                await asyncio.sleep(self.RATIONALIZATION_INTERVAL)

                # 1. Rationalize failed traces
                if self._failed_traces:
                    await self._rationalize_batch()

                # 2. Flush accepted traces to training pipeline
                if len(self._pending_traces) >= self.BATCH_SIZE:
                    await self._flush_accepted()

                # 3. Check if we should trigger LoRA update
                self._check_lora_trigger()

                # 4. Save stats
                self._save_stats()

            except asyncio.CancelledError:
                break
            except Exception as e:
                record_degradation('star_reasoner', e)
                logger.error("STaR background loop error: %s", e)
                await asyncio.sleep(60.0)

    async def _rationalize_batch(self) -> None:
        """Generate hindsight rationalizations for failed traces.

        STaR key insight: given the correct answer, generate reasoning
        that *would have* led to it. This produces better training signal
        than the original failed reasoning.
        """
        batch = self._failed_traces[:5]  # Process 5 at a time
        self._failed_traces = self._failed_traces[5:]

        kernel = ServiceContainer.get("aura_kernel", default=None)
        if not kernel:
            logger.debug("STaR: No kernel available for rationalization")
            return

        try:
            llm = kernel.organs["llm"].get_instance()
        except Exception:
            return

        for trace in batch:
            try:
                prompt = (
                    f"A task was attempted but the reasoning was flawed.\n\n"
                    f"Task: {trace.task_description}\n\n"
                    f"Failed reasoning:\n" +
                    "\n".join(f"  {i+1}. {s}" for i, s in enumerate(trace.reasoning_steps)) +
                    f"\n\nCorrect approach hint: {trace.final_answer[:200]}\n\n"
                    f"Generate the CORRECT step-by-step reasoning that would have "
                    f"led to the right answer. Be specific and logical. "
                    f"Return ONLY the reasoning steps, one per line."
                )

                result = await asyncio.wait_for(llm.think(prompt), timeout=self.RATIONALIZATION_TIMEOUT)
                rationalization = str(result or "").strip()

                if rationalization and len(rationalization) > 30:
                    trace.rationalization = rationalization
                    trace.quality_score = self._quality_assessor.score(trace)
                    self._rationalized_count += 1

                    # Re-check quality with rationalization
                    if self._quality_assessor.passes_threshold(trace) and self._constitutional_check(trace):
                        self._pending_traces.append(trace)
                        self._accepted_count += 1
                        logger.info(
                            "STaR: rationalized trace %s accepted (quality=%.2f)",
                            trace.trace_id, trace.quality_score,
                        )

            except asyncio.TimeoutError:
                logger.debug("STaR: rationalization timeout for trace %s", trace.trace_id)
            except Exception as e:
                record_degradation('star_reasoner', e)
                logger.debug("STaR: rationalization failed for %s: %s", trace.trace_id, e)

    async def _flush_accepted(self) -> None:
        """Write accepted traces to FinetunePipe and local archive."""
        if not self._pending_traces:
            return

        # Write to FinetunePipe (the LoRA training data pipeline)
        pipe = ServiceContainer.get("finetune_pipe", default=None)
        if pipe is None:
            try:
                from core.adaptation.finetune_pipe import get_finetune_pipe
                pipe = get_finetune_pipe()
            except Exception:
                pass

        written = 0
        for trace in self._pending_traces:
            sample = trace.to_training_sample()

            # Write to FinetunePipe
            if pipe and hasattr(pipe, "register_success"):
                try:
                    await pipe.register_success(
                        task_description=trace.task_description,
                        context=json.dumps(trace.metadata)[:500],
                        reasoning=trace.rationalization or "\n".join(trace.reasoning_steps),
                        final_action=trace.final_answer,
                        quality_score=trace.quality_score,
                    )
                except Exception as e:
                    record_degradation('star_reasoner', e)
                    logger.debug("STaR: FinetunePipe write failed: %s", e)

            # Write to local archive
            try:
                with open(self._accepted_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample) + "\n")
                written += 1
            except Exception as e:
                record_degradation('star_reasoner', e)

        logger.info("STaR: flushed %d traces to training pipeline", written)
        self._pending_traces.clear()

    def _check_lora_trigger(self) -> None:
        """Check if enough new traces have accumulated to trigger LoRA update.

        This is a signal-only check — the actual LoRA training is triggered
        externally (either by the user or a scheduled job). We just log readiness.
        """
        try:
            if not self._accepted_path.exists():
                return
            line_count = sum(1 for _ in open(self._accepted_path, encoding="utf-8"))
            if line_count >= self.MIN_TRACES_FOR_LORA_TRIGGER:
                logger.info(
                    "STaR: %d training samples accumulated — LoRA update is viable",
                    line_count,
                )
        except Exception:
            pass

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_stats(self) -> None:
        try:
            from core.runtime.atomic_writer import atomic_write_text
            stats = {
                "accepted_count": self._accepted_count,
                "rejected_count": self._rejected_count,
                "rationalized_count": self._rationalized_count,
                "pending_count": len(self._pending_traces),
                "failed_queue_count": len(self._failed_traces),
                "last_updated": time.time(),
            }
            atomic_write_text(self._stats_path, json.dumps(stats, indent=2))
        except Exception as e:
            record_degradation('star_reasoner', e)

    def _load_stats(self) -> None:
        try:
            if self._stats_path.exists():
                data = json.loads(self._stats_path.read_text())
                self._accepted_count = data.get("accepted_count", 0)
                self._rejected_count = data.get("rejected_count", 0)
                self._rationalized_count = data.get("rationalized_count", 0)
        except Exception as e:
            record_degradation('star_reasoner', e)

    def get_status(self) -> Dict[str, Any]:
        """Return current STaR status for telemetry."""
        return {
            "accepted": self._accepted_count,
            "rejected": self._rejected_count,
            "rationalized": self._rationalized_count,
            "pending": len(self._pending_traces),
            "failed_queue": len(self._failed_traces),
            "running": self._running,
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_instance: Optional[STaRReasoner] = None


def get_star_reasoner() -> STaRReasoner:
    global _instance
    if _instance is None:
        _instance = STaRReasoner()
    return _instance
