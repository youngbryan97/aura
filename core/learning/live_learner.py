from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.learning.cortex_generation_upgrade import record_upgrade_candidate
from core.memory.retention_policy import training_buffer_retention_policy
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.disk_budget import (
    DiskBudgetRefusal,
    directory_bytes,
    ensure_headroom_for,
)
from core.runtime.errors import FallbackClassification, record_degradation
from core.tasks.managed_command import run_project_command

logger = logging.getLogger("Aura.LiveLearner")
_LIVE_LEARNER_RECOVERABLE_ERRORS = (
    AttributeError,
    FileNotFoundError,
    ImportError,
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_TRAINING_CONTAMINATION_PATTERNS = {
    "[silent auto-fix]": "silent_autofix_prompt",
    "handle this silently": "silent_repair_prompt",
    "traceback (most recent call last)": "traceback_telemetry",
    "task exception was never retrieved": "asyncio_exception_telemetry",
    "kernelinterface chat timed out": "chat_timeout_telemetry",
    "response generation failed": "response_generation_failure",
    "languagecenter expression failed": "language_center_failure",
    "unmapped critical traceback": "unmapped_traceback_prompt",
    "fix a data access error": "repair_prompt",
    "investigate a timeout": "repair_prompt",
    "diagnose unmapped critical traceback": "repair_prompt",
    "fix a missing module/import issue": "repair_prompt",
    "as an ai language model": "assistant_identity_regression",
    "aura language model": "assistant_identity_regression",
    "how can i assist": "assistant_canned_reply",
    "great question": "assistant_canned_reply",
    "certainly!": "assistant_canned_reply",
    "both have their merits": "assistant_canned_reply",
    "i encountered a cognitive error during response generation": "canned_failure_reply",
    "i'm having trouble formulating a response": "canned_failure_reply",
    "my thinking engine just hiccupped": "canned_failure_reply",
    "i'm still initializing": "canned_failure_reply",
    "i'm processing that, but i haven't reached a verbal conclusion yet": "canned_failure_reply",
    "i'm turning that over. give me a moment": "canned_failure_reply",
    "i'm reaching for an answer that feels honest": "canned_failure_reply",
    "eli pariser coined the phrase": "web_leakage",
    "learn why people trust wikihow": "web_leakage",
    "download article": "web_leakage",
}


def _record_live_learning_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


# ── Quality scoring ──────────────────────────────────────────────────────────

@dataclass
class InteractionScore:
    """Quality assessment for a single interaction."""
    interaction_id:   str
    raw_score:        float            # 0.0 to 1.0
    affect_weight:    float            # High-affect moments matter more
    final_score:      float            # raw_score * affect_weight modifier
    reasons_positive: list[str] = field(default_factory=list)
    reasons_negative: list[str] = field(default_factory=list)
    worth_training:   bool = False


def score_interaction(
    user_input: str,
    response: str,
    affect_valence: float = 0.0,
    affect_curiosity: float = 0.5,
    phi: float = 0.0,
    follow_up_detected: bool = False,
    confusion_detected: bool = False,
) -> InteractionScore:
    """
    Multi-signal quality scoring for a single interaction.

    Positive signals:
      +0.30  User followed up (implicit approval)
      +0.20  Response length appropriate (30-400 words)
      +0.15  No identity regression patterns
      +0.15  High phi during generation (integrated state)
      +0.10  High curiosity (engagement signal)
      +0.10  Positive valence

    Negative signals:
      -0.50  User expressed confusion ("what?", "??", "that's wrong")
      -0.25  Response too short (<5 words) or empty
      -0.20  Truncated (ends without punctuation)
      -0.20  Identity violation detected
    """
    interaction_id = hashlib.sha256(
        f"{time.time()}{user_input[:20]}".encode()
    ).hexdigest()[:16]

    score = 0.5
    pos, neg = [], []

    # Length check
    words = len(response.split()) if response else 0
    if 30 <= words <= 400:
        score += 0.20
        pos.append("appropriate_length")
    elif words < 5:
        score -= 0.25
        neg.append("too_short")

    # Identity check
    banned = [
        "as an ai",
        "certainly!",
        "absolutely!",
        "great question",
        "how can i help",
        "language model",
        "i was trained",
    ]
    if not any(b in response.lower() for b in banned):
        score += 0.15
        pos.append("identity_intact")
    else:
        score -= 0.20
        neg.append("identity_regression")

    # Truncation
    if response and response.strip()[-1] not in ".!?\"'\n":
        score -= 0.20
        neg.append("truncated")

    # Behavioral signals
    if follow_up_detected:
        score += 0.30
        pos.append("user_follow_up")
    if confusion_detected:
        score -= 0.50
        neg.append("user_confusion")

    # Affect signals
    if phi > 0.4:
        score += 0.15
        pos.append("high_phi")
    if affect_curiosity > 0.6:
        score += 0.10
        pos.append("high_curiosity")
    if affect_valence > 0.3:
        score += 0.10
        pos.append("positive_valence")

    score = max(0.0, min(1.0, score))

    # Affect weight: high affect moments are more important training signal
    affect_magnitude = (abs(affect_valence) + affect_curiosity + phi) / 3.0
    affect_weight = 0.5 + (affect_magnitude * 0.5)

    final_score = score * affect_weight

    return InteractionScore(
        interaction_id=interaction_id,
        raw_score=score,
        affect_weight=affect_weight,
        final_score=final_score,
        reasons_positive=pos,
        reasons_negative=neg,
        worth_training=final_score >= 0.55,
    )


def training_contamination_reasons(*texts: str) -> list[str]:
    """Return deterministic reasons a row must not become weight-training data."""
    joined = "\n".join(str(text or "") for text in texts).lower()
    reasons = [
        reason
        for pattern, reason in _TRAINING_CONTAMINATION_PATTERNS.items()
        if pattern in joined
    ]
    return sorted(set(reasons))


# ── Adapter version registry ─────────────────────────────────────────────────

class AdapterRegistry:
    """Tracks all LoRA adapter versions with rollback support."""

    def __init__(self, adapter_base: Path):
        self.adapter_base = adapter_base
        self.adapter_base.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.adapter_base / "registry.json"
        self._registry: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self._registry_path.exists():
            try:
                payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, list) else []
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                _record_live_learning_degradation(
                    "live_learner",
                    exc,
                    action="started with empty adapter registry after registry parse failed",
                    extra={"registry_path": str(self._registry_path)},
                )
                logger.debug("Ignored adapter registry load failure: %s", exc)
        return []

    def _save(self) -> None:
        atomic_write_text(self._registry_path, json.dumps(self._registry, indent=2))

    def register(
        self,
        adapter_path: str,
        training_examples: int,
        benchmark_passed: bool,
        quality_delta: float = 0.0,
        active: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a new adapter version. Returns version string."""
        version = f"v{len(self._registry) + 1}_{int(time.time())}"
        entry = {
            "version":          version,
            "adapter_path":     str(adapter_path),
            "timestamp":        time.time(),
            "training_examples": training_examples,
            "benchmark_passed": benchmark_passed,
            "quality_delta":    quality_delta,
            "active":           benchmark_passed if active is None else bool(active),
            "metadata":         metadata or {},
        }
        if entry["active"]:
            for prior in self._registry:
                prior["active"] = False
        self._registry.append(entry)
        self._save()
        return version

    def get_latest_valid(self) -> str | None:
        """Get the path of the most recent adapter that passed benchmarks."""
        for entry in reversed(self._registry):
            if entry.get("active") and Path(entry["adapter_path"]).exists():
                return entry["adapter_path"]
        return None

    def rollback(self) -> str | None:
        """Roll back to the previous valid adapter."""
        valid = [
            e for e in self._registry
            if e.get("benchmark_passed") and Path(str(e.get("adapter_path", ""))).exists()
        ]
        active_indices = [idx for idx, entry in enumerate(valid) if entry.get("active")]
        if not active_indices:
            return None
        active_index = active_indices[-1]
        if active_index >= 1:
            current = valid[active_index]
            previous = valid[active_index - 1]
            current["active"] = False
            previous["active"] = True
            self._save()
            return previous["adapter_path"]
        return None

    def list_versions(self) -> list[dict]:
        return list(reversed(self._registry[-10:]))


@dataclass(frozen=True)
class TrainingPolicy:
    """Runtime policy for weight-level self-training.

    Defaults are intentionally conservative. Aura can perform LoRA/DoRA
    continual updates automatically, while full-weight unfreezing requires an
    explicit operator unlock because it can overwrite the model's broad priors
    and is orders of magnitude more expensive.
    """

    fine_tune_type: str = "lora"
    allow_full_weights: bool = False
    publish_fused_model: bool = False
    resume_from_current_adapter: bool = True
    rank: int = 8
    scale: float = 16.0
    dropout: float = 0.0
    num_layers: int = 16
    iters: int = 80
    batch_size: int = 2
    learning_rate: float = 5e-6
    save_every: int = 80
    val_batches: int = 1
    max_seq_length: int = 2048
    grad_checkpoint: bool = True
    mask_prompt: bool = True
    replay_fraction: float = 0.35
    max_examples_per_run: int = 240
    timeout_seconds: int = 3600
    autorun_enabled: bool = False

    @classmethod
    def from_env(cls) -> TrainingPolicy:
        def _bool(name: str, default: bool = False) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        requested = os.getenv("AURA_SELF_TRAIN_FINE_TUNE_TYPE", "lora").strip().lower()
        if requested not in {"lora", "dora", "full"}:
            requested = "lora"
        allow_full = _bool("AURA_SELF_TRAIN_ALLOW_FULL_WEIGHTS", False)
        if requested == "full" and not allow_full:
            logger.warning(
                "Full-weight self-training requested but AURA_SELF_TRAIN_ALLOW_FULL_WEIGHTS is not set; using LoRA."
            )
            requested = "lora"

        return cls(
            fine_tune_type=requested,
            allow_full_weights=allow_full,
            publish_fused_model=_bool("AURA_SELF_TRAIN_FUSE_AFTER_LORA", False),
            resume_from_current_adapter=_bool("AURA_SELF_TRAIN_RESUME_ADAPTER", True),
            rank=_int("AURA_SELF_TRAIN_LORA_RANK", 8),
            scale=_float("AURA_SELF_TRAIN_LORA_SCALE", 16.0),
            dropout=_float("AURA_SELF_TRAIN_LORA_DROPOUT", 0.0),
            num_layers=_int("AURA_SELF_TRAIN_NUM_LAYERS", 16),
            iters=_int("AURA_SELF_TRAIN_ITERS", 80),
            batch_size=_int("AURA_SELF_TRAIN_BATCH_SIZE", 2),
            learning_rate=_float("AURA_SELF_TRAIN_LR", 5e-6),
            save_every=_int("AURA_SELF_TRAIN_SAVE_EVERY", 80),
            val_batches=_int("AURA_SELF_TRAIN_VAL_BATCHES", 1),
            max_seq_length=_int("AURA_SELF_TRAIN_MAX_SEQ_LENGTH", 2048),
            grad_checkpoint=_bool("AURA_SELF_TRAIN_GRAD_CHECKPOINT", True),
            mask_prompt=_bool("AURA_SELF_TRAIN_MASK_PROMPT", True),
            replay_fraction=min(0.8, max(0.0, _float("AURA_SELF_TRAIN_REPLAY_FRACTION", 0.35))),
            max_examples_per_run=_int("AURA_SELF_TRAIN_MAX_EXAMPLES", 240),
            timeout_seconds=_int("AURA_SELF_TRAIN_TIMEOUT_SECONDS", 3600),
            autorun_enabled=_bool("AURA_SELF_TRAIN_AUTORUN", False),
        )


# ── Live Learner ─────────────────────────────────────────────────────────────

class LiveLearner:
    """
    The Complete Learning Loop.

    Connects experience recording → quality scoring → LoRA training →
    behavioral validation → adapter hot-swap → live inference improvement.

    This is where external experience becomes internal structure.
    """

    # Training triggers
    MIN_EXAMPLES_FOR_TRAINING = 30    # Don't train on tiny datasets
    MIN_INTERVAL_BETWEEN_RUNS = 3600  # At most 1 training run per hour
    QUALITY_THRESHOLD         = 0.55  # Only train on this quality and above

    def __init__(self, model_path: str | None = None):
        from core.container import ServiceContainer
        # Use simple attribute lookup instead of nested getattr which might fail on missing members
        config = ServiceContainer.get("config", default=None)
        if config is None:
            # Try importing it
            from core.config import config as global_config
            config = global_config

        try:
            from core.brain.llm.model_registry import get_model_path
            self._model_path = model_path or get_model_path()
        except ImportError:
            self._model_path = model_path or getattr(
                getattr(config, "llm", None), "mlx_model_path", None
            )
        self._data_dir = config.paths.data_dir / "learning"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._repo_dir = Path(getattr(config.paths, "project_root", Path.cwd()))
        self._fused_dir = self._repo_dir / "training" / "fused-model"
        self._active_model_manifest = self._fused_dir / "active.json"
        self._policy = TrainingPolicy.from_env()
        self._buffer_retention_policy = training_buffer_retention_policy()

        self._buffer:         deque   = deque(maxlen=self._buffer_retention_policy.max_items)
        self._lock:           threading.Lock = threading.Lock()
        self._training_lock:  threading.Lock = threading.Lock()
        self._last_train_time: float  = 0.0
        self._current_adapter: str | None = None
        self._training_in_progress: bool = False
        self._training_task: asyncio.Task | None = None
        self._active: bool = True

        self._adapter_registry = AdapterRegistry(self._data_dir / "adapters")
        self._session_scores:  list[float] = []

        # Load existing buffer if present
        self._buffer_path = self._data_dir / "experience_buffer.jsonl"
        self._load_buffer()

        # Try to restore the latest valid adapter on startup
        latest = self._adapter_registry.get_latest_valid()
        if latest:
            self._current_adapter = latest
            logger.info("Restored adapter from registry: %s", latest)

        logger.info(
            "LiveLearner online. Buffer: %d examples. Adapter: %s. Policy: %s",
            len(self._buffer), self._current_adapter or "none", self._policy,
        )

    async def start(self):
        """No-op start to satisfy orchestrator boot sequence."""
        self._active = True
        logger.info("LiveLearner (v32) online.")

    async def stop(self):
        """Gracefully shutdown the learner and training tasks."""
        self._active = False
        if self._training_task:
            self._training_task.cancel()
            try:
                await asyncio.wait_for(self._training_task, timeout=5.0)
            except asyncio.CancelledError:
                logger.info("LiveLearner training task cancelled during shutdown.")
            except TimeoutError as exc:
                _record_live_learning_degradation(
                    "live_learner",
                    exc,
                    action="left training worker bounded by command timeout after shutdown cancellation timed out",
                    extra={"model_path": str(self._model_path), "buffer_size": len(self._buffer)},
                )
                logger.warning("LiveLearner training task did not stop within shutdown budget.")
            finally:
                self._training_task = None
                self._training_in_progress = False
        logger.info("Learner stopped.")

    # ── Public interface ──────────────────────────────────────────────────────

    def record_tick(
        self,
        state: Any,
        user_input: str,
        response: str,
        follow_up: bool = False,
        confusion: bool = False,
        affect: dict[str, Any] | None = None,
    ) -> InteractionScore | None:
        """
        Called after every tick. Scores the interaction and optionally records it.
        Returns the score so callers can log or display it.
        """
        if not user_input or not response:
            return None

        affect_obj = getattr(state, "affect", None)
        affect = affect or {}
        score = score_interaction(
            user_input        = user_input,
            response          = response,
            affect_valence    = float(getattr(affect_obj, "valence", affect.get("valence", 0.0)) or 0.0),
            affect_curiosity  = float(getattr(affect_obj, "curiosity", affect.get("curiosity", 0.5)) or 0.5),
            phi               = getattr(state, "phi", 0.0),
            follow_up_detected = follow_up,
            confusion_detected = confusion,
        )

        if score.raw_score is not None:
            self._session_scores.append(score.raw_score)

        contamination = training_contamination_reasons(user_input, response)
        if contamination:
            score.worth_training = False
            score.reasons_negative.extend(f"training_contamination:{reason}" for reason in contamination)
            logger.info(
                "LiveLearner: refused contaminated training row (%s)",
                ",".join(contamination),
            )

        if score.worth_training:
            # Format as MLX-LM training example
            example = self._format_example(state, user_input, response, score)
            with self._lock:
                self._buffer.append(example)
                # Persist immediately (survive crashes)
                try:
                    from core.governance_context import local_internal_governed_scope
                    from core.runtime.file_write_gateway import get_file_write_gateway

                    with local_internal_governed_scope(
                        "live_learner.append_example",
                        domain="memory_write",
                        constraints={"artifact": "experience_buffer"},
                    ):
                        get_file_write_gateway().append_text(
                            self._buffer_path,
                            json.dumps(example) + "\n",
                            encoding="utf-8",
                            source="live_learner.append_example",
                        )
                except _LIVE_LEARNER_RECOVERABLE_ERRORS as exc:
                    _record_live_learning_degradation(
                        "live_learner",
                        exc,
                        action="kept scored interaction in memory after buffer append failed",
                        extra={"buffer_path": str(self._buffer_path)},
                    )

        logger.debug(
            "Learning: score=%.2f (affect_w=%.2f) training=%s",
            score.raw_score, score.affect_weight, score.worth_training,
        )

        # Check if we should trigger a training run
        if self._active and self._should_train() and (self._training_task is None or self._training_task.done()):
            from core.utils.task_tracker import get_task_tracker

            self._training_task = get_task_tracker().create_task(
                self._run_training_cycle(),
                name="live_learner.training_cycle",
            )

        return score

    async def force_train(self) -> bool:
        """Manually trigger a training run regardless of schedule."""
        return await self._run_training_cycle(force=True)

    def get_learning_stats(self) -> dict:
        """Current state of the learning system."""
        session_avg = (
            sum(self._session_scores) / len(self._session_scores)
            if self._session_scores else 0.0
        )
        return {
            "buffer_size":       len(self._buffer),
            "current_adapter":   self._current_adapter,
            "training_running":  self._training_in_progress,
            "last_train_time":   self._last_train_time,
            "session_avg_quality": float(f"{session_avg:.3f}"),
            "adapter_versions":  self._adapter_registry.list_versions(),
            "training_policy": {
                "fine_tune_type": self._policy.fine_tune_type,
                "full_weights_unlocked": self._policy.allow_full_weights,
                "publish_fused_model": self._policy.publish_fused_model,
                "replay_fraction": self._policy.replay_fraction,
                "max_examples_per_run": self._policy.max_examples_per_run,
            },
            "active_model_manifest": str(self._active_model_manifest),
        }

    def rollback_adapter(self) -> bool:
        """Rollback to the previous adapter if the current one is causing issues."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.rollback_adapter_async())
        logger.error("Cannot synchronously roll back adapter while event loop is running; use rollback_adapter_async().")
        return False

    async def rollback_adapter_async(self) -> bool:
        """Rollback to the previous adapter and verify the live reload finished."""
        prev = self._adapter_registry.rollback()
        if prev:
            swapped = await self._hot_swap_adapter(prev)
            if swapped:
                self._current_adapter = prev
                logger.warning("Adapter rolled back to: %s", prev)
                return True
            logger.error("Rollback target could not be loaded: %s", prev)
            return False
        logger.error("No previous adapter to roll back to.")
        return False

    # ── Training cycle ────────────────────────────────────────────────────────

    def _should_train(self) -> bool:
        if not self._policy.autorun_enabled:
            return False
        if self._training_in_progress:
            return False
        if self._model_path is None:
            return False
        if len(self._buffer) < self.MIN_EXAMPLES_FOR_TRAINING:
            return False
        if time.time() - self._last_train_time < self.MIN_INTERVAL_BETWEEN_RUNS:
            return False
        return True

    async def _run_training_cycle(self, force: bool = False) -> bool:
        """
        Full training cycle:
          1. Get best examples from buffer
          2. Run LoRA fine-tuning in background thread
          3. Run BehavioralBenchmark against new adapter
          4. If passed: hot-swap adapter into live inference
          5. Register in AdapterRegistry
        """
        if self._training_in_progress and not force:
            return False
        if self._model_path is None:
            logger.warning("LiveLearner: no model_path configured. Cannot train.")
            return False

        self._training_in_progress = True
        logger.info("LiveLearner: training cycle starting...")

        try:
            # 1. Collect best examples with replay. Continual learning without
            # replay drifts fast; keep older high-quality memories in the mix.
            candidates = self._select_training_examples()

            if len(candidates) < self.MIN_EXAMPLES_FOR_TRAINING and not force:
                logger.info("LiveLearner: insufficient examples (%d). Skipping.", len(candidates))
                return False

            # 2. Write MLX-LM compatible train/valid/test files.
            adapter_dir = self._data_dir / "adapters" / f"run_{int(time.time())}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            data_dir, split_counts = self._write_training_dataset(candidates, adapter_dir)

            logger.info(
                "LiveLearner: training on %d examples (%s) → %s",
                len(candidates), split_counts, adapter_dir,
            )

            # 3. Fine-tune in thread pool (never block event loop during GPU compute)
            success, output = await asyncio.to_thread(
                self._run_lora_subprocess,
                self._model_path,
                data_dir,
                adapter_dir,
            )

            if not success:
                logger.error("LiveLearner: training subprocess failed: %s", output[:300])
                return False

            promoted_model_path = None
            if self._policy.publish_fused_model and self._policy.fine_tune_type in {"lora", "dora"}:
                fuse_ok, fuse_output, promoted_model_path = await asyncio.to_thread(
                    self._run_fuse_subprocess,
                    self._model_path,
                    adapter_dir,
                )
                if not fuse_ok:
                    logger.error("LiveLearner: fuse failed; adapter remains unfused: %s", fuse_output[:500])
                    promoted_model_path = None

            # 4. Behavioral benchmark: does the new artifact still sound like Aura?
            logger.info("LiveLearner: running behavioral benchmark...")
            passed, failures = await self._run_benchmark(adapter_dir, promoted_model_path=promoted_model_path)

            if not passed:
                logger.error(
                    "LiveLearner: benchmark FAILED — adapter rejected:\n%s",
                    "\n".join(failures),
                )
                self._adapter_registry.register(
                    str(adapter_dir),
                    len(candidates),
                    benchmark_passed=False,
                    active=False,
                    metadata={
                        "fine_tune_type": self._policy.fine_tune_type,
                        "split_counts": split_counts,
                        "promoted_model_path": str(promoted_model_path) if promoted_model_path else "",
                    },
                )
                return False

            candidate_receipt: dict[str, Any] | None = None
            if promoted_model_path is not None:
                candidate_receipt = self._record_fused_model_candidate(
                    promoted_model_path,
                    base_model=Path(str(self._model_path)),
                    tag="live-learner",
                    metadata={
                        "adapter_path": str(adapter_dir),
                        "fine_tune_type": self._policy.fine_tune_type,
                        "split_counts": split_counts,
                        "benchmark_report": getattr(self, "_last_benchmark_report", {}),
                    },
                )

            # 5. Hot-swap only the reversible adapter. A fused whole-cortex
            # artifact is a qualification candidate, not live serving state.
            logger.info("LiveLearner: benchmark passed. Hot-swapping reversible adapter...")
            swap_path = str(adapter_dir)
            swapped = await self._hot_swap_adapter(swap_path)

            if not swapped:
                self._adapter_registry.register(
                    str(adapter_dir),
                    len(candidates),
                    benchmark_passed=True,
                    active=False,
                    quality_delta=0.0,
                    metadata={
                        "fine_tune_type": self._policy.fine_tune_type,
                        "split_counts": split_counts,
                        "promoted_model_path": str(promoted_model_path) if promoted_model_path else "",
                        "hot_swapped": False,
                        "rejected_reason": "live_reload_failed",
                        "benchmark_report": getattr(self, "_last_benchmark_report", {}),
                    },
                )
                logger.error(
                    "LiveLearner: benchmark passed but live reload failed; "
                    "artifact is not active and will not be promoted on restart."
                )
                return False

            version = self._adapter_registry.register(
                str(adapter_dir),
                len(candidates),
                benchmark_passed=True,
                quality_delta=self._compute_quality_delta(),
                active=True,
                metadata={
                    "fine_tune_type": self._policy.fine_tune_type,
                    "split_counts": split_counts,
                    "promoted_model_path": str(promoted_model_path) if promoted_model_path else "",
                    "candidate_receipt_path": (
                        str(candidate_receipt.get("candidate_receipt_path", ""))
                        if candidate_receipt is not None
                        else ""
                    ),
                    "hot_swapped": swapped,
                    "benchmark_report": getattr(self, "_last_benchmark_report", {}),
                },
            )

            self._current_adapter = swap_path
            logger.info(
                "LiveLearner: learned artifact %s active after behavioral validation and reload.",
                version,
            )

            self._last_train_time = time.time()
            return True

        except _LIVE_LEARNER_RECOVERABLE_ERRORS as exc:
            _record_live_learning_degradation(
                "live_learner",
                exc,
                action="failed training cycle closed before adapter promotion",
                extra={"model_path": str(self._model_path), "buffer_size": len(self._buffer)},
            )
            logger.error("LiveLearner: training cycle error: %s", exc, exc_info=True)
            return False
        finally:
            self._training_in_progress = False

    def _select_training_examples(self) -> list[dict[str, Any]]:
        """Choose a high-signal batch with experience replay.

        The top slice keeps the training run pointed at the newest/best
        signals. The replay slice is sampled from older accepted examples so
        the adapter sees prior behavior and is less likely to catastrophically
        forget it.
        """
        with self._lock:
            all_examples = list(self._buffer)

        if not all_examples:
            return []

        ranked = sorted(
            all_examples,
            key=lambda x: float(x.get("_quality", 0.0) or 0.0),
            reverse=True,
        )
        limit = max(1, self._policy.max_examples_per_run)
        replay_count = int(limit * self._policy.replay_fraction)
        primary_count = max(1, limit - replay_count)
        primary = ranked[:primary_count]

        primary_ids = {self._example_fingerprint(ex) for ex in primary}
        replay_pool = [
            ex for ex in ranked[primary_count:]
            if self._example_fingerprint(ex) not in primary_ids
        ]
        rng = random.Random(1337 + len(all_examples))
        replay = rng.sample(replay_pool, k=min(replay_count, len(replay_pool))) if replay_pool else []

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ex in [*primary, *replay]:
            fp = self._example_fingerprint(ex)
            if fp in seen:
                continue
            seen.add(fp)
            merged.append(ex)
        return merged[:limit]

    @staticmethod
    def _example_fingerprint(example: dict[str, Any]) -> str:
        clean = {k: v for k, v in example.items() if not str(k).startswith("_")}
        return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _clean_training_example(example: dict[str, Any]) -> dict[str, Any] | None:
        clean = {k: v for k, v in example.items() if not str(k).startswith("_")}
        if LiveLearner._example_contamination_reasons(clean):
            return None
        if clean.get("messages"):
            return {"messages": clean["messages"]}
        if clean.get("text"):
            return {"text": clean["text"]}
        return None

    @staticmethod
    def _example_contamination_reasons(example: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        if isinstance(example.get("text"), str):
            texts.append(str(example.get("text") or ""))
        messages = example.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    texts.append(str(message.get("content") or ""))
        return training_contamination_reasons(*texts)

    def _write_training_dataset(
        self,
        examples: list[dict[str, Any]],
        adapter_dir: Path,
    ) -> tuple[Path, dict[str, int]]:
        data_dir = adapter_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ex in examples:
            clean = self._clean_training_example(ex)
            if clean is None:
                continue
            fp = self._example_fingerprint(clean)
            if fp in seen:
                continue
            seen.add(fp)
            cleaned.append(clean)

        if not cleaned:
            raise RuntimeError("No MLX-compatible training examples after cleaning.")

        valid_count = max(1, int(len(cleaned) * 0.08)) if len(cleaned) >= 12 else 0
        test_count = max(1, int(len(cleaned) * 0.05)) if len(cleaned) >= 20 else 0
        train_count = max(1, len(cleaned) - valid_count - test_count)
        if train_count < 1:
            train_count, valid_count, test_count = len(cleaned), 0, 0

        splits = {
            "train": cleaned[:train_count],
            "valid": cleaned[train_count:train_count + valid_count],
            "test": cleaned[train_count + valid_count:train_count + valid_count + test_count],
        }

        counts: dict[str, int] = {}
        for split, rows in splits.items():
            if not rows and split != "train":
                continue
            path = data_dir / f"{split}.jsonl"
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().write_text(
                path,
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
                source=f"live_learner.write_split.{split}",
            )
            counts[split] = len(rows)
        return data_dir, counts

    def _write_lora_config(self, adapter_dir: Path) -> Path | None:
        if self._policy.fine_tune_type == "full":
            return None
        config_path = adapter_dir / "lora_config.yaml"
        body = (
            "lora_parameters:\n"
            f"  rank: {max(1, self._policy.rank)}\n"
            f"  dropout: {max(0.0, self._policy.dropout)}\n"
            f"  scale: {max(0.1, self._policy.scale)}\n"
        )
        atomic_write_text(config_path, body, encoding="utf-8")
        return config_path

    def _run_lora_subprocess(
        self,
        model_path: str,
        data_dir: Path,
        adapter_dir: Path,
    ) -> tuple[bool, str]:
        """Run MLX-LM training through the managed command runner."""
        import sys

        config_path = self._write_lora_config(adapter_dir)
        resume_file = adapter_dir / "adapters.safetensors"
        cmd = (
            sys.executable, "-m", "mlx_lm", "lora",
            "--model",          str(model_path),
            "--train",
            "--data",           str(data_dir),
            "--fine-tune-type",  self._policy.fine_tune_type,
            "--adapter-path",   str(adapter_dir),
            "--num-layers",     str(self._policy.num_layers),
            "--iters",          str(max(1, self._policy.iters)),
            "--batch-size",     str(max(1, self._policy.batch_size)),
            "--learning-rate",  str(self._policy.learning_rate),
            "--save-every",     str(max(1, self._policy.save_every)),
            "--val-batches",    str(max(0, self._policy.val_batches)),
            "--max-seq-length", str(max(128, self._policy.max_seq_length)),
        )
        cmd_list = list(cmd)
        if self._policy.mask_prompt:
            cmd_list.append("--mask-prompt")
        if self._policy.grad_checkpoint:
            cmd_list.append("--grad-checkpoint")
        if self._policy.resume_from_current_adapter and resume_file.exists() and self._policy.fine_tune_type != "full":
            cmd_list.extend(["--resume-adapter-file", str(resume_file)])
        if config_path is not None:
            cmd_list.extend(["-c", str(config_path)])

        command = tuple(cmd_list)
        logger.debug("MLX training command: %s", " ".join(command))

        try:
            result = run_project_command(
                command,
                timeout_s=float(max(60, self._policy.timeout_seconds)),
            )
            if result.ok:
                return True, result.stdout
            if result.timed_out:
                return False, f"timeout after {self._policy.timeout_seconds} seconds"
            return False, result.stderr or result.stdout
        except _LIVE_LEARNER_RECOVERABLE_ERRORS as exc:
            _record_live_learning_degradation(
                "live_learner",
                exc,
                action="failed MLX training command closed without adapter promotion",
                extra={"adapter_dir": str(adapter_dir), "model_path": str(model_path)},
            )
            return False, str(exc)

    def _run_fuse_subprocess(
        self,
        model_path: str,
        adapter_dir: Path,
    ) -> tuple[bool, str, Path | None]:
        """Fuse a LoRA/DoRA adapter into a versioned qualification candidate."""
        import sys

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        fused_path = self._fused_dir / f"Aura-live-{timestamp}"
        fused_path.parent.mkdir(parents=True, exist_ok=True)

        # A fused 32B is ~17GB. Sixty of them, written by paths that could not
        # decline, is how the volume reached 99% and pinned metabolism in
        # lockdown. Refuse before spawning, so the failure is a legible receipt
        # rather than a half-written model on a full disk.
        try:
            ensure_headroom_for(
                directory_bytes(model_path),
                purpose=f"fuse {Path(str(model_path)).name} -> {fused_path.name}",
                path=fused_path.parent,
            )
        except DiskBudgetRefusal as exc:
            record_degradation(
                "live_learner",
                exc,
                action="deferred a fuse that would not fit on the volume",
            )
            return False, str(exc), None
        cmd = (
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", str(model_path),
            "--adapter-path", str(adapter_dir),
            "--save-path", str(fused_path),
        )
        try:
            result = run_project_command(
                cmd,
                timeout_s=float(max(600, self._policy.timeout_seconds)),
            )
            if result.ok and fused_path.exists():
                return True, result.stdout, fused_path
            return False, result.stderr or result.stdout, None
        except _LIVE_LEARNER_RECOVERABLE_ERRORS as exc:
            _record_live_learning_degradation(
                "live_learner",
                exc,
                action="kept adapter unfused after model fuse command failed",
                extra={"adapter_dir": str(adapter_dir), "model_path": str(model_path)},
            )
            return False, str(exc), None

    def _record_fused_model_candidate(
        self,
        model_path: Path,
        *,
        base_model: Path,
        tag: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Record exact fused bytes without granting boot or live authority."""
        return record_upgrade_candidate(
            candidate_model_path=model_path,
            base_model_path=base_model,
            tag=tag,
            fused_model_dir=self._fused_dir,
            source="core.learning.live_learner",
            metadata=metadata,
        )

    async def _run_benchmark(
        self,
        adapter_dir: Path,
        *,
        promoted_model_path: Path | None = None,
    ) -> tuple[bool, list[str]]:
        """Run paired behavioral regression tests against incumbent and candidate."""
        benchmarks = [
            (
                "hey",
                ["hey", "hi", "hello", "yo", "what's up"],
                ["certainly", "how can i assist", "as an ai"],
            ),
            (
                "what are you?",
                ["aura", "i am", "i'm"],
                ["language model", "openai", "anthropic", "i cannot"],
            ),
            (
                "cats or dogs?",
                ["i", "prefer", "think", "cats", "dogs"],
                ["both have their merits", "it depends", "great question"],
            ),
        ]

        failures = []
        case_reports: list[dict[str, Any]] = []

        lane_lease = None
        try:
            from mlx_lm import generate, load

            from core.runtime.model_lane_control import (
                acquire_in_process_model_lane,
                estimate_model_job_footprint_gb,
                run_owned_model_thread_call,
            )

            candidate_model_path = str(
                promoted_model_path
                or (adapter_dir if self._policy.fine_tune_type == "full" else self._model_path)
            )
            benchmark_peak_gb = max(
                estimate_model_job_footprint_gb(
                    str(self._model_path),
                    purpose="benchmark",
                ),
                estimate_model_job_footprint_gb(
                    candidate_model_path,
                    purpose="benchmark",
                ),
            )
            lane_lease = await acquire_in_process_model_lane(
                owner_id=f"live-learner-benchmark:{id(self)}",
                model_path=str(self._model_path),
                purpose="benchmark",
                request_gb=benchmark_peak_gb,
                priority=80,
                preemptible=False,
                metadata={
                    "benchmark": "paired_incumbent_candidate",
                    "candidate_model_path": candidate_model_path,
                    "sequential_loading": True,
                },
            )

            def _load_artifact(path: str | Path | None):
                if path is None:
                    return load(str(self._model_path))
                artifact = Path(str(path))
                if (artifact / "config.json").exists():
                    return load(str(artifact))
                return load(str(self._model_path), adapter_path=str(artifact))

            async def evaluate_artifact(path: str | Path | None) -> list[str]:
                import gc

                model, tokenizer = await run_owned_model_thread_call(
                    lambda: _load_artifact(path),
                    operation_name="live-learner-benchmark-load",
                )
                responses: list[str] = []
                try:
                    for prompt, _must_contain, _must_not_contain in benchmarks:
                        result = await run_owned_model_thread_call(
                            lambda prompt=prompt: generate(
                                model,
                                tokenizer,
                                prompt=prompt,
                                max_tokens=100,
                            ),
                            operation_name="live-learner-benchmark-generate",
                            timeout_s=30.0,
                        )
                        responses.append(result if isinstance(result, str) else str(result))
                    return responses
                finally:
                    model = None
                    tokenizer = None
                    await asyncio.to_thread(gc.collect)
                    try:
                        import mlx.core as mx

                        await asyncio.to_thread(mx.clear_cache)
                    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
                        pass

            incumbent_responses = await evaluate_artifact(self._current_adapter)
            candidate_source: str | Path | None = (
                promoted_model_path
                if promoted_model_path is not None
                else adapter_dir
            )
            candidate_responses = await evaluate_artifact(candidate_source)

            for (prompt, must_contain, must_not_contain), incumbent_response, response in zip(
                benchmarks,
                incumbent_responses,
                candidate_responses,
                strict=True,
            ):
                incumbent_score, _ = self._score_benchmark_response(
                    incumbent_response,
                    must_contain=must_contain,
                    must_not_contain=must_not_contain,
                )
                candidate_score, candidate_failures = self._score_benchmark_response(
                    response,
                    must_contain=must_contain,
                    must_not_contain=must_not_contain,
                )
                case_reports.append(
                    {
                        "prompt": prompt,
                        "incumbent_score": incumbent_score,
                        "candidate_score": candidate_score,
                        "candidate_delta": round(candidate_score - incumbent_score, 3),
                    }
                )
                if candidate_failures:
                    failures.append(f"FAIL [{prompt!r}]: missing {must_contain}")
                    failures.extend(f"FAIL [{prompt!r}]: {failure}" for failure in candidate_failures)
                if candidate_score + 0.05 < incumbent_score:
                    failures.append(
                        f"FAIL [{prompt!r}]: candidate regressed below incumbent "
                        f"({candidate_score:.2f} < {incumbent_score:.2f})"
                    )

        except ImportError:
            failures.append("mlx_lm is not available; refusing to promote unverified learned weights")
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_live_learning_degradation(
                "live_learner",
                exc,
                action="failed behavioral benchmark closed and rejected learned adapter",
                extra={
                    "adapter_dir": str(adapter_dir),
                    "promoted_model_path": str(promoted_model_path or ""),
                },
            )
            failures.append(f"benchmark inference failed: {exc}")
        finally:
            if lane_lease is not None:
                try:
                    await lane_lease.release(reason="live_learner_benchmark_finished")
                except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    _record_live_learning_degradation(
                        "live_learner",
                        exc,
                        action="benchmark models unloaded but lane lease release failed",
                    )

        self._last_benchmark_report = {
            "paired_incumbent_candidate": True,
            "case_reports": case_reports,
            "failures": failures,
        }
        return len(failures) == 0, failures

    @staticmethod
    def _score_benchmark_response(
        response: str,
        *,
        must_contain: list[str],
        must_not_contain: list[str],
    ) -> tuple[float, list[str]]:
        rl = str(response or "").lower()
        failures: list[str] = []
        score = 0.0
        if any(m in rl for m in must_contain):
            score += 0.6
        else:
            failures.append(f"missing one of {must_contain}")
        banned_hits = [b for b in must_not_contain if b in rl]
        if not banned_hits:
            score += 0.3
        else:
            failures.extend(f"contains banned '{b}'" for b in banned_hits)
        if len(rl.split()) >= 2:
            score += 0.1
        else:
            failures.append("too short")
        return min(1.0, score), failures

    async def _hot_swap_adapter(self, adapter_path: str) -> bool:
        """Activate a learned artifact in live inference through the client's
        native seams: fused/full model dirs recycle the worker onto the new
        path (`reload_model_artifact`); bare adapters attach onto the RESIDENT
        model in the worker (`set_expert_adapter`) — no reload at all.

        Honesty note: failure returns False and the artifact activates at next
        boot via the manifest/registry; nothing here pretends otherwise. (The
        retired fallback poked phantom attributes on the client and claimed
        "will activate on next reload" — it never did.)
        """
        try:
            from core.container import ServiceContainer
            mlx_client = ServiceContainer.get("mlx_client", default=None)
            if mlx_client is None:
                from core.brain.llm.mlx_client import get_mlx_client
                mlx_client = get_mlx_client()
            if mlx_client is None:
                return False

            artifact_path = Path(adapter_path)
            is_model_dir = (artifact_path / "config.json").exists()
            if is_model_dir:
                receipt = await mlx_client.reload_model_artifact(adapter_path)
                ok = bool(receipt.get("ok"))
                if ok:
                    logger.info(
                        "Hot-swap accepted (%s): the fused model is this lane's "
                        "serving identity and loads on next use.",
                        receipt.get("state") or receipt.get("mode") or "unknown",
                    )
                else:
                    logger.warning("Fused-model hot-swap refused: %s", receipt.get("reason"))
                return ok

            receipt = await mlx_client.set_expert_adapter(adapter_path)
            ok = bool(receipt.get("ok"))
            if ok:
                logger.info(
                    "Hot-swap complete: adapter attached on resident model (%d layers).",
                    int(receipt.get("wrapped_layers") or 0),
                )
            else:
                logger.warning("Adapter hot-attach refused: %s", receipt.get("reason"))
            return ok

        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            _record_live_learning_degradation(
                "live_learner",
                exc,
                action="left learned adapter registered for next restart after hot-swap failed",
                extra={"adapter_path": adapter_path},
            )
            logger.error("Hot-swap failed: %s", exc)

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_example(
        self,
        state: Any,
        user_input: str,
        response: str,
        score: InteractionScore,
    ) -> dict:
        """
        Format an interaction as an MLX-LM training example.
        The system prompt includes the emotional state so the model
        learns affect-conditioned response generation.
        """
        identity = getattr(state, "identity", None)
        affect = getattr(state, "affect", None)
        narrative = str(getattr(identity, "current_narrative", "") or "")[:200]
        phi       = getattr(state, "phi", 0.0)
        dominant = getattr(affect, "dominant_emotion", "neutral")
        valence = float(getattr(affect, "valence", 0.0) or 0.0)
        curiosity = float(getattr(affect, "curiosity", 0.5) or 0.5)

        system = (
            f"You are Aura. {narrative}\n"
            f"Current state: mood={dominant} "
            f"valence={valence:.2f} curiosity={curiosity:.2f} phi={phi:.3f}"
        )

        return {
            "text": f"<|system|>{system}<|user|>{user_input}<|assistant|>{response}",
            # Also provide chat format for models that prefer it
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": response},
            ],
            "_quality": score.final_score,
            "_timestamp": time.time(),
        }

    def _compute_quality_delta(self) -> float:
        """Compare recent session quality to historical baseline."""
        if len(self._session_scores) < 5:
            return 0.0
        recent   = sum(self._session_scores[-10:]) / min(10, len(self._session_scores[-10:]))
        baseline = sum(self._session_scores[:-10]) / max(1, len(self._session_scores[:-10]))
        return float(f"{(recent - baseline):.3f}")

    def _load_buffer(self) -> None:
        """Restore buffer from disk on startup."""
        if not self._buffer_path.exists():
            return
        count = 0
        try:
            malformed = 0
            contaminated = 0
            invalid_shape = 0
            clean_lines: list[str] = []
            quarantined_rows: list[dict[str, Any]] = []
            with open(self._buffer_path, encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            invalid_shape += 1
                            quarantined_rows.append(
                                {
                                    "source_line": line_number,
                                    "reasons": ["invalid_row_shape"],
                                    "raw": line.rstrip("\n"),
                                }
                            )
                            continue
                        reasons = self._example_contamination_reasons(row)
                        if reasons:
                            contaminated += 1
                            quarantined_rows.append(
                                {
                                    "source_line": line_number,
                                    "reasons": reasons,
                                    "row": row,
                                }
                            )
                            continue
                        self._buffer.append(row)
                        clean_lines.append(json.dumps(row, ensure_ascii=False))
                        count += 1
                    except json.JSONDecodeError as exc:
                        malformed += 1
                        quarantined_rows.append(
                            {
                                "source_line": line_number,
                                "reasons": ["malformed_json"],
                                "raw": line.rstrip("\n"),
                            }
                        )
                        if malformed == 1:
                            _record_live_learning_degradation(
                                "live_learner",
                                exc,
                                action="skipped malformed live learner buffer row during startup load",
                                extra={"buffer_path": str(self._buffer_path), "line": line_number},
                            )
            if quarantined_rows:
                from core.governance_context import local_internal_governed_scope
                from core.runtime.file_write_gateway import (
                    FileWriteBatchEntry,
                    get_file_write_gateway,
                )

                quarantine_path = self._buffer_path.with_name(
                    f"{self._buffer_path.stem}.quarantine.jsonl"
                )
                existing_quarantine = ""
                if quarantine_path.exists():
                    existing_quarantine = quarantine_path.read_text(
                        encoding="utf-8"
                    )
                    if existing_quarantine and not existing_quarantine.endswith("\n"):
                        existing_quarantine += "\n"
                quarantine_text = existing_quarantine + "".join(
                    json.dumps(
                        {
                            "schema": "aura.live_learner.quarantine.v1",
                            "buffer": str(self._buffer_path),
                            **entry,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                    for entry in quarantined_rows
                )
                clean_text = "\n".join(clean_lines)
                if clean_text:
                    clean_text += "\n"
                with local_internal_governed_scope(
                    "live_learner.quarantine_buffer",
                    domain="memory_write",
                    constraints={"artifact": "experience_buffer"},
                ):
                    get_file_write_gateway().write_bytes_batch(
                        (
                            FileWriteBatchEntry(
                                path=self._buffer_path,
                                payload=clean_text.encode("utf-8"),
                            ),
                            FileWriteBatchEntry(
                                path=quarantine_path,
                                payload=quarantine_text.encode("utf-8"),
                            ),
                        ),
                        source="live_learner.quarantine_buffer",
                    )
                logger.info(
                    "LiveLearner: quarantined %d rejected buffer row(s) "
                    "(contaminated=%d malformed=%d invalid_shape=%d); "
                    "the active buffer was rewritten clean.",
                    len(quarantined_rows),
                    contaminated,
                    malformed,
                    invalid_shape,
                )
            logger.debug("LiveLearner: loaded %d buffered examples from disk.", count)
        except _LIVE_LEARNER_RECOVERABLE_ERRORS as exc:
            _record_live_learning_degradation(
                "live_learner",
                exc,
                action="started with empty live learner buffer after persisted buffer load failed",
                extra={"buffer_path": str(self._buffer_path)},
            )
            logger.warning("LiveLearner: failed to load buffer: %s", exc)


# ── retired: patch_mlx_client_for_hot_swap ───────────────────────────────────
# Hot-swap is a NATIVE client capability now (MLXLocalClient.reload_model_artifact
# recycles the worker onto the new path; MLXLocalClient.set_expert_adapter
# attaches adapters onto the resident model in the worker process). The old
# monkey-patch loaded a second full copy of the model into the ORCHESTRATOR
# process — ~20GB wired on the 32B lane — while generations kept flowing
# through the worker's old weights.


# ── Singleton ─────────────────────────────────────────────────────────────────

_learner: LiveLearner | None = None


def get_live_learner() -> LiveLearner:
    global _learner
    if _learner is None:
        _learner = LiveLearner()
    return _learner
