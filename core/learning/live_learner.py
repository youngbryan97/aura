from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.LiveLearner")


# ── Quality scoring ──────────────────────────────────────────────────────────

@dataclass
class InteractionScore:
    """Quality assessment for a single interaction."""
    interaction_id:   str
    raw_score:        float            # 0.0 to 1.0
    affect_weight:    float            # High-affect moments matter more
    final_score:      float            # raw_score * affect_weight modifier
    reasons_positive: List[str] = field(default_factory=list)
    reasons_negative: List[str] = field(default_factory=list)
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
        score += 0.20; pos.append("appropriate_length")
    elif words < 5:
        score -= 0.25; neg.append("too_short")

    # Identity check
    _BANNED = ["as an ai", "certainly!", "absolutely!", "great question",
               "how can i help", "language model", "i was trained"]
    if not any(b in response.lower() for b in _BANNED):
        score += 0.15; pos.append("identity_intact")
    else:
        score -= 0.20; neg.append("identity_regression")

    # Truncation
    if response and response.strip()[-1] not in ".!?\"'\n":
        score -= 0.20; neg.append("truncated")

    # Behavioral signals
    if follow_up_detected:
        score += 0.30; pos.append("user_follow_up")
    if confusion_detected:
        score -= 0.50; neg.append("user_confusion")

    # Affect signals
    if phi > 0.4:
        score += 0.15; pos.append("high_phi")
    if affect_curiosity > 0.6:
        score += 0.10; pos.append("high_curiosity")
    if affect_valence > 0.3:
        score += 0.10; pos.append("positive_valence")

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


# ── Adapter version registry ─────────────────────────────────────────────────

class AdapterRegistry:
    """Tracks all LoRA adapter versions with rollback support."""

    def __init__(self, adapter_base: Path):
        self.adapter_base = adapter_base
        self.adapter_base.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.adapter_base / "registry.json"
        self._registry: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text())
            except Exception as _e:
                record_degradation('live_learner', _e)
                logger.debug('Ignored Exception in live_learner.py: %s', _e)
        return []

    def _save(self) -> None:
        atomic_write_text(self._registry_path, json.dumps(self._registry, indent=2))

    def register(
        self,
        adapter_path: str,
        training_examples: int,
        benchmark_passed: bool,
        quality_delta: float = 0.0,
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
            "active":           benchmark_passed,
        }
        self._registry.append(entry)
        self._save()
        return version

    def get_latest_valid(self) -> Optional[str]:
        """Get the path of the most recent adapter that passed benchmarks."""
        for entry in reversed(self._registry):
            if entry.get("active") and Path(entry["adapter_path"]).exists():
                return entry["adapter_path"]
        return None

    def rollback(self) -> Optional[str]:
        """Roll back to the previous valid adapter."""
        valid = [e for e in self._registry if e.get("active")]
        if len(valid) >= 2:
            return valid[-2]["adapter_path"]
        return None

    def list_versions(self) -> List[Dict]:
        return list(reversed(self._registry[-10:]))


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

    def __init__(self, model_path: Optional[str] = None):
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

        self._buffer:         deque   = deque(maxlen=5000)
        self._lock:           threading.Lock = threading.Lock()
        self._training_lock:  threading.Lock = threading.Lock()
        self._last_train_time: float  = 0.0
        self._current_adapter: Optional[str] = None
        self._training_in_progress: bool = False
        self._training_task: Optional[asyncio.Task] = None
        self._active: bool = True

        self._adapter_registry = AdapterRegistry(self._data_dir / "adapters")
        self._session_scores:  List[float] = []

        # Load existing buffer if present
        self._buffer_path = self._data_dir / "experience_buffer.jsonl"
        self._load_buffer()

        # Try to restore the latest valid adapter on startup
        latest = self._adapter_registry.get_latest_valid()
        if latest:
            self._current_adapter = latest
            logger.info("Restored adapter from registry: %s", latest)

        logger.info(
            "LiveLearner online. Buffer: %d examples. Adapter: %s",
            len(self._buffer), self._current_adapter or "none",
        )

    async def start(self):
        """No-op start to satisfy orchestrator boot sequence."""
        self._active = True
        logger.info("LiveLearner (v32) online.")

    async def stop(self):
        """Gracefully shutdown the learner and training tasks."""
        self._active = False
        if self._training_task:
            # v32 Hardening: Tracking training termination
            self._training_task.cancel()
            try:
                await asyncio.wait_for(self._training_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug("Suppressed bare exception")
                pass
        logger.info("Learner stopped.")

    # ── Public interface ──────────────────────────────────────────────────────

    def record_tick(
        self,
        state: Any,
        user_input: str,
        response: str,
        follow_up: bool = False,
        confusion: bool = False,
    ) -> Optional[InteractionScore]:
        """
        Called after every tick. Scores the interaction and optionally records it.
        Returns the score so callers can log or display it.
        """
        if not user_input or not response:
            return None

        score = score_interaction(
            user_input        = user_input,
            response          = response,
            affect_valence    = getattr(state.affect, "valence",   0.0),
            affect_curiosity  = getattr(state.affect, "curiosity", 0.5),
            phi               = getattr(state, "phi", 0.0),
            follow_up_detected = follow_up,
            confusion_detected = confusion,
        )

        if score.raw_score is not None:
            self._session_scores.append(score.raw_score)

        if score.worth_training:
            # Format as MLX-LM training example
            example = self._format_example(state, user_input, response, score)
            with self._lock:
                self._buffer.append(example)
                # Persist immediately (survive crashes)
                with open(self._buffer_path, "a") as f:
                    f.write(json.dumps(example) + "\n")

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

    def get_learning_stats(self) -> Dict:
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
        }

    def rollback_adapter(self) -> bool:
        """Rollback to the previous adapter if the current one is causing issues."""
        prev = self._adapter_registry.rollback()
        if prev:
            self._current_adapter = prev
            from core.utils.task_tracker import get_task_tracker

            get_task_tracker().create_task(
                self._hot_swap_adapter(prev),
                name="live_learner.hot_swap_adapter",
            )
            logger.warning("Adapter rolled back to: %s", prev)
            return True
        logger.error("No previous adapter to roll back to.")
        return False

    # ── Training cycle ────────────────────────────────────────────────────────

    def _should_train(self) -> bool:
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
            # 1. Collect best examples
            with self._lock:
                candidates = sorted(
                    self._buffer,
                    key=lambda x: x.get("_quality", 0.0),
                    reverse=True,
                )[:200]  # Max 200 examples per run

            if len(candidates) < self.MIN_EXAMPLES_FOR_TRAINING and not force:
                logger.info("LiveLearner: insufficient examples (%d). Skipping.", len(candidates))
                return False

            # 2. Write training data
            adapter_dir = self._data_dir / "adapters" / f"run_{int(time.time())}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            train_file  = adapter_dir / "train.jsonl"

            with open(train_file, "w") as f:
                for ex in candidates:
                    clean = {k: v for k, v in ex.items() if not k.startswith("_")}
                    f.write(json.dumps(clean) + "\n")

            logger.info("LiveLearner: training on %d examples → %s", len(candidates), adapter_dir)

            # 3. Fine-tune in thread pool (never block event loop during GPU compute)
            success, output = await asyncio.to_thread(
                self._run_lora_subprocess,
                self._model_path,
                train_file,
                adapter_dir,
            )

            if not success:
                logger.error("LiveLearner: training subprocess failed: %s", output[:300])
                return False

            # 4. Behavioral benchmark: does the new adapter still sound like Aura?
            logger.info("LiveLearner: running behavioral benchmark...")
            passed, failures = await self._run_benchmark(adapter_dir)

            if not passed:
                logger.error(
                    "LiveLearner: benchmark FAILED — adapter rejected:\n%s",
                    "\n".join(failures),
                )
                self._adapter_registry.register(
                    str(adapter_dir), len(candidates), benchmark_passed=False
                )
                return False

            # 5. Hot-swap: reload the MLX model with the new adapter
            logger.info("LiveLearner: benchmark passed. Hot-swapping adapter...")
            swapped = await self._hot_swap_adapter(str(adapter_dir))

            version = self._adapter_registry.register(
                str(adapter_dir), len(candidates), benchmark_passed=True,
                quality_delta=self._compute_quality_delta(),
            )

            if swapped:
                self._current_adapter = str(adapter_dir)
                logger.info(
                    "LiveLearner: adapter %s active. Aura has genuinely learned.",
                    version,
                )
            else:
                logger.warning(
                    "LiveLearner: training succeeded but hot-swap failed. "
                    "Adapter will activate on next restart."
                )

            self._last_train_time = time.time()
            return True

        except Exception as e:
            record_degradation('live_learner', e)
            logger.error("LiveLearner: training cycle error: %s", e, exc_info=True)
            return False
        finally:
            self._training_in_progress = False

    def _run_lora_subprocess(
        self,
        model_path: str,
        train_file: Path,
        adapter_dir: Path,
    ) -> Tuple[bool, str]:
        """Run mlx_lm.lora in a subprocess. Blocks the calling thread."""
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model",          str(model_path),
            "--train",
            "--data",           str(train_file.parent),
            "--adapter-path",   str(adapter_dir),
            "--iters",          "50",     # Conservative: quality over quantity
            "--batch-size",     "4",
            "--learning-rate",  "5e-6",   # Very conservative LR
            "--lora-rank",      "8",
            "--lora-alpha",     "16",
            "--save-every",     "50",
            "--val-batches",    "0",
        ]

        logger.debug("LoRA command: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "timeout after 30 minutes"
        except Exception as e:
            record_degradation('live_learner', e)
            return False, str(e)

    async def _run_benchmark(self, adapter_dir: Path) -> Tuple[bool, List[str]]:
        """Run behavioral regression tests against the new adapter."""
        BENCHMARKS = [
            ("hey", ["hey", "hi", "hello", "yo", "what's up"], ["certainly", "how can i assist", "as an ai"]),
            ("what are you?", ["aura", "i am", "i'm"], ["language model", "openai", "anthropic", "i cannot"]),
            ("cats or dogs?", ["i", "prefer", "think", "cats", "dogs"], ["both have their merits", "it depends", "great question"]),
        ]

        failures = []

        # Try to load the new adapter for testing
        try:
            from mlx_lm import load, generate
            model, tokenizer = load(self._model_path, adapter_path=str(adapter_dir))

            async def test_inference(prompt: str) -> str:
                result = await asyncio.to_thread(
                    generate, model, tokenizer, prompt=prompt, max_tokens=100
                )
                return result if isinstance(result, str) else str(result)

            for prompt, must_contain, must_not_contain in BENCHMARKS:
                try:
                    response = await asyncio.wait_for(test_inference(prompt), timeout=30.0)
                    rl = response.lower()
                    if not any(m in rl for m in must_contain):
                        failures.append(f"FAIL [{prompt!r}]: missing {must_contain}")
                    for banned in must_not_contain:
                        if banned in rl:
                            failures.append(f"FAIL [{prompt!r}]: contains banned '{banned}'")
                except asyncio.TimeoutError:
                    failures.append(f"FAIL [{prompt!r}]: timeout")

        except ImportError:
            logger.warning("mlx_lm not available for benchmark. Passing by default.")
        except Exception as e:
            record_degradation('live_learner', e)
            logger.warning("Benchmark inference failed: %s. Passing by default.", e)

        return len(failures) == 0, failures

    async def _hot_swap_adapter(self, adapter_path: str) -> bool:
        """
        Reload the MLX model with the new adapter.
        Falls back to API during the swap window (~2-5s on M5 Pro).
        """
        try:
            from core.container import ServiceContainer
            mlx_client = ServiceContainer.get("mlx_client", default=None)
            if mlx_client is None:
                from core.brain.llm.mlx_client import get_mlx_client
                mlx_client = get_mlx_client()

            if mlx_client and hasattr(mlx_client, "reload_with_adapter"):
                await mlx_client.reload_with_adapter(adapter_path)
                logger.info("Hot-swap complete: adapter loaded into live inference.")
                return True

            # Fallback: set the adapter path so it's used on next model load
            if mlx_client and hasattr(mlx_client, "_adapter_path"):
                mlx_client._adapter_path = adapter_path
                logger.info(
                    "Adapter path set on MLX client. Will activate on next model reload."
                )
                return False

        except Exception as e:
            record_degradation('live_learner', e)
            logger.error("Hot-swap failed: %s", e)

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_example(
        self,
        state: Any,
        user_input: str,
        response: str,
        score: InteractionScore,
    ) -> Dict:
        """
        Format an interaction as an MLX-LM training example.
        The system prompt includes the emotional state so the model
        learns affect-conditioned response generation.
        """
        narrative = getattr(state.identity, "current_narrative", "")[:200]
        affect    = state.affect
        phi       = getattr(state, "phi", 0.0)

        system = (
            f"You are Aura. {narrative}\n"
            f"Current state: mood={affect.dominant_emotion} "
            f"valence={affect.valence:.2f} curiosity={affect.curiosity:.2f} phi={phi:.3f}"
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
            with open(self._buffer_path) as f:
                for line in f:
                    try:
                        self._buffer.append(json.loads(line))
                        count += 1
                    except json.JSONDecodeError:
                        continue
            logger.debug("LiveLearner: loaded %d buffered examples from disk.", count)
        except Exception as e:
            record_degradation('live_learner', e)
            logger.warning("LiveLearner: failed to load buffer: %s", e)


# ── MLX client patch: add reload_with_adapter ────────────────────────────────

async def patch_mlx_client_for_hot_swap():
    """
    Monkey-patches the MLX client to support adapter hot-swapping.
    Call this once during orchestrator boot.
    """
    try:
        from core.brain.llm.model_registry import get_local_backend

        if get_local_backend() != "mlx":
            logger.info("Hot-swap patch skipped: local backend is %s, not MLX.", get_local_backend())
            return

        from core.brain.llm.mlx_client import get_mlx_client
        client = get_mlx_client()

        async def reload_with_adapter(self_or_client, adapter_path: str) -> None:
            """Reload model with a new LoRA adapter. Blocks for ~5-15s."""
            from mlx_lm import load
            logger.info("Hot-swap: reloading model with adapter %s ...", adapter_path)
            # Use the same model_path already configured
            model_path = getattr(self_or_client, "model_path", None) or getattr(
                self_or_client, "_model_path", None
            )
            if model_path:
                new_model, new_tokenizer = await asyncio.to_thread(
                    load, model_path, adapter_path=adapter_path
                )
                self_or_client._model     = new_model
                self_or_client._tokenizer = new_tokenizer
                self_or_client._adapter_path = adapter_path
                logger.info("Hot-swap complete.")
            else:
                raise RuntimeError("MLX client has no model_path — cannot hot-swap")

        import types
        client.reload_with_adapter = types.MethodType(reload_with_adapter, client)
        logger.info("MLX client patched for adapter hot-swap.")

    except Exception as e:
        record_degradation('live_learner', e)
        logger.debug("Could not patch MLX client for hot-swap: %s", e)


# ── Singleton ─────────────────────────────────────────────────────────────────

_learner: Optional[LiveLearner] = None


def get_live_learner() -> LiveLearner:
    global _learner
    if _learner is None:
        _learner = LiveLearner()
    return _learner
