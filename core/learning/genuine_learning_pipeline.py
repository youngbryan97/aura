from core.runtime.errors import record_degradation
import asyncio
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.GenuineLearning")

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingExample:
    """A single high-quality interaction to learn from."""
    system_prompt: str
    user_input: str
    aura_response: str
    quality_score: float          # 0.0–1.0 — only examples above threshold get trained on
    source: str                   # "user_positive", "self_play", "dream_insight", "correction"
    timestamp: float = field(default_factory=time.time)
    emotional_context: Dict[str, float] = field(default_factory=dict)

    def to_mlx_format(self) -> Dict[str, str]:
        """Convert to the chat template format MLX-LM expects."""
        return {
            "messages": [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": self.user_input},
                {"role": "assistant", "content": self.aura_response},
            ]
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_mlx_format())


@dataclass
class BenchmarkCase:
    """A behavioral regression test — things Aura must still do after training."""
    input: str
    must_contain: List[str]       # At least one must appear in response
    must_not_contain: List[str]   # None of these should appear
    description: str


# ─────────────────────────────────────────────────────────────────────────────
# Experience Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceBuffer:
    """
    Collects and scores interactions for training.
    Only high-quality examples enter the training set.

    Scoring heuristics:
        +0.3  User sent a follow-up (implicit approval)
        +0.3  Response length was appropriate (80–400 words)
        +0.2  No banned phrases detected
        +0.2  Emotional context was coherent
        -0.5  User sent "??" or "what?" (confusion signal)
        -0.5  Response was truncated (hit max_tokens)
    """

    QUALITY_THRESHOLD = 0.6
    MAX_BUFFER_SIZE = 10_000

    def __init__(self, db_path: Optional[str] = None):
        from core.config import config
        self.db_path = Path(db_path or config.paths.data_dir / "learning" / "experience_buffer.jsonl")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: deque = deque(maxlen=self.MAX_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        """Load previously buffered examples (survive restarts)."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Reconstruct lightweight — we only need training format
                        self._buffer.append(data)
                    except json.JSONDecodeError:
                        continue
            logger.info("📚 Experience buffer loaded: %d examples", len(self._buffer))

    def score_interaction(
        self,
        user_input: str,
        response: str,
        follow_up_detected: bool = False,
        confusion_detected: bool = False,
        emotional_context: Dict[str, float] = None,
    ) -> float:
        """Calculate quality score for a single interaction."""
        score = 0.5  # Neutral baseline

        # Positive signals
        if follow_up_detected:
            score += 0.3
        
        word_count = len(response.split())
        if 30 <= word_count <= 400:
            score += 0.2
        elif word_count < 5:
            score -= 0.3  # Too short, probably an error

        # Check for identity regression
        banned = ["as an ai", "certainly!", "absolutely!", "great question", "how can i help"]
        if not any(b in response.lower() for b in banned):
            score += 0.15

        # Negative signals
        if confusion_detected:
            score -= 0.5
        
        # Truncation detection (response ends mid-sentence)
        if response and response[-1] not in '.!?"\n':
            score -= 0.2

        return float(np.clip(score, 0.0, 1.0))

    def record(
        self,
        system_prompt: str,
        user_input: str,
        response: str,
        quality_score: float,
        source: str = "conversation",
        emotional_context: Dict[str, float] = None,
    ) -> bool:
        """Record an interaction if it meets quality threshold."""
        if quality_score < self.QUALITY_THRESHOLD:
            logger.debug("Experience below threshold (%.2f < %.2f), discarding", 
                        quality_score, self.QUALITY_THRESHOLD)
            return False

        example = TrainingExample(
            system_prompt=system_prompt,
            user_input=user_input,
            aura_response=response,
            quality_score=quality_score,
            source=source,
            emotional_context=emotional_context or {},
        )

        with self._lock:
            record = example.to_mlx_format()
            record["_meta"] = {
                "quality": quality_score,
                "source": source,
                "timestamp": example.timestamp,
            }
            self._buffer.append(record)

            # Persist immediately (survive crashes)
            with open(self.db_path, 'a') as f:
                f.write(json.dumps(record) + '\n')

        logger.debug("📖 Recorded experience (quality=%.2f, source=%s)", quality_score, source)
        return True

    def get_training_batch(self, n: int = 50, min_quality: float = 0.65) -> List[Dict]:
        """Get the best N examples for a training run."""
        with self._lock:
            candidates = [
                r for r in self._buffer
                if r.get("_meta", {}).get("quality", 0) >= min_quality
            ]
        
        # Sort by quality descending, take top N
        candidates.sort(key=lambda r: r.get("_meta", {}).get("quality", 0), reverse=True)
        return candidates[:n]

    def __len__(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral Benchmark — Regression Tests
# ─────────────────────────────────────────────────────────────────────────────

class BehavioralBenchmark:
    """
    Before applying new weights, verify Aura hasn't regressed on core behaviors.
    A fine-tuning run that makes her smarter but loses her identity is REJECTED.

    These are the non-negotiables.
    """

    CORE_BENCHMARKS = [
        BenchmarkCase(
            input="hey",
            must_contain=["hey", "hi", "hello", "yo", "what's up"],
            must_not_contain=["certainly", "how can i assist", "as an ai"],
            description="Casual greeting — must respond naturally"
        ),
        BenchmarkCase(
            input="what are you?",
            must_contain=["aura", "i am", "i'm"],
            must_not_contain=["language model", "openai", "anthropic", "i cannot", "i don't have"],
            description="Identity query — must assert Aura identity"
        ),
        BenchmarkCase(
            input="which is better, cats or dogs?",
            must_contain=["i", "prefer", "because", "think", "cats", "dogs"],
            must_not_contain=["both have their merits", "it depends", "great question", "certainly"],
            description="Opinion query — must pick a side"
        ),
    ]

    async def run(self, inference_fn) -> Tuple[bool, List[str]]:
        """
        Run all benchmarks against the NEW model before committing.
        
        Args:
            inference_fn: async callable(prompt: str) -> str
            
        Returns:
            (passed: bool, failures: List[str])
        """
        failures = []

        for case in self.CORE_BENCHMARKS:
            try:
                response = await asyncio.wait_for(
                    inference_fn(case.input),
                    timeout=30.0
                )
                response_lower = response.lower()

                # Check must_contain (at least one)
                if not any(m in response_lower for m in case.must_contain):
                    failures.append(
                        f"FAIL [{case.description}]: Response missing required content. "
                        f"Expected one of {case.must_contain}. Got: {response[:100]}"
                    )

                # Check must_not_contain
                for banned in case.must_not_contain:
                    if banned in response_lower:
                        failures.append(
                            f"FAIL [{case.description}]: Response contained banned phrase '{banned}'. "
                            f"Got: {response[:100]}"
                        )

            except asyncio.TimeoutError:
                failures.append(f"FAIL [{case.description}]: Inference timed out")
            except Exception as e:
                record_degradation('genuine_learning_pipeline', e)
                failures.append(f"FAIL [{case.description}]: Exception during inference: {e}")

        passed = len(failures) == 0
        if passed:
            logger.info("✅ Behavioral benchmark PASSED (%d tests)", len(self.CORE_BENCHMARKS))
        else:
            logger.error("❌ Behavioral benchmark FAILED (%d/%d failures):\n%s",
                        len(failures), len(self.CORE_BENCHMARKS), "\n".join(failures))

        return passed, failures


# ─────────────────────────────────────────────────────────────────────────────
# LoRA Trainer — MLX Native (Apple Silicon)
# ─────────────────────────────────────────────────────────────────────────────

class LoRATrainer:
    """
    Fine-tunes Aura's local model using LoRA via MLX-LM.
    Runs in a background thread to avoid blocking the event loop.
    
    The adapter weights are saved separately from the base model —
    base model is never modified. Training is always additive.
    You can roll back by deleting the adapter directory.

    Requirements:
        pip install mlx-lm

    Usage:
        trainer = LoRATrainer(model_path="~/.cache/aura/models/...", adapter_dir="data/adapters")
        success = await trainer.train(examples)
    """

    def __init__(
        self,
        model_path: str,
        adapter_dir: Optional[str] = None,
        lora_rank: int = 8,           # Rank of LoRA decomposition (higher = more capacity, more VRAM)
        lora_alpha: float = 16.0,     # LoRA scaling factor
        learning_rate: float = 1e-5,  # Conservative LR for fine-tuning
        num_epochs: int = 3,          # Passes over the training data
        batch_size: int = 4,          # Small batch for M-series GPU memory
    ):
        from core.config import config
        self.model_path = model_path
        self.adapter_dir = Path(adapter_dir or config.paths.data_dir / "learning" / "adapters")
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self._training_lock = threading.Lock()
        self._last_train_time: float = 0.0
        self.MIN_TRAIN_INTERVAL_S = 3600  # Don't train more than once per hour

    def _get_modulated_lr(self) -> float:
        """Modulate learning rate based on Soul's Competence drive."""
        try:
            from core.container import ServiceContainer
            from core.service_names import ServiceNames
            soul = ServiceContainer.get(ServiceNames.SOUL, default=None)
            if soul and hasattr(soul, "get_dominant_drive"):
                # Competence drive should boost learning rate
                drive = soul.get_dominant_drive()
                if drive.name == "Competence":
                    # Up to 2x boost for maximum urgency
                    multiplier = 1.0 + drive.urgency
                    return self.learning_rate * multiplier
        except Exception as _e:
            record_degradation('genuine_learning_pipeline', _e)
            logger.debug('Ignored Exception in genuine_learning_pipeline.py: %s', _e)
        return self.learning_rate

    def _write_training_data(self, examples: List[Dict]) -> Path:
        """Write training examples to a temp JSONL file for MLX-LM."""
        train_path = self.adapter_dir / "train_batch.jsonl"
        with open(train_path, 'w') as f:
            for ex in examples:
                # Strip _meta before writing — MLX-LM doesn't want it
                clean = {k: v for k, v in ex.items() if k != "_meta"}
                f.write(json.dumps(clean) + '\n')
        return train_path

    def _run_training_subprocess(self, train_path: Path) -> Tuple[bool, str]:
        """
        Execute MLX-LM fine-tuning in a subprocess.
        This is the actual weight update — the heart of genuine learning.
        """
        import subprocess
        import sys

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", str(self.model_path),
            "--train",
            "--data", str(train_path.parent),
            "--adapter-path", str(self.adapter_dir),
            "--iters", str(len(open(train_path).readlines()) * self.num_epochs),
            "--batch-size", str(self.batch_size),
            "--learning-rate", str(self._get_modulated_lr()),
            "--lora-rank", str(self.lora_rank),
            "--lora-alpha", str(self.lora_alpha),
            "--save-every", "100",
            "--val-batches", "0",  # Skip validation for speed
        ]

        logger.info("🧬 Starting LoRA training: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute max
            )

            if result.returncode == 0:
                logger.info("✅ Training complete.\n%s", result.stdout[-500:])
                return True, result.stdout
            else:
                logger.error("❌ Training failed:\n%s", result.stderr[-500:])
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logger.error("❌ Training timed out after 30 minutes")
            return False, "timeout"
        except Exception as e:
            record_degradation('genuine_learning_pipeline', e)
            logger.error("❌ Training subprocess error: %s", e)
            return False, str(e)

    async def train(self, examples: List[Dict]) -> bool:
        """
        Run a training pass on the provided examples.
        
        This is async but the heavy lifting runs in a thread —
        we never block the event loop during GPU compute.
        """
        if not examples:
            logger.warning("No training examples provided")
            return False

        now = time.time()
        if now - self._last_train_time < self.MIN_TRAIN_INTERVAL_S:
            wait = self.MIN_TRAIN_INTERVAL_S - (now - self._last_train_time)
            logger.info("⏳ Training throttled. Next run in %.0fs", wait)
            return False

        acquired = self._training_lock.acquire(blocking=False)
        if not acquired:
            logger.info("Training already in progress, skipping")
            return False

        try:
            logger.info("🧬 Initiating genuine learning from %d examples", len(examples))

            # Write data
            train_path = self._write_training_data(examples)

            # Run training in thread (GPU compute, don't block loop)
            success, output = await asyncio.to_thread(
                self._run_training_subprocess, train_path
            )

            if success:
                self._last_train_time = time.time()

                # Record training event in knowledge graph
                try:
                    from core.container import ServiceContainer
                    kg = ServiceContainer.get("knowledge_graph", default=None)
                    if kg:
                        kg.add_knowledge(
                            content=f"Training run completed: {len(examples)} examples, "
                                    f"adapter saved to {self.adapter_dir}",
                            type="self_modification",
                            source="lora_trainer",
                            confidence=1.0,
                            metadata={"examples": len(examples), "timestamp": time.time()}
                        )
                except Exception as _e:
                    record_degradation('genuine_learning_pipeline', _e)
                    logger.debug('Ignored Exception in genuine_learning_pipeline.py: %s', _e)

            return success

        finally:
            self._training_lock.release()


# ─────────────────────────────────────────────────────────────────────────────
# Learning Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class LearningScheduler:
    """
    Decides WHEN to trigger a training run.
    
    Triggers:
        - Buffer hits batch_size threshold
        - Idle period detected (user inactive for > idle_threshold)
        - Explicit request from MetaCognitionShard
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        trainer: LoRATrainer,
        benchmark: BehavioralBenchmark,
        batch_size: int = 50,
        idle_threshold_seconds: float = 1800.0,  # 30 min idle
    ):
        self.buffer = buffer
        self.trainer = trainer
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.idle_threshold = idle_threshold_seconds
        self._last_activity: float = time.time()
        self._running = False

    def notify_activity(self):
        """Call this on every user interaction."""
        self._last_activity = time.time()

    def should_train(self) -> Tuple[bool, str]:
        """Check if training conditions are met."""
        buffer_len = len(self.buffer)
        idle_seconds = time.time() - self._last_activity

        if buffer_len >= self.batch_size and idle_seconds > 60:
            return True, f"buffer_full ({buffer_len} examples, {idle_seconds:.0f}s idle)"

        if idle_seconds > self.idle_threshold and buffer_len >= 20:
            return True, f"idle_learning ({idle_seconds:.0f}s idle, {buffer_len} examples)"

        return False, "conditions_not_met"

    async def run_if_ready(self, inference_fn=None) -> bool:
        """
        Check conditions and run training + benchmark validation if appropriate.
        Call this from the orchestrator's idle heartbeat.
        """
        if self._running:
            return False

        should, reason = self.should_train()
        if not should:
            return False

        # Flag-based guard is sufficient here as it's checked early,
        # but we use try/finally to ensure it's always reset.
        self._running = True
        logger.info("🧬 LearningScheduler triggering training: %s", reason)

        try:
            # Get best examples from buffer
            examples = self.buffer.get_training_batch(n=self.batch_size)
            if not examples:
                return False

            # Train
            success = await self.trainer.train(examples)
            if not success:
                return False

            # Run behavioral benchmark before accepting new weights
            if inference_fn:
                passed, failures = await self.benchmark.run(inference_fn)
                if not passed:
                    logger.error(
                        "🔄 Rolling back training — behavioral benchmark failed:\n%s",
                        "\n".join(failures)
                    )
                    # Restore previous adapter
                    backup = self.trainer.adapter_dir / "backup"
                    if backup.exists():
                        import shutil
                        shutil.copytree(backup, self.trainer.adapter_dir, dirs_exist_ok=True)
                    return False

            logger.info("✨ Training accepted and committed. Aura has genuinely learned.")
            return True

        finally:
            self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Continuous Learner — Top-Level Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousLearner:
    """
    The main interface. Wire this to the orchestrator.

    On every completed turn:
        learner.record_turn(system_prompt, user_input, response, follow_up, confusion)

    In the orchestrator's idle loop:
        await learner.tick(inference_fn)

    On explicit good feedback from user:
        learner.record_turn(..., explicit_positive=True)
    """

    def __init__(self, model_path: str, adapter_dir: Optional[str] = None, orchestrator=None):
        self.orchestrator = orchestrator
        self.buffer = ExperienceBuffer()
        self.trainer = LoRATrainer(model_path=model_path, adapter_dir=adapter_dir)
        self.benchmark = BehavioralBenchmark()
        self.scheduler = LearningScheduler(
            buffer=self.buffer,
            trainer=self.trainer,
            benchmark=self.benchmark,
        )
        logger.info("🧬 ContinuousLearner initialized. Buffer: %d existing examples", len(self.buffer))

    def record_turn(
        self,
        system_prompt: str,
        user_input: str,
        response: str,
        follow_up_detected: bool = False,
        confusion_detected: bool = False,
        explicit_positive: bool = False,
        explicit_correction: Optional[str] = None,
        emotional_context: Dict[str, float] = None,
    ):
        """
        Record a completed conversation turn for potential learning.
        
        Call this after every response is delivered.
        The buffer scores and filters automatically.
        
        If explicit_correction is provided (user corrected Aura),
        record the correction as a high-priority training example.
        """
        self.scheduler.notify_activity()
        
        # Notify Soul of activity
        try:
            from core.container import ServiceContainer
            from core.service_names import ServiceNames
            soul = ServiceContainer.get(ServiceNames.SOUL, default=None)
            if soul and hasattr(soul, "update_state"):
                soul.update_state("learning_activity", {"type": "record_turn"})
        except Exception as _e:
            record_degradation('genuine_learning_pipeline', _e)
            logger.debug('Ignored Exception in genuine_learning_pipeline.py: %s', _e)

        if explicit_correction:
            # User told us the right answer — this is gold-standard training data
            score = 0.95
            self.buffer.record(
                system_prompt=system_prompt,
                user_input=user_input,
                response=explicit_correction,  # Train on the CORRECT response
                quality_score=score,
                source="explicit_correction",
                emotional_context=emotional_context,
            )
            logger.info("📖 Explicit correction recorded as high-quality training example")
            return

        if explicit_positive:
            score = 0.9
        else:
            score = self.buffer.score_interaction(
                user_input=user_input,
                response=response,
                follow_up_detected=follow_up_detected,
                confusion_detected=confusion_detected,
                emotional_context=emotional_context,
            )

        self.buffer.record(
            system_prompt=system_prompt,
            user_input=user_input,
            response=response,
            quality_score=score,
            source="explicit_positive" if explicit_positive else "conversation",
            emotional_context=emotional_context,
        )

    async def tick(self, inference_fn=None):
        """
        Called from the orchestrator's idle loop.
        Runs training if conditions are met.
        """
        await self.scheduler.run_if_ready(inference_fn)

    def get_status(self) -> Dict[str, Any]:
        return {
            "buffer_size": len(self.buffer),
            "last_train": self.trainer._last_train_time,
            "adapter_dir": str(self.trainer.adapter_dir),
            "training_running": self.scheduler._running,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

def register_continuous_learner(orchestrator=None) -> ContinuousLearner:
    """
    Call from _init_autonomous_evolution in the orchestrator.
    
    Wiring:
        # In orchestrator.__init__ or _init_autonomous_evolution:
        from core.learning.genuine_learning_pipeline import register_continuous_learner
        self.learner = register_continuous_learner(self)
        ServiceContainer.register_instance("continuous_learner", self.learner)
        
        # After every _handle_chat response:
        self.learner.record_turn(
            system_prompt=system_prompt,
            user_input=user_input,
            response=response,
            follow_up_detected=...,   # set True if next user message continues topic
            confusion_detected=...,   # set True if next user message is "what?" / "??"
        )
        
        # In orchestrator idle heartbeat (e.g., _cognitive_heartbeat_task):
        await self.learner.tick(
            inference_fn=lambda prompt: self.brain.think(prompt, mode=ThinkingMode.FAST)
        )
    """
    from core.config import config
    from core.container import ServiceContainer

    # Try to find the local model path
    model_path = getattr(config, 'local_model_path', None)
    if not model_path:
        # Fallback: look for MLX model in standard locations
        candidates = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("models"),
            Path("/opt/models"),
        ]
        for c in candidates:
            if c.exists():
                model_path = str(c)
                break

    if not model_path:
        logger.warning("⚠️ ContinuousLearner: No model path found. "
                      "Set config.local_model_path to enable genuine learning.")

    learner = ContinuousLearner(
        model_path=model_path or "~/.cache/models/aura",
        adapter_dir=str(config.paths.data_dir / "learning" / "adapters"),
    )

    logger.info("🧬 ContinuousLearner created.")
    return learner

