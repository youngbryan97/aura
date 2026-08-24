"""core/adaptation/self_optimizer.py - MLX Self-Optimization Engine

Orchestrates on-device LoRA fine-tuning for Aura's internal Nucleus models.
This allows Aura to update her own weights based on captured experiences.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.learning.cortex_generation_upgrade import record_upgrade_candidate
from core.runtime import resource_psutil as psutil
from core.runtime.background_policy import background_activity_reason
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.SelfOptimizer")

class SelfOptimizer:
    """Manages the lifecycle of LoRA updates for local models."""
    
    def __init__(self, 
                 base_model_path: str,
                 dataset_path: str,
                 adapter_path: str = "data/adapters/adapters.safetensors",
                 python_path: str | None = None,
                 event_bus: Any | None = None):
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path)
        self.adapter_output = Path(adapter_path)
        self.python_path = python_path or sys.executable

        self.event_bus = event_bus
        
        self.adapter_output.parent.mkdir(parents=True, exist_ok=True)

        self._is_optimizing = False
        # Guards the check-and-set of _is_optimizing so two concurrent
        # optimize() callers cannot both pass the "already in progress" gate and
        # launch competing LoRA trainers against the same adapter path.
        self._guard_lock = asyncio.Lock()

    async def optimize(self, iters: int = 50, batch_size: int = 2) -> dict[str, Any]:
        """Runs a LoRA training cycle on the cortex model.
        
        This is a resource-intensive operation and should only run when idle or dreaming.
        """
        if os.getenv("AURA_SELF_OPTIMIZER_ENABLED", "").strip().lower() not in {"1", "true", "yes"}:
            return {"ok": False, "error": "self_optimizer_disabled_for_live_runtime"}

        if is_shutdown_requested():
            return {"ok": False, "error": "shutdown_requested"}

        reason = background_activity_reason(
            ServiceContainer.get("orchestrator", default=None),
            min_idle_seconds=float(os.getenv("AURA_SELF_OPTIMIZER_MIN_IDLE_S", "900") or 900),
            max_memory_percent=float(os.getenv("AURA_SELF_OPTIMIZER_MAX_RAM_PCT", "65") or 65),
            max_failure_pressure=float(os.getenv("AURA_SELF_OPTIMIZER_MAX_FAILURE_PRESSURE", "0.20") or 0.20),
            require_conversation_ready=True,
        )
        if reason:
            return {"ok": False, "error": f"background_deferred:{reason}"}

        mem = psutil.virtual_memory()
        if mem.available < 8 * 1024**3:  # <8 GB free -> abort
            logger.warning("🧠 Nucleus: Insufficient RAM for LoRA. Requires 8GB free.")
            return {"ok": False, "error": "insufficient RAM (need 8 GB free)"}
            
        if not self.dataset_path.exists():
            return {"ok": False, "error": f"Dataset missing: {self.dataset_path}"}
            
        # Check if we have enough data to bother (min 5 samples). Parse and
        # validate here so a single malformed JSONL line can neither crash the
        # cycle later nor let a mostly-empty/garbage file reach the trainer.
        raw_lines = self.dataset_path.read_text(encoding="utf-8").splitlines()
        data: list[Any] = []
        malformed = 0
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                malformed += 1
                continue
            if isinstance(entry, dict):
                data.append(entry)
            else:
                malformed += 1
        if malformed:
            logger.warning("🧠 Nucleus: Skipped %d malformed dataset line(s).", malformed)
        if len(data) < 5:
            return {"ok": False, "error": "Insufficient valid data in dataset for training"}

        # Atomic claim: hold the guard lock across the read-and-set so a race
        # between two callers (all preflight checks having passed) resolves to
        # exactly one active optimization. Only past this point does the
        # try/finally below own the reset of _is_optimizing.
        async with self._guard_lock:
            if self._is_optimizing:
                return {"ok": False, "error": "Optimization already in progress"}
            self._is_optimizing = True
        self._abort_requested = False # Reset abort flag for new optimization cycle
        logger.info("🧠 Nucleus: Starting self-optimization (LoRA) cycle...")
        
        if self.event_bus:
            get_task_tracker().create_task(self.event_bus.publish("core/optimizer/started", {
                "model": self.base_model_path.name,
                "iters": iters,
                "batch_size": batch_size
            }))
        
        temp_dir = None
        try:
            # 1. Prepare temp directory for training
            temp_dir = self.dataset_path.parent / "temp_train"
            temp_dir.mkdir(exist_ok=True)
            
            # Split into train/valid (80/20). `data` was already parsed and
            # validated above.
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            valid_data = data[split_idx:]
            
            train_file = temp_dir / "train.jsonl"
            valid_file = temp_dir / "valid.jsonl"
            
            await get_file_write_gateway().write_text_async(
                train_file,
                "".join(json.dumps(entry) + "\n" for entry in train_data),
                source="adaptation.self_optimizer.train_split",
            )
            await get_file_write_gateway().write_text_async(
                valid_file,
                "".join(json.dumps(entry) + "\n" for entry in valid_data),
                source="adaptation.self_optimizer.valid_split",
            )

            # 2. Construct training command
            log_file = self.adapter_output.parent / "last_training.log"
            cmd = [
                self.python_path, "-m", "mlx_lm", "lora",
                "--model", str(self.base_model_path),
                "--train",
                "--data", str(temp_dir),
                "--iters", str(iters),
                "--batch-size", str(batch_size),
                "--num-layers", "16",           # updated from --lora-layers
                "--grad-checkpoint",            # memory saver for large models
                "--adapter-path", str(self.adapter_output.parent),
                "--max-seq-length", "4096",
                "--steps-per-report", "5",      # Less chatty for 64GB
                "--save-every", str(iters) 
            ]
            
            start_time = time.time()
            train_result = await self._run_gateway_command(
                cmd,
                source="adaptation.self_optimizer.lora_train",
                abortable=True,
            )
            await get_file_write_gateway().write_text_async(
                log_file,
                train_result["log"],
                source="adaptation.self_optimizer.training_log",
            )
            duration = time.time() - start_time
            
            if train_result["returncode"] == 0:
                logger.info("🧠 Nucleus: Training complete. Fusing weights into new model...")
                from core.brain.llm.model_registry import get_fused_model_root

                fused_root = get_fused_model_root()
                fused_dir = fused_root / f"aura-self-optimized-{int(time.time())}"
                fuse_cmd = [
                    self.python_path, "-m", "mlx_lm.fuse",
                    "--model", str(self.base_model_path),
                    "--adapter-path", str(self.adapter_output.parent),
                    "--save-path", str(fused_dir)
                ]
                fuse_result = await self._run_gateway_command(
                    fuse_cmd,
                    source="adaptation.self_optimizer.lora_fuse",
                    abortable=False,
                )
                await get_file_write_gateway().append_text_async(
                    log_file,
                    "\n=== fuse ===\n" + fuse_result["log"],
                    source="adaptation.self_optimizer.training_log",
                )
                    
                fusion_ok = fuse_result["returncode"] == 0
                candidate_receipt: dict[str, Any] | None = None
                if fusion_ok:
                    logger.info("🧠 Nucleus: Fusion successful. Recording qualification candidate...")
                    candidate_receipt = await asyncio.to_thread(
                        record_upgrade_candidate,
                        candidate_model_path=fused_dir,
                        base_model_path=self.base_model_path,
                        tag="self-optimizer",
                        fused_model_dir=fused_root,
                        source="core.adaptation.self_optimizer",
                        metadata={
                            "adapter_path": str(self.adapter_output.parent),
                            "samples": len(data),
                            "iters": int(iters),
                            "batch_size": int(batch_size),
                        },
                    )
                else:
                    logger.error("❌ Nucleus: Fusion failed.")

                # Delivery truth: overall success requires BOTH training AND
                # fusion. The previous code returned ok=True (and published
                # status=success) whenever training succeeded, even when fusion
                # failed and produced no usable model.
                if self.event_bus:
                    get_task_tracker().create_task(self.event_bus.publish("core/optimizer/completed", {
                        "status": "candidate" if fusion_ok else "fusion_failed",
                        "duration": duration,
                        "samples": len(data),
                        "candidate_model": str(fused_dir) if fusion_ok else None,
                        "candidate_receipt": (
                            candidate_receipt.get("candidate_receipt_path")
                            if candidate_receipt is not None
                            else None
                        ),
                    }))
                return {
                    "ok": fusion_ok,
                    "duration": duration,
                    "adapter": str(self.adapter_output),
                    "samples": len(data),
                    "fused_model": str(fused_dir) if fusion_ok else None,
                    "candidate_receipt": (
                        candidate_receipt.get("candidate_receipt_path")
                        if candidate_receipt is not None
                        else None
                    ),
                    "error": None if fusion_ok else "fusion_failed",
                }
            else:
                # Read error from log file as stdout/stderr were redirected
                error_msg = train_result["log"].strip()[-500:] or "Unknown error (check logs)"
                
                if self.event_bus:
                    get_task_tracker().create_task(self.event_bus.publish("core/optimizer/completed", {
                        "status": "failed",
                        "error": error_msg
                    }))
                logger.error("❌ Nucleus: Self-optimization failed: %s", error_msg)
                return {"ok": False, "error": error_msg}
                
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError, json.JSONDecodeError, UnicodeError) as e:
            # Broadened: JSON/OS/Unicode/subprocess-shaped failures previously
            # escaped this narrow handler and crashed the optimizer.
            record_degradation('self_optimizer', e)
            logger.error("❌ Nucleus: Critical optimizer failure: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            if temp_dir:
                await self._cleanup_temp(temp_dir)
            self._is_optimizing = False
            try:
                import mlx.core as mx
                try:
                    from core.utils.gpu_sentinel import GPUPriority, get_gpu_sentinel
                    sentinel = get_gpu_sentinel()
                    if sentinel.acquire(priority=GPUPriority.REFLEX, timeout=5.0):
                        try:
                            if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                                mx.metal.clear_cache()
                            else:
                                mx.clear_cache()
                            logger.info("🧠 Nucleus: MLX cache cleared post-optimization.")
                        finally:
                            sentinel.release()
                except (ImportError, AttributeError, RuntimeError) as e:
                    record_degradation('self_optimizer', e)
                    logger.debug("[MLX] Cache clear skipped: %s", e)
            except ImportError as _e:
                logger.debug('Ignored ImportError in self_optimizer.py: %s', _e)
    async def _cleanup_temp(self, temp_dir: Path):
        """Removes the temporary training directory."""
        try:
            if await get_file_write_gateway().delete_path_async(
                temp_dir,
                recursive=True,
                source="core.adaptation.self_optimizer.cleanup_temp",
            ):
                logger.debug("🧠 Nucleus: Temporary training data cleaned.")
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as e:
            record_degradation('self_optimizer', e)
            logger.warning("Failed to cleanup optimizer temp dir: %s", e)

    async def _run_gateway_command(
        self,
        cmd: list[str],
        *,
        source: str,
        abortable: bool,
        log_limit: int = 2_000_000,
    ) -> dict[str, Any]:
        """Run a governed training command while capturing bounded logs."""
        process = await get_subprocess_gateway().spawn_async(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            source=source,
            accelerator_capability="auto",
        )
        chunks: list[str] = []
        total = 0

        def add_chunk(text: str) -> None:
            nonlocal total
            if not text:
                return
            chunks.append(text)
            total += len(text)
            while total > log_limit and chunks:
                removed = chunks.pop(0)
                total -= len(removed)

        stdout = process.stdout
        while process.returncode is None:
            if abortable and getattr(self, "_abort_requested", False):
                logger.warning("🧠 Nucleus: Memory critical! Terminating LoRA training...")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                break

            if stdout is not None:
                try:
                    raw = await asyncio.wait_for(stdout.readline(), timeout=2.0)
                except TimeoutError:
                    raw = b""
                if raw:
                    add_chunk(raw.decode("utf-8", errors="replace"))
                    continue

            if process.returncode is not None:
                break
            await asyncio.sleep(0.1)

        if stdout is not None:
            try:
                remaining = await asyncio.wait_for(stdout.read(), timeout=1.0)
            except TimeoutError:
                remaining = b""
            if remaining:
                add_chunk(remaining.decode("utf-8", errors="replace"))

        return {
            "returncode": int(process.returncode if process.returncode is not None else -1),
            "log": "".join(chunks),
        }



# Integration Singleton
_optimizer_instance = None
# Serialize first-access construction so concurrent callers cannot build two
# SelfOptimizer instances (each with its own event-bus binding and training
# owner) racing against the same adapter path.
_optimizer_lock = threading.Lock()

def get_self_optimizer() -> SelfOptimizer:
    global _optimizer_instance
    if _optimizer_instance is not None:
        return _optimizer_instance
    with _optimizer_lock:
        if _optimizer_instance is not None:
            return _optimizer_instance
        from core.brain.llm.model_registry import BASE_DIR, get_adapter_path, get_model_path
        from core.event_bus import get_event_bus

        base_path = get_model_path()
        dataset = BASE_DIR / "data" / "synthetic_training" / "lora_dataset.jsonl"
        adapter = get_adapter_path() / "adapters.safetensors"
        
        _optimizer_instance = SelfOptimizer(
            base_model_path=str(base_path),
            dataset_path=str(dataset),
            adapter_path=str(adapter),
            event_bus=get_event_bus()
        )
    return _optimizer_instance
