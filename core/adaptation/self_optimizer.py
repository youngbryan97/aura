"""core/adaptation/self_optimizer.py - MLX Self-Optimization Engine

Orchestrates on-device LoRA fine-tuning for Aura's internal Nucleus models.
This allows Aura to update her own weights based on captured experiences.
"""

import os
import json
import logging
import asyncio
import time
import sys
import psutil
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("Aura.SelfOptimizer")

class SelfOptimizer:
    """Manages the lifecycle of LoRA updates for local models."""
    
    def __init__(self, 
                 base_model_path: str,
                 dataset_path: str,
                 adapter_path: str = "data/adapters/adapters.safetensors",
                 python_path: Optional[str] = None,
                 event_bus: Optional[Any] = None):
        self.base_model_path = Path(base_model_path)
        self.dataset_path = Path(dataset_path)
        self.adapter_output = Path(adapter_path)
        self.python_path = python_path or sys.executable

        self.event_bus = event_bus
        
        self.adapter_output.parent.mkdir(parents=True, exist_ok=True)
        
        self._is_optimizing = False

    async def optimize(self, iters: int = 50, batch_size: int = 2) -> Dict[str, Any]:
        """Runs a LoRA training cycle on the cortex model.
        
        This is a resource-intensive operation and should only run when idle or dreaming.
        """
        if self._is_optimizing:
            return {"ok": False, "error": "Optimization already in progress"}
            
        mem = psutil.virtual_memory()
        if mem.available < 8 * 1024**3:  # <8 GB free -> abort
            logger.warning("🧠 Nucleus: Insufficient RAM for LoRA. Requires 8GB free.")
            return {"ok": False, "error": "insufficient RAM (need 8 GB free)"}
            
        if not self.dataset_path.exists():
            return {"ok": False, "error": f"Dataset missing: {self.dataset_path}"}
            
        # Check if we have enough data to bother (min 5 samples)
        with open(self.dataset_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 5:
                return {"ok": False, "error": "Insufficient data in dataset for training"}

        self._is_optimizing = True
        self._abort_requested = False # Reset abort flag for new optimization cycle
        logger.info("🧠 Nucleus: Starting self-optimization (LoRA) cycle...")
        
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish("core/optimizer/started", {
                "model": self.base_model_path.name,
                "iters": iters,
                "batch_size": batch_size
            }))
        
        temp_dir = None
        try:
            # 1. Prepare temp directory for training
            temp_dir = self.dataset_path.parent / "temp_train"
            temp_dir.mkdir(exist_ok=True)
            
            # Split into train/valid (80/20)
            data = [json.loads(line) for line in lines]
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            valid_data = data[split_idx:]
            
            train_file = temp_dir / "train.jsonl"
            valid_file = temp_dir / "valid.jsonl"
            
            with open(train_file, "w") as f:
                for entry in train_data:
                    f.write(json.dumps(entry) + "\n")
            with open(valid_file, "w") as f:
                for entry in valid_data:
                    f.write(json.dumps(entry) + "\n")

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
            with open(log_file, "w") as out:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=out,
                    stderr=out
                )
                
                # Poll for abort signal while training
                while process.returncode is None:
                    if getattr(self, "_abort_requested", False):
                        logger.warning("🧠 Nucleus: Memory critical! Terminating LoRA training...")
                        process.terminate()
                        try: await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError: process.kill()
                        break
                    await asyncio.sleep(2.0)
                
                await process.wait()
            duration = time.time() - start_time
            
            if process.returncode == 0:
                if self.event_bus:
                    asyncio.create_task(self.event_bus.publish("core/optimizer/completed", {
                        "status": "success",
                        "duration": duration,
                        "samples": len(data)
                    }))
                return {
                    "ok": True, 
                    "duration": duration, 
                    "adapter": str(self.adapter_output),
                    "samples": len(data)
                }
            else:
                # Read error from log file as stdout/stderr were redirected
                error_msg = "Unknown error (check logs)"
                try:
                    with open(log_file, "r") as f:
                        error_msg = f.read().strip()[-500:] # Get last 500 chars
                except Exception as _e:
                    logger.debug('Ignored Exception in self_optimizer.py: %s', _e)
                
                if self.event_bus:
                    asyncio.create_task(self.event_bus.publish("core/optimizer/completed", {
                        "status": "failed",
                        "error": error_msg
                    }))
                logger.error("❌ Nucleus: Self-optimization failed: %s", error_msg)
                return {"ok": False, "error": error_msg}
                
        except Exception as e:
            logger.error("❌ Nucleus: Critical optimizer failure: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            if temp_dir:
                await self._cleanup_temp(temp_dir)
            self._is_optimizing = False
            try:
                import mlx.core as mx
                try:
                    from core.utils.gpu_sentinel import get_gpu_sentinel, GPUPriority
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
                except Exception as e:
                    logger.debug(f"[MLX] Cache clear skipped: {e}")
            except ImportError as _e:
                logger.debug('Ignored ImportError in self_optimizer.py: %s', _e)
    async def _cleanup_temp(self, temp_dir: Path):
        """Removes the temporary training directory."""
        try:
            if temp_dir and temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug("🧠 Nucleus: Temporary training data cleaned.")
        except Exception as e:
            logger.warning(f"Failed to cleanup optimizer temp dir: {e}")



# Integration Singleton
_optimizer_instance = None

def get_self_optimizer() -> SelfOptimizer:
    global _optimizer_instance
    if _optimizer_instance is None:
        from core.brain.llm.model_registry import get_model_path, get_adapter_path, BASE_DIR
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
