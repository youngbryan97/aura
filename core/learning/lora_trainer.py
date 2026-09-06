"""core/learning/lora_trainer.py
Coordinates fine-tuning of local models via MLX LoRA scripts.
"""
import asyncio
import logging
import sys
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

from core.config import get_config
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Learning.LoraTrainer")

_LORA_TRAINING_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class LoraTrainer:
    """Invokes fine-tuning adapters locally under memory-aware profiles."""

    def __init__(self):
        self.config = get_config()

    async def train_adapter(
        self,
        dataset_path: str,
        output_path: str,
        *,
        model_path: str | None = None,
        iters: int = 100,
        batch_size: int = 2,
        num_layers: int = 8,
        fine_tune_type: str = "lora",
        timeout: float = 1800.0,  # noqa: ASYNC109 - delegated to the subprocess gateway.
    ) -> dict[str, Any]:
        """Execute a local mlx_lm.lora fine-tune if resources permit.

        ``dataset_path`` is a DIRECTORY holding ``{train,valid}.jsonl`` and
        ``output_path`` is the adapter DIRECTORY (mlx-lm ``--adapter-path``).
        ``model_path`` overrides the configured cortex (e.g. a smaller local
        model for a fast, reproducible sleep-consolidation cycle).
        """
        model = model_path or self.config.llm.local_cortex_path
        if not model or not await asyncio.to_thread(Path(model).exists):
            return {"status": "skipped", "reason": "Active MLX model path not configured"}

        # MLX LoRA script invocation command (mlx-lm 0.31.x: --adapter-path is a
        # directory; --data is a directory of {train,valid,test}.jsonl).
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", model,
            "--data", dataset_path,
            "--train",
            "--fine-tune-type", fine_tune_type,
            "--num-layers", str(int(num_layers)),
            "--iters", str(int(iters)),
            "--batch-size", str(int(batch_size)),
            "--adapter-path", output_path,
        ]

        logger.info("Initiating local model parameter adaptation: %s", " ".join(cmd))
        # One fine-tune at a time. Nothing stopped two of these starting
        # together, and two mlx_lm.lora subprocesses on one machine do not
        # halve each other's speed — they compete for the same wired memory
        # the resident model already holds. The queueing bound is the run's
        # own timeout: a job unwilling to spend that long running is
        # unwilling to spend it waiting.
        from core.runtime.who_gets_it_next import GaveUp, claim

        try:
            async with claim("training", "lora_trainer.train_adapter", seconds=timeout):
                return await self._run_the_finetune(cmd, output_path, timeout=timeout)
        except GaveUp as exc:
            return {
                "status": "skipped",
                "reason": f"another fine-tune held the GPU: {exc}",
            }

    async def _run_the_finetune(
        self,
        cmd: list[str],
        output_path: str,
        *,
        timeout: float,  # noqa: ASYNC109 - delegated to the subprocess gateway.
    ) -> dict[str, Any]:
        try:
            res = await get_subprocess_gateway().run_async(
                cmd,
                capture_output=True,
                timeout=timeout,
                offline_tooling=True,
                source="training_tooling:lora_trainer",
                accelerator_capability="auto",
            )
            if res.returncode == 0:
                # Adapter creation is not loop closure. Only the atomic
                # train→fuse→verify→publish pipeline may commit the public CRSM
                # consumed marker, after the active model points at its output.
                return {
                    "status": "success",
                    "adapter_path": output_path,
                    "stdout": res.stdout[:1000]
                }
            return {"status": "failed", "error": res.stderr}
        except _LORA_TRAINING_ERRORS as e:
            record_degradation("learning.lora_trainer", e)
            logger.error("LoRA fine-tuning invocation failed: %s", e)
            return {"status": "failed", "error": str(e)}
