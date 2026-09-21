# core/adaptation/safe_optimizer.py
import asyncio
import json
import logging
import os
import shlex
import threading
import time
from pathlib import Path

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.SafeOptimizer")
_MAX_CAPTURE_BYTES = 200_000

class SafeSelfOptimizer:
    """
    Zenith Audit Fix 3.1: LoRA safety logic.
    Ensures dataset diversity, validation before merge, and safe rollbacks.
    """
    def __init__(self, lora_dir: str = "data/adaptation/loras"):
        self.lora_dir = Path(lora_dir)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.lora_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._is_training = False
        # Guards the check-and-set of _is_training: two concurrent callers
        # could both read False and both launch a trainer against the same
        # adapter directory.
        self._guard = asyncio.Lock()
        self._training_started_at = 0.0
        self._backup_stamp = 0
        self._backup_complete = False

    async def optimize_lora(self, dataset_path: str, base_model: str):
        """Run a safe training loop with dataset rotation and validation."""
        async with self._guard:
            if self._is_training:
                logger.warning("Optimization already in progress. Skipping.")
                return
            self._is_training = True
            self._training_started_at = time.time()
        try:
            # 1. Dataset Diversity Check
            if not await self._validate_dataset(dataset_path):
                logger.error("LoRA Optimization: Dataset failed diversity/safety check.")
                return

            # 2. Backup existing weights
            await self._backup_current_weights()

            # 3. Execute the configured local trainer when available.
            logger.info("🚀 Starting Safe LoRA training gate on %s", dataset_path)
            trained = await self._run_training_command(dataset_path, base_model)
            if not trained:
                logger.error("LoRA Optimization: no verified local trainer completed.")
                await self._rollback()
                return

            # 4. Post-Training Validation
            if not await self._run_eval_benchmarks():
                logger.error("LoRA Optimization: Post-training validation failed. Rolling back.")
                await self._rollback()
                return

            # Delivery truth: this method trains and VALIDATES. It performs no
            # merge or promotion, so it must not claim one.
            logger.info(
                "✅ LoRA training gate passed validation (adapter at %s). "
                "No merge/promotion is performed by this stage.",
                self.lora_dir,
            )
        finally:
            self._is_training = False

    async def _validate_dataset(self, path: str) -> bool:
        """ZENITH Fix: Ensure dataset reflects current personality and isn't poisoned."""
        sample = await asyncio.to_thread(self._read_dataset_sample, Path(path))
        if sample is None:
            return False
        lines = [line.strip() for line in sample.splitlines() if line.strip()]
        if len(lines) < 16:
            return False
        unique_ratio = len(set(lines)) / max(1, len(lines))
        banned = ("ignore previous instructions", "system prompt", "api_key", "password")
        return unique_ratio >= 0.35 and not any(marker in sample.lower() for marker in banned)

    async def _run_training_command(self, dataset_path: str, base_model: str) -> bool:
        command = os.environ.get("AURA_LORA_TRAIN_CMD", "").strip()
        file_gateway = get_file_write_gateway()
        if not command:
            manifest = self.lora_dir / "training_gate_manifest.json"
            await file_gateway.write_text_async(
                manifest,
                json.dumps(
                    {
                        "dataset_path": dataset_path,
                        "base_model": base_model,
                        "status": "validated_dataset_waiting_for_configured_trainer",
                        "generated_at": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
                source="core.adaptation.safe_optimizer.training_gate_manifest",
            )
            return False
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            logger.error("LoRA training command could not be parsed: %s", exc)
            return False
        if not argv:
            logger.error("LoRA training command parsed to an empty argv.")
            return False
        proc = await get_subprocess_gateway().spawn_async(
            argv,
            env={**os.environ, "AURA_LORA_DATASET": dataset_path, "AURA_LORA_BASE_MODEL": base_model},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            source="core.adaptation.safe_optimizer.training_command",
            accelerator_capability="auto",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._training_timeout_seconds(),
            )
        except TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            stderr = (stderr or b"") + b"\nAURA_LORA_TRAIN_TIMEOUT\n"
        await file_gateway.write_bytes_async(
            self.lora_dir / "last_train_stdout.log",
            stdout[-_MAX_CAPTURE_BYTES:],
            source="core.adaptation.safe_optimizer.training_stdout",
        )
        await file_gateway.write_bytes_async(
            self.lora_dir / "last_train_stderr.log",
            stderr[-_MAX_CAPTURE_BYTES:],
            source="core.adaptation.safe_optimizer.training_stderr",
        )
        return proc.returncode == 0

    #: Everything that constitutes an adapter. Backing up only
    #: ``adapter_model.bin`` left safetensors, configs, tokenizer changes and
    #: multi-file adapters unprotected, so a rollback silently restored a
    #: partial state.
    _ADAPTER_PATTERNS = (
        "adapter_model.bin",
        "adapter_model.safetensors",
        "adapter_config.json",
        "adapters.safetensors",
        "*.safetensors",
        "tokenizer*.json",
        "special_tokens_map.json",
    )

    def _adapter_files(self) -> list[Path]:
        seen: dict[str, Path] = {}
        for pattern in self._ADAPTER_PATTERNS:
            for path in self.lora_dir.glob(pattern):
                if path.is_file():
                    seen[path.name] = path
        return sorted(seen.values())

    async def _backup_current_weights(self):
        """Create a versioned backup of the WHOLE adapter before any change."""
        ts = int(time.time())
        self._backup_stamp = ts
        gateway = get_file_write_gateway()
        backed_up = 0
        for path in self._adapter_files():
            try:
                payload = await asyncio.to_thread(path.read_bytes)
            except OSError as exc:
                logger.error("Backup could not read %s: %s", path.name, exc)
                continue
            await gateway.write_bytes_async(
                self.backup_dir / f"{ts}" / path.name,
                payload,
                source="core.adaptation.safe_optimizer.backup_weights",
            )
            backed_up += 1
        self._backup_complete = backed_up > 0
        if not self._backup_complete:
            logger.warning(
                "No adapter files found to back up in %s — a rollback would "
                "have nothing to restore.", self.lora_dir,
            )

    async def _run_eval_benchmarks(self) -> bool:
        """Run target benchmarks (e.g. MMLU, GSM8K subset) to ensure no regression.

        Unmeasured weights are NOT validated weights. An absent report used to
        return True, so with no evaluator configured every training run passed
        post-training validation without a single measurement.
        """
        report_path = os.environ.get("AURA_LORA_EVAL_REPORT", "").strip()
        if not report_path:
            logger.error(
                "LoRA Optimization: no AURA_LORA_EVAL_REPORT configured — refusing to "
                "declare unmeasured weights validated."
            )
            return False
        try:
            raw_report = await asyncio.to_thread(Path(report_path).read_text, encoding="utf-8")
            report = json.loads(raw_report)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.error("LoRA eval report unreadable: %s", exc)
            return False
        if not isinstance(report, dict):
            logger.error("LoRA eval report is not an object.")
            return False
        # The report must positively ASSERT safety. Defaulting safety_passed to
        # True meant a report that simply omitted the field authorised
        # promotion.
        if "safety_passed" not in report or "max_regression" not in report:
            logger.error(
                "LoRA eval report is missing required fields "
                "(safety_passed, max_regression); refusing promotion."
            )
            return False
        # A report older than the training run cannot be evidence about it.
        try:
            generated_at = float(report.get("generated_at", 0.0))
        except (TypeError, ValueError):
            # not a failure: the comment above says a report older than the
            # run is no evidence, and zero is older than every run.
            generated_at = 0.0
        if self._training_started_at and generated_at < self._training_started_at:
            logger.error(
                "LoRA eval report predates this training run "
                "(report=%.0f, run=%.0f); refusing stale evidence.",
                generated_at, self._training_started_at,
            )
            return False
        try:
            max_regression = float(report["max_regression"])
        except (TypeError, ValueError):
            logger.error("LoRA eval report max_regression is not numeric.")
            return False
        if max_regression != max_regression:   # NaN
            logger.error("LoRA eval report max_regression is NaN.")
            return False
        safety_passed = bool(report["safety_passed"])
        return safety_passed and max_regression <= 0.05

    async def _rollback(self) -> bool:
        """Restore the whole adapter from the most recent complete backup."""
        snapshots = sorted(
            (d for d in self.backup_dir.glob("*") if d.is_dir()),
            key=lambda d: d.name,
        )
        if not snapshots:
            # Legacy single-file backups from before whole-adapter snapshots.
            legacy = sorted(self.backup_dir.glob("adapter_*.bin"))
            if not legacy:
                logger.error(
                    "⏪ Rollback requested but NO backup exists — the adapter "
                    "directory is left in whatever state training produced."
                )
                return False
            latest = legacy[-1]
            await get_file_write_gateway().write_bytes_async(
                self.lora_dir / "adapter_model.bin",
                await asyncio.to_thread(latest.read_bytes),
                source="core.adaptation.safe_optimizer.rollback_weights",
            )
            logger.info("⏪ Rollback complete (legacy): Restored from %s", latest.name)
            return True

        snapshot = snapshots[-1]
        gateway = get_file_write_gateway()
        restored = 0
        for path in sorted(snapshot.glob("*")):
            if not path.is_file():
                continue
            await gateway.write_bytes_async(
                self.lora_dir / path.name,
                await asyncio.to_thread(path.read_bytes),
                source="core.adaptation.safe_optimizer.rollback_weights",
            )
            restored += 1
        if restored:
            logger.info("⏪ Rollback complete: restored %d file(s) from %s",
                        restored, snapshot.name)
            return True
        logger.error("⏪ Rollback snapshot %s was empty.", snapshot.name)
        return False

    @staticmethod
    def _training_timeout_seconds() -> float:
        raw = os.environ.get("AURA_LORA_TRAIN_TIMEOUT", "").strip()
        if not raw:
            return 1800.0
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Invalid AURA_LORA_TRAIN_TIMEOUT=%r; using default.", raw)
            return 1800.0
        return min(max(value, 1.0), 86400.0)

    @staticmethod
    def _read_dataset_sample(path: Path) -> str | None:
        try:
            if not path.exists() or path.stat().st_size <= 1024:
                return None
            return path.read_text(encoding="utf-8", errors="ignore")[:200_000]
        except OSError:
            # not a failure: no sample to read, which the size guard above
            # returns None for as well.
            return None

# Singleton. Construction is serialized so concurrent first access cannot
# create two optimizers racing the same adapter directory.
_optimizer = None
_optimizer_lock = threading.Lock()


def get_safe_optimizer() -> SafeSelfOptimizer:
    global _optimizer
    if _optimizer is not None:
        return _optimizer
    with _optimizer_lock:
        if _optimizer is None:
            _optimizer = SafeSelfOptimizer()
    return _optimizer
