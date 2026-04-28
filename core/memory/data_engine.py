"""core/memory/data_engine.py
──────────────────────────
Hard-example mining for the self-improvement loop.

Analyses execution traces and persists "hard examples" — cases where
the agent failed, timed out, or behaved unexpectedly — so they can be
used for fine-tuning or curriculum learning.

Changes from previous version:
  - Removed `print(f"DEBUG: Mining Hard Example: ...")` in production code
  - Replaced bare open()+write with an atomic write (write to .tmp, rename)
    to prevent corruption if the process dies mid-write
  - Added `max_examples` cap to prevent unbounded file growth
  - Added type annotations
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Kernel.DataEngine")

_DEFAULT_DATASET = "data/training/hard_examples.json"
_MAX_EXAMPLES    = 10_000   # cap file growth


class DataEngine:
    """Mines "hard examples" from execution traces for later fine-tuning.

    A trace is considered "hard" when:
      - Its outcome dict has ``ok=False``
      - Its outcome string contains "fail" or "error"
      - Its latency (``cost`` field) exceeds ``latency_threshold_s``
    """

    def __init__(
        self,
        dataset_file: str = None,
        latency_threshold_s: float = 10.0,
        max_examples: int = _MAX_EXAMPLES,
    ) -> None:
        if dataset_file is None:
            from core.config import config
            self.dataset_file = config.paths.data_dir / "training" / "hard_examples.json"
        else:
            self.dataset_file = Path(dataset_file)
            
        self.latency_threshold  = latency_threshold_s
        self.max_examples       = max_examples
        self.dataset_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ───────────────────────────────────────────

    def analyze_trace(self, trace: Dict[str, Any]) -> Optional[str]:
        """Analyse a trace dict. If it qualifies as a hard example, persist it.

        Args:
            trace: Must contain at minimum ``trace_id``, ``goal``,
                   ``outcome``, and optionally ``latency`` / ``cost``.

        Returns:
            The reason string if the trace was mined, else None.

        """
        reason = self._classify(trace)
        if reason:
            self._save(trace, reason)
        return reason

    def load(self) -> List[Dict[str, Any]]:
        """Return all persisted hard examples."""
        if not self.dataset_file.exists():
            return []
        try:
            return json.loads(self.dataset_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load hard examples: %s", exc)
            return []

    def count(self) -> int:
        """Return the number of stored hard examples."""
        return len(self.load())

    # ── Internal ─────────────────────────────────────────────

    def _classify(self, trace: Dict[str, Any]) -> Optional[str]:
        """Return a reason string if the trace is hard, else None."""
        outcome = trace.get("outcome")

        if isinstance(outcome, dict) and not outcome.get("ok", True):
            return "Explicit Failure (outcome.ok=False)"

        if isinstance(outcome, str):
            low = outcome.lower()
            if "fail" in low or "error" in low:
                return "Textual Failure"

        latency = trace.get("latency") or trace.get("cost", 0)
        if latency > self.latency_threshold:
            return f"High Latency ({latency:.1f}s > {self.latency_threshold}s)"

        return None

    def _save(self, trace: Dict[str, Any], reason: str) -> None:
        """Atomically append a hard example to the dataset file."""
        logger.info("Mining hard example: %s (trace_id=%s)", reason, trace.get("trace_id"))

        example = {
            "trace_id":  trace.get("trace_id"),
            "reason":    reason,
            "goal":      trace.get("goal"),
            "outcome":   trace.get("outcome"),
            "mined_at":  time.time(),
        }

        data = self.load()

        # Enforce cap — evict oldest entries when over limit
        if len(data) >= self.max_examples:
            excess = len(data) - self.max_examples + 1
            data   = data[excess:]
            logger.debug("Hard-example cap reached — evicted %d oldest entries", excess)

        data.append(example)

        # Atomic write: write to a temp file in the same directory, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.dataset_file.parent, suffix=".json.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            Path(tmp_path).replace(self.dataset_file)   # atomic on POSIX; near-atomic on Windows
        except Exception as exc:
            record_degradation('data_engine', exc)
            logger.error("Failed to save hard example: %s", exc)
            try:
                os.unlink(tmp_path)
            except OSError:
                import logging
                logger.debug("Exception caught during execution", exc_info=True)
