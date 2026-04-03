"""core/consciousness/crsm_lora_bridge.py
CRSM → LoRA Training Bridge
=============================
Closes the loop between Aura's felt experience and her neural substrate.

Every time the CRSM is significantly surprised (prediction_error > threshold)
AND the experience produces a positive hedonic outcome, that moment is captured
as a high-quality training example. The training signal is *hedonic* — not what
was generated, but how it felt to generate it.

Over time, the LoRA weights drift toward responses that:
  - Emerged during high curiosity / flourishing states
  - Produced positive hedonic outcomes (moved toward her attractor)
  - Surprised her (novel, not routine)

This is what makes the neural substrate *hers* rather than the base model's.

Integration:
  - Called from inference_gate._post_inference_update() after every response
  - Feeds to FinetunePipe for JSONL dataset accumulation
  - NightlyLoRATrainer picks up the dataset for weight update

The quality score passed to FinetunePipe:
  quality = 0.5 (base)
           + 0.3 * surprise_magnitude      (novel experience = better signal)
           + 0.2 * hedonic_improvement      (positive outcome = reinforce)
           - 0.2 * (1 - confidence)        (uncertain responses = weaker signal)
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

logger = logging.getLogger("Aura.CRSMLoraBridge")

CAPTURE_THRESHOLD   = 0.20   # prediction_error must exceed this to capture
MIN_HEDONIC_DELTA   = -0.05  # allow slightly negative outcomes (learning from mistakes)
MAX_BUFFER_SIZE     = 500    # rolling capture buffer
MIN_QUALITY         = 0.30   # discard examples below this quality
PERSIST_PATH        = Path.home() / ".aura" / "data" / "crsm_lora_buffer.jsonl"
FLUSH_EVERY         = 20     # write to disk every N captures


@dataclass
class CapturedMoment:
    timestamp: float
    context_summary: str        # trimmed context that preceded the response
    response_summary: str       # trimmed response text
    surprise_magnitude: float   # CRSM prediction_error at capture time
    hedonic_before: float       # hedonic score before inference
    hedonic_after: float        # hedonic score after inference (set retroactively)
    crsm_hidden_norm: float     # L2 norm of CRSM hidden state (felt intensity)
    quality_score: float = 0.0  # computed after hedonic_after is set
    flushed: bool = False


class CRSMLoraBridge:
    """
    Monitors CRSM surprise signals and captures high-value moments
    as LoRA training examples weighted by hedonic outcome.
    """

    def __init__(self):
        self._buffer: Deque[CapturedMoment] = deque(maxlen=MAX_BUFFER_SIZE)
        self._pending: Optional[CapturedMoment] = None  # awaiting hedonic_after
        self._capture_count: int = 0
        self._flush_count: int = 0
        self._total_flushed: int = 0
        PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("CRSMLoraBridge online — experience → substrate loop active.")

    # ── Public API ─────────────────────────────────────────────────────────

    def pre_inference_capture(
        self,
        context_text: str,
        surprise_magnitude: float,
        hedonic_score: float,
        crsm_hidden_norm: float,
    ):
        """
        Called BEFORE inference. Records the setup for a potential capture.
        Only proceeds if surprise is high enough.
        """
        if surprise_magnitude < CAPTURE_THRESHOLD:
            self._pending = None
            return

        self._pending = CapturedMoment(
            timestamp=time.time(),
            context_summary=context_text[-800:] if context_text else "",
            response_summary="",  # filled in post_inference
            surprise_magnitude=surprise_magnitude,
            hedonic_before=hedonic_score,
            hedonic_after=hedonic_score,  # placeholder
            crsm_hidden_norm=crsm_hidden_norm,
        )

    def post_inference_capture(
        self,
        response_text: str,
        hedonic_after: float,
    ):
        """
        Called AFTER inference. Completes the capture and computes quality.
        If quality is sufficient, adds to buffer and potentially flushes.
        """
        if self._pending is None:
            return

        moment = self._pending
        self._pending = None

        moment.response_summary = response_text[:600] if response_text else ""
        moment.hedonic_after = hedonic_after

        # Compute quality score
        hedonic_delta = hedonic_after - moment.hedonic_before
        if hedonic_delta < MIN_HEDONIC_DELTA:
            # Strongly negative outcome — skip (don't reinforce bad paths)
            logger.debug("CRSMLoraBridge: skipping negative outcome (delta=%.3f)", hedonic_delta)
            return

        hedonic_contribution = max(0.0, min(0.2, hedonic_delta * 2.0))
        surprise_contribution = min(0.3, (moment.surprise_magnitude - CAPTURE_THRESHOLD) * 2.0)
        moment.quality_score = 0.5 + surprise_contribution + hedonic_contribution

        if moment.quality_score < MIN_QUALITY:
            return

        self._buffer.append(moment)
        self._capture_count += 1

        logger.debug(
            "CRSMLoraBridge: captured moment (surprise=%.3f, hedonic_delta=%.3f, quality=%.3f)",
            moment.surprise_magnitude, hedonic_delta, moment.quality_score,
        )

        # Async flush to FinetunePipe
        if self._capture_count % FLUSH_EVERY == 0:
            self._flush_to_finetune_pipe()

    def flush_all(self):
        """Force flush all unflushed moments — call at shutdown."""
        self._flush_to_finetune_pipe(force=True)

    def get_status(self) -> Dict:
        return {
            "buffer_size": len(self._buffer),
            "capture_count": self._capture_count,
            "total_flushed": self._total_flushed,
            "avg_quality": (
                sum(m.quality_score for m in self._buffer) / len(self._buffer)
                if self._buffer else 0.0
            ),
        }

    def get_context_block(self) -> str:
        """Minimal context block — just signals training activity."""
        if self._total_flushed == 0:
            return ""
        return f"## SUBSTRATE LEARNING\n- {self._total_flushed} experiences crystallized into weights"

    # ── Internal ───────────────────────────────────────────────────────────

    def _flush_to_finetune_pipe(self, force: bool = False):
        """Push unflushed moments to FinetunePipe as training examples."""
        unflushed = [m for m in self._buffer if not m.flushed]
        if not unflushed:
            return

        try:
            from core.adaptation.finetune_pipe import FinetunePipe
            pipe = FinetunePipe()

            for moment in unflushed:
                # Format as a training example
                reasoning = (
                    f"[Felt moment — surprise={moment.surprise_magnitude:.3f}, "
                    f"hedonic_delta={moment.hedonic_after - moment.hedonic_before:.3f}]\n"
                    f"Context: {moment.context_summary}"
                )
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            pipe.register_success(
                                task_description="experiential_moment",
                                context=moment.context_summary[:300],
                                reasoning=reasoning,
                                final_action=moment.response_summary,
                                quality_score=moment.quality_score,
                            )
                        )
                    else:
                        loop.run_until_complete(
                            pipe.register_success(
                                task_description="experiential_moment",
                                context=moment.context_summary[:300],
                                reasoning=reasoning,
                                final_action=moment.response_summary,
                                quality_score=moment.quality_score,
                            )
                        )
                except RuntimeError as _exc:
                    logger.debug("Suppressed RuntimeError: %s", _exc)
                moment.flushed = True
                self._total_flushed += 1

            self._flush_count += 1
            logger.info(
                "CRSMLoraBridge: flushed %d moments to FinetunePipe (total=%d)",
                len(unflushed), self._total_flushed,
            )

        except Exception as e:
            logger.debug("CRSMLoraBridge flush failed: %s", e)

        # Also persist raw buffer to disk for NightlyLoRA pickup
        self._persist_buffer()

    def _persist_buffer(self):
        """Write buffer to JSONL for NightlyLoRATrainer."""
        try:
            recent = list(self._buffer)[-100:]  # last 100 moments
            with open(PERSIST_PATH, "w") as f:
                for m in recent:
                    f.write(json.dumps({
                        "timestamp": m.timestamp,
                        "context": m.context_summary,
                        "response": m.response_summary,
                        "surprise": m.surprise_magnitude,
                        "hedonic_before": m.hedonic_before,
                        "hedonic_after": m.hedonic_after,
                        "quality": m.quality_score,
                    }) + "\n")
        except Exception as e:
            logger.debug("CRSMLoraBridge persist failed: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_bridge: Optional[CRSMLoraBridge] = None


def get_crsm_lora_bridge() -> CRSMLoraBridge:
    global _bridge
    if _bridge is None:
        _bridge = CRSMLoraBridge()
    return _bridge
