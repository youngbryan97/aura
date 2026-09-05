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
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.CRSMLoraBridge")

CAPTURE_THRESHOLD   = 0.20   # prediction_error must exceed this to capture
MIN_HEDONIC_DELTA   = -0.05  # allow slightly negative outcomes (learning from mistakes)
MAX_BUFFER_SIZE     = 500    # rolling capture buffer
MIN_QUALITY         = 0.30   # discard examples below this quality
PERSIST_PATH        = state_root() / "data" / "crsm_lora_buffer.jsonl"

#: How long a captured moment may remain in the training corpus.
#:
#: CP126 asked for a retention policy and there was none: the buffer is
#: size-bounded at 500, so the file cannot grow without limit, but a moment
#: captured months ago stays in it indefinitely as long as the buffer never
#: fills. Size is not retention.
#:
#: 30 days is a POLICY choice, not a derived quantity, and is written here
#: as one — this is conversational data about a person, and "we keep it
#: until the ring buffer happens to evict it" is not an answer anyone should
#: have to give. Override with AURA_TRAINING_RETENTION_DAYS.
TRAINING_RETENTION_DAYS = 30
FLUSH_EVERY         = 20     # write to disk every N captures

# Direct identifiers that must never enter the persisted training corpus.
# The MetadataScrubber handles hosts/paths/IPs/secrets; these cover the
# user-content classes it does not.
_TRAINING_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{3,4}(?:[ .-]\d{2,4})?\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
)


def _training_capture_enabled() -> bool:
    return str(
        os.environ.get("AURA_SUBSTRATE_TRAINING_CAPTURE", "on")
    ).strip().lower() not in {"0", "false", "no", "off"}


def _redact_for_training(text: str) -> str:
    """Scrub direct identifiers before text becomes training data.

    Applied at the capture boundary so raw prompts/responses never reach the
    persisted JSONL corpus with emails, phone numbers, card-like digit runs,
    SSNs, secrets, or local host identity intact.
    """
    if not text:
        return ""
    try:
        from core.utils.privacy_hygiene import get_stealth_mode

        scrubbed = get_stealth_mode().scrubber.scrub_text(text)
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("crsm_lora_bridge", exc)
        scrubbed = text
    for pattern, replacement in _TRAINING_PII_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed


#: Topic markers that make a whole exchange unfit for a training corpus.
#:
#: Distinct from _TRAINING_PII_PATTERNS, which removes identifiers from text
#: that is otherwise fine to keep. Some exchanges are not fixable by
#: redaction: strip the account number from a conversation about someone's
#: medical results or their bank dispute and what remains is still their
#: medical results and their bank dispute, now with a gap in it. The unit of
#: exclusion has to be the moment, not the substring.
_SENSITIVE_TOPIC_MARKERS: tuple[str, ...] = (
    "medical record", "diagnosis", "prescription", "therapist", "psychiatric",
    "mental health", "self-harm", "suicide",
    "bank account", "routing number", "credit card", "tax return", "social security",
    "immigration status", "visa application", "deportation",
    "criminal record", "arrest", "lawsuit", "attorney-client", "legal advice",
    "sexual", "intimate", "religious belief", "political affiliation",
    "password", "seed phrase", "private key", "two-factor", "recovery code",
)


def _retention_window_s() -> float:
    """Retention window in seconds, from policy or its environment override."""
    raw = os.environ.get("AURA_TRAINING_RETENTION_DAYS", "")
    try:
        days = float(raw) if str(raw).strip() else float(TRAINING_RETENTION_DAYS)
    except (TypeError, ValueError):
        days = float(TRAINING_RETENTION_DAYS)
    if not (days > 0):
        # A non-positive window would mean "keep nothing", which is far more
        # likely a typo than an intention. Fall back to the stated policy.
        days = float(TRAINING_RETENTION_DAYS)
    return days * 86_400.0


def _training_eligible(text: str) -> tuple[bool, str]:
    """May this text enter a persisted training corpus at all?

    CP126 raised this twice — once against the pre-inference prompt capture
    and once against the response capture: "no sensitivity classification,
    user consent, redaction, per-principal namespace, retention policy,
    training eligibility check, or operation receipt".

    Redaction (which now exists) answers only the identifier half. This
    answers the other half: an exchange whose SUBJECT is sensitive should
    not be retained at all, redacted or otherwise.

    Deliberately conservative and deliberately dumb. A keyword screen will
    over-exclude, and over-excluding from a training corpus costs a training
    example; under-excluding costs someone's medical history sitting in a
    JSONL file on disk. Those are not comparable, so the cheap check is the
    right check.
    """
    if not text:
        return True, ""
    lowered = text.lower()
    for marker in _SENSITIVE_TOPIC_MARKERS:
        if marker in lowered:
            return False, marker
    return True, ""


@dataclass
class CapturedMoment:
    timestamp: float
    context_summary: str        # trimmed context that preceded the response
    response_summary: str       # trimmed response text
    surprise_magnitude: float   # CRSM prediction_error at capture time
    hedonic_before: float       # hedonic score before inference
    hedonic_after: float        # hedonic score after inference (set retroactively)
    crsm_hidden_norm: float     # L2 norm of CRSM hidden state (felt intensity)
    processing_context: dict[str, object] = field(default_factory=dict)
    quality_score: float = 0.0  # computed after hedonic_after is set
    flushed: bool = False


class CRSMLoraBridge:
    """
    Monitors CRSM surprise signals and captures high-value moments
    as LoRA training examples weighted by hedonic outcome.
    """

    def __init__(self):
        self._buffer: deque[CapturedMoment] = deque(maxlen=MAX_BUFFER_SIZE)
        self._pending: CapturedMoment | None = None  # awaiting hedonic_after
        self._capture_count: int = 0
        self._flush_count: int = 0
        self._total_flushed: int = 0
        self._excluded_sensitive: int = 0
        self._expired_by_retention: int = 0
        self._status_cache: tuple[float, dict] | None = None
        PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("CRSMLoraBridge online — experience → substrate loop active.")

    def _capture_processing_context(self) -> dict[str, object]:
        context: dict[str, object] = {}
        try:
            from core.container import ServiceContainer

            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs and hasattr(ncs, "get_mood_vector"):
                context["mood"] = {
                    str(key): round(float(value), 4)
                    for key, value in dict(ncs.get_mood_vector() or {}).items()
                }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("crsm_lora_bridge", exc)
        try:
            from core.consciousness.affective_steering import get_steering_engine

            steering = get_steering_engine().get_status()
            production = steering.get("production_caa", {}) if isinstance(steering, dict) else {}
            readiness = production.get("readiness", {}) if isinstance(production, dict) else {}
            alpha_state = production.get("alpha_state", {}) if isinstance(production, dict) else {}
            context["steering"] = {
                "alpha": round(float(steering.get("alpha", 0.0) or 0.0), 4),
                "vector_source": steering.get("vector_source", "unloaded"),
                "vector_count": int(steering.get("vector_count", 0) or 0),
                "readiness_level": readiness.get("level", "bootstrap"),
                "readiness_detail": readiness.get("detail", ""),
                "adaptive_alpha": alpha_state.get("current_alpha", steering.get("alpha", 0.0)),
            }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("crsm_lora_bridge", exc)
        return context

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
        Only proceeds if surprise is high enough, training capture is enabled,
        and the context text has been redacted for training eligibility.
        """
        if not _training_capture_enabled():
            self._pending = None
            return
        if surprise_magnitude < CAPTURE_THRESHOLD:
            self._pending = None
            return

        eligible, marker = _training_eligible(context_text)
        if not eligible:
            self._pending = None
            self._excluded_sensitive += 1
            # The marker, never the text. Recording what tripped the screen
            # is how it gets tuned; recording the excerpt would recreate the
            # thing being excluded.
            logger.info(
                "CRSMLoraBridge: excluded a sensitive exchange from training "
                "capture (marker=%r; %d excluded since boot).",
                marker,
                self._excluded_sensitive,
            )
            return

        self._pending = CapturedMoment(
            timestamp=time.time(),
            context_summary=_redact_for_training(context_text[-800:] if context_text else ""),
            response_summary="",  # filled in post_inference
            surprise_magnitude=surprise_magnitude,
            hedonic_before=hedonic_score,
            hedonic_after=hedonic_score,  # updated after inference
            crsm_hidden_norm=crsm_hidden_norm,
            processing_context=self._capture_processing_context(),
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

        eligible, marker = _training_eligible(response_text)
        if not eligible:
            # A benign prompt can still draw a sensitive answer, so the
            # screen runs on both halves of the exchange.
            self._excluded_sensitive += 1
            logger.info(
                "CRSMLoraBridge: excluded a sensitive response from training "
                "capture (marker=%r; %d excluded since boot).",
                marker,
                self._excluded_sensitive,
            )
            return

        moment.response_summary = _redact_for_training(
            response_text[:600] if response_text else ""
        )
        moment.hedonic_after = hedonic_after

        # Compute quality score
        hedonic_delta = hedonic_after - moment.hedonic_before
        if hedonic_delta < MIN_HEDONIC_DELTA:
            # Strongly negative outcome — skip (don't reinforce bad paths)
            logger.debug("CRSMLoraBridge: skipping negative outcome (delta=%.3f)", hedonic_delta)
            return

        hedonic_contribution = max(0.0, min(0.2, hedonic_delta * 2.0))
        surprise_contribution = min(0.3, (moment.surprise_magnitude - CAPTURE_THRESHOLD) * 2.0)
        readiness_level = (
            str(moment.processing_context.get("steering", {}).get("readiness_level", "bootstrap"))
            if isinstance(moment.processing_context.get("steering"), dict)
            else "bootstrap"
        )
        readiness_bonus = 0.08 if readiness_level == "production" else 0.04 if readiness_level == "validated" else 0.0
        moment.quality_score = min(1.0, 0.5 + surprise_contribution + hedonic_contribution + readiness_bonus)

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

    def _status_cache_ttl_s(self) -> float:
        default = 0.0 if os.environ.get("PYTEST_CURRENT_TEST") else 5.0
        try:
            return max(0.0, float(os.getenv("AURA_CRSM_STATUS_CACHE_TTL_S", str(default))))
        except (TypeError, ValueError):
            return default

    def get_status(self, *, force_refresh: bool = False) -> dict:
        ttl = self._status_cache_ttl_s()
        now = time.monotonic()
        if not force_refresh and ttl > 0.0 and self._status_cache is not None:
            cached_at, cached = self._status_cache
            if now - cached_at <= ttl:
                status = dict(cached)
                status["_cached"] = True
                status["_cache_age_s"] = round(now - cached_at, 3)
                return status

        status = {
            "buffer_size": len(self._buffer),
            "capture_count": self._capture_count,
            "total_flushed": self._total_flushed,
            # The capture receipt CP126 asked for: how much was deliberately
            # withheld. A corpus with no exclusions is either a very tame
            # conversation history or a screen that is not running.
            "excluded_sensitive": self._excluded_sensitive,
            "expired_by_retention": self._expired_by_retention,
            "retention_days": TRAINING_RETENTION_DAYS,
            "training_capture_enabled": _training_capture_enabled(),
            "avg_quality": (
                sum(m.quality_score for m in self._buffer) / len(self._buffer)
                if self._buffer else 0.0
            ),
            "last_processing_context": self._buffer[-1].processing_context if self._buffer else {},
        }
        # Loop-closure verification can hash and parse the training corpus. Normal
        # status readers consume the prewarmed integrity snapshot so inference and
        # HTTP callers never perform that work inline. Training/diagnostic callers
        # retain an explicit force-refresh path for current transaction proof.
        try:
            if force_refresh:
                from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

                status["loop"] = get_crsm_loop_monitor().loop_state()
                status["loop_read_model"] = {"serving": "forced_current"}
            else:
                from core.runtime.integrity_audit import read_integrity_audit

                integrity = read_integrity_audit()
                status["loop"] = integrity.get("crsm_loop") or None
                status["loop_read_model"] = dict(
                    integrity.get("integrity_read_model") or {}
                )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            status["loop"] = None
            status["loop_read_model"] = {"serving": "unavailable"}
        if ttl > 0.0:
            self._status_cache = (now, dict(status))
        return status

    def get_context_block(self) -> str:
        """Minimal context block — just signals training activity."""
        if self._total_flushed == 0:
            return ""
        readiness = ""
        if self._buffer:
            steering = self._buffer[-1].processing_context.get("steering", {})
            if isinstance(steering, dict):
                readiness = str(steering.get("readiness_level", "bootstrap"))
        return (
            f"## SUBSTRATE LEARNING\n- {self._total_flushed} experiences crystallized into weights"
            f"\n- steering readiness: {readiness or 'bootstrap'}"
        )

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
                steering = moment.processing_context.get("steering", {}) if isinstance(moment.processing_context, dict) else {}
                mood = moment.processing_context.get("mood", {}) if isinstance(moment.processing_context, dict) else {}
                reasoning = (
                    f"[Felt moment — surprise={moment.surprise_magnitude:.3f}, "
                    f"hedonic_delta={moment.hedonic_after - moment.hedonic_before:.3f}, "
                    f"steering={steering.get('readiness_level', 'bootstrap')}, "
                    f"alpha={steering.get('adaptive_alpha', steering.get('alpha', 0.0))}]\n"
                    f"Processing context: mood={mood} steering={steering}\n"
                    f"Context: {moment.context_summary}"
                )
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        get_task_tracker().track(
                            pipe.register_success(
                                task_description="experiential_moment",
                                context=moment.context_summary[:300],
                                reasoning=reasoning,
                                final_action=moment.response_summary,
                                quality_score=moment.quality_score,
                                metadata=moment.processing_context,
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
                                metadata=moment.processing_context,
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

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('crsm_lora_bridge', e)
            logger.debug("CRSMLoraBridge flush failed: %s", e)

        # Also persist raw buffer to disk for NightlyLoRA pickup
        self._persist_buffer()

    def _expire_stale_moments(self) -> None:
        """Drop captured moments past the retention window.

        Runs at persist time so the file on disk never carries a moment
        older than the policy allows, and so the expiry is observable in the
        same place the corpus is written.
        """
        cutoff = time.time() - _retention_window_s()
        live = [m for m in self._buffer if float(m.timestamp or 0.0) >= cutoff]
        expired = len(self._buffer) - len(live)
        if not expired:
            return
        self._buffer.clear()
        self._buffer.extend(live)
        self._expired_by_retention += expired
        logger.info(
            "CRSMLoraBridge: expired %d captured moment(s) past the %s-day "
            "retention window (%d expired since boot).",
            expired,
            TRAINING_RETENTION_DAYS,
            self._expired_by_retention,
        )

    def _persist_buffer(self):
        """Write buffer to JSONL for NightlyLoRATrainer."""
        try:
            self._expire_stale_moments()
            recent = list(self._buffer)[-100:]  # last 100 moments
            lines = []
            for m in recent:
                lines.append(
                    json.dumps({
                        "timestamp": m.timestamp,
                        "context": m.context_summary,
                        "response": m.response_summary,
                        "surprise": m.surprise_magnitude,
                        "hedonic_before": m.hedonic_before,
                        "hedonic_after": m.hedonic_after,
                        "quality": m.quality_score,
                        "processing_context": m.processing_context,
                    })
                )
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().write_text(
                PERSIST_PATH,
                "\n".join(lines) + ("\n" if lines else ""),
                source="crsm_lora_bridge.persist_buffer",
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation('crsm_lora_bridge', e)
            logger.debug("CRSMLoraBridge persist failed: %s", e)


# ── Singleton ──────────────────────────────────────────────────────────────────

_bridge: CRSMLoraBridge | None = None


def get_crsm_lora_bridge() -> CRSMLoraBridge:
    global _bridge
    if _bridge is None:
        _bridge = CRSMLoraBridge()
    return _bridge
