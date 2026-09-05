"""core/resilience/tmr.py
━━━━━━━━━━━━━━━━━━━━━━━
Triple Modular Redundancy (TMR) for reliability-grade decision integrity.

Three independent execution channels run the same computation. A
Byzantine-fault-tolerant majority voter selects the result. If channels
diverge, a FAULT-F19 is recorded and the dissenting channel is flagged in
the voter's quarantine set. Quarantine is advisory bookkeeping: channels
are per-call callables with no persistent identity, so the voter cannot
skip a "quarantined" channel on later calls — it surfaces repeat offenders
via status() for the operator instead.

Designed for Aura's most critical decision paths:
- Will REFUSE/CRITICAL decisions
- Identity hash verification
- Memory integrity checks

Usage:
    from core.resilience.tmr import TMRVoter

    voter = TMRVoter("will_decision")
    result = voter.execute(
        lambda: compute_will_decision(context),  # Channel A
        lambda: compute_will_decision(context),  # Channel B (independent)
        lambda: compute_will_decision(context),  # Channel C (independent)
    )

Performance: TMR triples CPU cost. Enable only for critical paths via
AURA_TMR_ENABLED=1 (default: off). When disabled, runs channel A only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger("Aura.TMR")

T = TypeVar("T")

_TMR_ENABLED = os.environ.get("AURA_TMR_ENABLED", "0") == "1"

# Guarded-callable failure envelope: user-supplied callables (checks, guards,
# actions, hooks, channels) may raise anything. House discipline forbids broad
# `except Exception`; this tuple names the realistic failure universe
# explicitly. Exotic escapes — custom Exception subtypes outside these bases,
# SystemExit, KeyboardInterrupt — propagate loudly by design.
_GUARDED_CALLABLE_ERRORS = (
    RuntimeError, AttributeError, TypeError, ValueError,
    LookupError, ArithmeticError, OSError, ImportError,
)


@dataclass
class ChannelResult:
    """Result from a single TMR channel."""
    channel_id: str
    value: Any
    fingerprint: str
    latency_ms: float
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class TMRResult:
    """Result from TMR voting."""
    value: Any
    unanimous: bool
    winning_fingerprint: str
    channels: list[ChannelResult]
    divergent_channels: list[str] = field(default_factory=list)
    vote_count: int = 0

    @property
    def had_divergence(self) -> bool:
        return len(self.divergent_channels) > 0


def _fingerprint(value: Any) -> str:
    """Compute a deterministic fingerprint for comparison.

    For complex objects, we JSON-serialize and hash. For simple types,
    we hash the repr directly.
    """
    try:
        if isinstance(value, (str, int, float, bool, type(None))):
            raw = repr(value)
        elif hasattr(value, "__dict__"):
            raw = json.dumps(
                {k: repr(v) for k, v in sorted(vars(value).items())},
                sort_keys=True,
            )
        else:
            raw = repr(value)
    except (TypeError, ValueError):
        raw = repr(value)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


class TMRVoter:
    """Triple Modular Redundancy voter.

    Thread-safe. Each instance tracks divergence history for its named
    subsystem.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._divergence_count = 0
        self._total_votes = 0
        self._quarantined_channels: set[str] = set()

    def execute(
        self,
        channel_a: Callable[[], T],
        channel_b: Callable[[], T] | None = None,
        channel_c: Callable[[], T] | None = None,
    ) -> TMRResult:
        """Execute up to three channels and vote on the result.

        If TMR is disabled globally or fewer than 3 channels are provided,
        only channel_a is executed (simplex mode).
        """
        if not _TMR_ENABLED or channel_b is None or channel_c is None:
            return self._simplex(channel_a)

        channels = [
            ("A", channel_a),
            ("B", channel_b),
            ("C", channel_c),
        ]

        results: list[ChannelResult] = []
        for cid, fn in channels:
            t0 = time.perf_counter()
            try:
                value = fn()
                latency = (time.perf_counter() - t0) * 1000
                fp = _fingerprint(value)
                results.append(ChannelResult(
                    channel_id=cid, value=value,
                    fingerprint=fp, latency_ms=latency,
                ))
            except _GUARDED_CALLABLE_ERRORS as exc:
                latency = (time.perf_counter() - t0) * 1000
                results.append(ChannelResult(
                    channel_id=cid, value=None,
                    fingerprint="ERROR", latency_ms=latency, error=exc,
                ))

        return self._vote(results)

    def _simplex(self, fn: Callable[[], T]) -> TMRResult:
        """Single-channel execution (TMR disabled). Channel errors propagate."""
        t0 = time.perf_counter()
        value = fn()
        latency = (time.perf_counter() - t0) * 1000
        fp = _fingerprint(value)
        cr = ChannelResult(channel_id="A", value=value,
                           fingerprint=fp, latency_ms=latency)
        with self._lock:
            # Simplex executions count toward total_votes: status() must
            # reflect real traffic in the default (TMR-off) configuration.
            self._total_votes += 1
        return TMRResult(
            value=value, unanimous=True, winning_fingerprint=fp,
            channels=[cr], vote_count=1,
        )

    def _vote(self, results: list[ChannelResult]) -> TMRResult:
        """Byzantine-fault-tolerant majority vote."""
        successful = [r for r in results if r.succeeded]

        if not successful:
            # All channels failed — raise the first error
            first_error = next((r.error for r in results if r.error), None)
            if first_error:
                raise first_error
            raise RuntimeError(f"TMR {self.name}: all channels failed with no error")

        # Count fingerprints
        fp_counter = Counter(r.fingerprint for r in successful)
        majority_fp, majority_count = fp_counter.most_common(1)[0]

        # Find the winning value (first channel with majority fingerprint)
        winner = next(r for r in successful if r.fingerprint == majority_fp)

        # Identify divergent channels
        divergent = [r.channel_id for r in successful if r.fingerprint != majority_fp]
        # Also include failed channels as divergent
        divergent.extend(r.channel_id for r in results if r.error is not None)

        unanimous = len(divergent) == 0

        with self._lock:
            self._total_votes += 1
            if not unanimous:
                self._divergence_count += 1
                for ch in divergent:
                    self._quarantined_channels.add(ch)
                self._record_divergence(results, majority_fp, divergent)

        return TMRResult(
            value=winner.value,
            unanimous=unanimous,
            winning_fingerprint=majority_fp,
            channels=results,
            divergent_channels=divergent,
            vote_count=majority_count,
        )

    def _record_divergence(
        self,
        results: list[ChannelResult],
        majority_fp: str,
        divergent: list[str],
    ) -> None:
        """Log divergence and record fault."""
        logger.error(
            "TMR DIVERGENCE in %s: majority=%s, divergent_channels=%s",
            self.name, majority_fp, divergent,
        )
        try:
            from core.resilience.fault_taxonomy import get_fault_registry
            get_fault_registry().record_fault(
                "F19", subsystem=f"tmr.{self.name}",
                details=f"Divergent channels: {divergent}, majority: {majority_fp}",
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            # Fault registry may not be importable during early boot.
            logger.debug("Fault registry unavailable for TMR divergence: %s", exc)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "enabled": _TMR_ENABLED,
                "total_votes": self._total_votes,
                "divergences": self._divergence_count,
                "divergence_rate": (
                    self._divergence_count / max(self._total_votes, 1)
                ),
                "quarantined_channels": list(self._quarantined_channels),
            }
