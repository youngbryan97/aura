"""core/consciousness/quantum_entropy.py

External Entropy Bridge — High-Quality Random Number Generation via ANU QRNG API.

Sources random bytes from the ANU Quantum Random Number Generator, which
produces genuine quantum random numbers from vacuum fluctuations.  This
provides a high-quality external entropy source that improves unpredictability
compared to purely algorithmic seeds.

Important: Once the entropy is consumed as a seed, downstream computations
are deterministic given that seed.  This module provides high-quality
randomness at the point of injection — it does not make the entire decision
pipeline non-deterministic.  os.urandom would be functionally equivalent
for the system's purposes; ANU QRNG is used for entropy quality, not for
any claimed quantum effect on cognition.

Falls back gracefully to OS urandom when the network is unavailable.
Maintains a local cache to amortize API latency.

Integration: Enhances ManagedEntropy as a backend source. Also usable directly
via collapse_decision() for weighted random selection.
"""

import logging
import os
import struct
import threading
import time
from typing import Any, List, Optional, Sequence

logger = logging.getLogger("Consciousness.QuantumEntropy")

# ANU QRNG API endpoint (free tier)
_ANU_API_URL = "https://qrng.anu.edu.au/API/jsonI.php"
_POOL_SIZE = 1024          # Bytes to fetch per API call
_POOL_REFILL_THRESHOLD = 64  # Refill when pool drops below this
_API_TIMEOUT = 5.0         # Seconds
_MIN_REFILL_INTERVAL = 30.0  # Don't hammer the API


class QuantumEntropyBridge:
    """High-quality external entropy source backed by ANU QRNG.

    Architecture:
    - Fetches batches of random bytes from ANU QRNG API
    - Maintains a thread-safe local pool for low-latency reads
    - Falls back to os.urandom on network failure (functionally equivalent)
    - Tracks source provenance (external vs fallback) for telemetry

    Note: The value here is entropy quality and external sourcing, not any
    claimed quantum effect on the decision-making process itself.
    """

    def __init__(self, pool_size: int = _POOL_SIZE):
        self._pool: bytearray = bytearray()
        self._pool_size = pool_size
        self._lock = threading.Lock()
        self._last_refill_attempt: float = 0.0
        self._refill_in_flight = False
        self._quantum_reads: int = 0
        self._fallback_reads: int = 0
        self._api_failures: int = 0
        self._initialized = False

        # Seed immediate fallback entropy so startup never blocks on the network.
        self._seed_fallback_pool(min(256, self._pool_size))
        self._schedule_refill()
        logger.info("Quantum Entropy Bridge initialized (pool=%d bytes)", len(self._pool))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_quantum_float(self) -> float:
        """Return a float in [0.0, 1.0) from quantum entropy.
        Falls back to OS entropy if quantum pool is exhausted.
        """
        raw = self._consume_bytes(4)
        # Convert 4 bytes to uint32, then normalize to [0, 1)
        value = struct.unpack(">I", raw)[0]
        return value / (0xFFFFFFFF + 1.0)

    def get_quantum_bytes(self, n: int) -> bytes:
        """Return n bytes of quantum entropy."""
        return bytes(self._consume_bytes(n))

    def collapse_decision(
        self,
        options: Sequence[Any],
        weights: Optional[List[float]] = None,
    ) -> Any:
        """Select from options using quantum entropy, with optional weights.

        This is the primary interface for replacing deterministic random.choice()
        calls with quantum-grounded decisions.

        Args:
            options: Sequence of choices to select from.
            weights: Optional probability weights (will be normalized).

        Returns:
            Selected option.
        """
        if not options:
            raise ValueError("collapse_decision requires at least one option")

        if len(options) == 1:
            return options[0]

        r = self.get_quantum_float()

        if weights is None:
            # Uniform selection
            idx = int(r * len(options))
            idx = min(idx, len(options) - 1)  # Clamp
            return options[idx]

        # Weighted selection via cumulative distribution
        total = sum(weights)
        if total <= 0:
            # Degenerate weights — fall back to uniform
            idx = int(r * len(options))
            return options[min(idx, len(options) - 1)]

        normalized = [w / total for w in weights]
        cumulative = 0.0
        for i, w in enumerate(normalized):
            cumulative += w
            if r < cumulative:
                return options[i]

        return options[-1]  # Rounding safety

    def get_stats(self) -> dict:
        """Telemetry for the entropy bridge."""
        total = self._quantum_reads + self._fallback_reads
        return {
            "quantum_reads": self._quantum_reads,
            "fallback_reads": self._fallback_reads,
            "api_failures": self._api_failures,
            "pool_size": len(self._pool),
            "quantum_ratio": (
                self._quantum_reads / total if total > 0 else 0.0
            ),
            "source": "quantum" if self._quantum_reads > self._fallback_reads else "fallback",
        }

    # ------------------------------------------------------------------
    # Internal pool management
    # ------------------------------------------------------------------

    def _consume_bytes(self, n: int) -> bytes:
        """Consume n bytes from the pool without stalling the caller."""
        with self._lock:
            available = len(self._pool)
            needs_refill = available < max(n, _POOL_REFILL_THRESHOLD)

            if available >= n:
                result = bytes(self._pool[:n])
                del self._pool[:n]
                self._quantum_reads += 1
            else:
                quantum_part = bytes(self._pool)
                self._pool.clear()
                self._fallback_reads += 1
                result = quantum_part + os.urandom(max(0, n - len(quantum_part)))

        if needs_refill:
            self._schedule_refill()
        return result

    def _seed_fallback_pool(self, size: int) -> None:
        if size <= 0:
            return
        with self._lock:
            if not self._pool:
                self._pool.extend(os.urandom(size))

    def _schedule_refill(self) -> None:
        """Start a background refill without blocking current work."""
        now = time.time()
        with self._lock:
            if self._refill_in_flight:
                return
            if now - self._last_refill_attempt < _MIN_REFILL_INTERVAL:
                return
            self._refill_in_flight = True
            self._last_refill_attempt = now

        thread = threading.Thread(
            target=self._refill_worker,
            name="aura_quantum_entropy_refill",
            daemon=True,
        )
        thread.start()

    def _refill_worker(self) -> None:
        try:
            self._try_refill_blocking()
        finally:
            with self._lock:
                self._refill_in_flight = False

    def _try_refill_blocking(self) -> None:
        """Attempt to refill the pool from ANU QRNG API in the background."""

        try:
            import urllib.request
            import json

            url = f"{_ANU_API_URL}?length={self._pool_size}&type=uint8"
            req = urllib.request.Request(url, headers={"User-Agent": "AURA/1.0"})
            with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())

            if data.get("success") and "data" in data:
                new_bytes = bytearray(data["data"])
                with self._lock:
                    self._pool.extend(new_bytes)
                self._initialized = True
                logger.debug(
                    "Quantum pool refilled: +%d bytes (total: %d)",
                    len(new_bytes),
                    len(self._pool),
                )
            else:
                self._api_failures += 1
                logger.warning("ANU QRNG API returned non-success: %s", data)

        except Exception as e:
            self._api_failures += 1
            logger.debug("ANU QRNG API unavailable (using OS fallback): %s", e)
            self._seed_fallback_pool(min(256, self._pool_size))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_instance: Optional[QuantumEntropyBridge] = None


def get_quantum_entropy() -> QuantumEntropyBridge:
    """Get or create the singleton QuantumEntropyBridge."""
    global _instance
    if _instance is None:
        _instance = QuantumEntropyBridge()
    return _instance
