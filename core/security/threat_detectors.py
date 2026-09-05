"""Per-class threat detectors — the senses that feed the immune system.

The immune system reasons and responds; these are the receptors that notice an attack in the
first place and hand it a classified signal. Each is a cheap, stateful detector over an event
stream that raises a ThreatEvent only when a real threshold is crossed — and stays quiet on
ordinary activity, so the immune system isn't drowned in false positives.

  RateAnomalyDetector   — request/connection floods, spam, resource-exhaustion volume
  BruteForceDetector    — failed-auth bursts (brute force / credential stuffing / spray)
  ExfilDetector         — abnormal outbound data volume to a destination
  InjectionDetector     — SQL / prompt / code-injection markers in untrusted input

Defensive only — they observe and report; the immune system decides the response.
"""
from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from core.security.immune_system import ThreatClass, ThreatEvent, get_immune_system


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class _Windowed:
    """Per-key rolling event counter."""

    def __init__(self, window_s: float) -> None:
        self._window = window_s
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()

    def hit(self, key: str, now: float, weight: float = 1.0) -> int:
        with self._lock:
            dq = self._events[key]
            dq.append(now)
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()
            return len(dq)

    def count(self, key: str, now: float) -> int:
        with self._lock:
            dq = self._events.get(key)
            if not dq:
                return 0
            cutoff = now - self._window
            while dq and dq[0] < cutoff:
                dq.popleft()
            return len(dq)


@dataclass
class _DetectorBase:
    def _emit(self, **kw: Any) -> ThreatEvent:
        return get_immune_system().assess(**kw)


class RateAnomalyDetector(_DetectorBase):
    """Flood / spam / resource-exhaustion via request rate per source."""

    def __init__(self, *, window_s: float = 5.0, threshold: int = 100) -> None:
        self._w = _Windowed(window_s)
        self._threshold = threshold

    def observe(self, source: str, *, kind: str = "request", now: float | None = None) -> ThreatEvent | None:
        now = time.time() if now is None else now
        n = self._w.hit(source, now)
        if n >= self._threshold:
            severity = _clamp(0.5 + 0.4 * min(1.0, n / (self._threshold * 2)))
            cls = ThreatClass.RESOURCE_EXHAUSTION if kind == "compute" else ThreatClass.NETWORK_FLOOD
            return self._emit(
                source="rate_detector", description=f"{kind} flood from {source}: {n}/window",
                severity=severity, origin=source, targeted_vuln="rate_limit", vector=kind,
                threat_class=cls, evidence={"count": n},
            )
        return None


class BruteForceDetector(_DetectorBase):
    """Failed-auth bursts: brute force / credential stuffing / password spray."""

    def __init__(self, *, window_s: float = 60.0, threshold: int = 8) -> None:
        self._w = _Windowed(window_s)
        self._threshold = threshold

    def observe_auth(self, source: str, *, success: bool, now: float | None = None) -> ThreatEvent | None:
        if success:
            return None
        now = time.time() if now is None else now
        n = self._w.hit(source, now)
        if n >= self._threshold:
            severity = _clamp(0.55 + 0.3 * min(1.0, n / (self._threshold * 2)))
            return self._emit(
                source="bruteforce_detector", description=f"{n} failed auths from {source}",
                severity=severity, origin=source, targeted_vuln="auth", vector="credential",
                threat_class=ThreatClass.CREDENTIAL, evidence={"failures": n},
            )
        return None


class ExfilDetector(_DetectorBase):
    """Abnormal outbound volume to a destination → possible data exfiltration."""

    def __init__(self, *, window_s: float = 30.0, byte_threshold: int = 50 * 1024 * 1024) -> None:
        self._window = window_s
        self._threshold = byte_threshold
        self._bytes: dict[str, deque[tuple]] = defaultdict(deque)
        self._lock = threading.RLock()

    def observe_egress(self, dest: str, n_bytes: int, *, now: float | None = None) -> ThreatEvent | None:
        now = time.time() if now is None else now
        with self._lock:
            dq = self._bytes[dest]
            dq.append((now, n_bytes))
            cutoff = now - self._window
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            total = sum(b for _t, b in dq)
        if total >= self._threshold:
            severity = _clamp(0.5 + 0.4 * min(1.0, total / (self._threshold * 2)))
            return self._emit(
                source="exfil_detector", description=f"large outbound {total} bytes to {dest}",
                severity=severity, origin=dest, targeted_vuln="data_egress", vector="network",
                threat_class=ThreatClass.DATA_EXFIL, evidence={"bytes": total},
            )
        return None


_INJECTION = re.compile(
    r"('\s*or\s*'?\d|union\s+select|drop\s+table|;\s*--|<script|javascript:|\$\{|"
    r"__import__|eval\s*\(|exec\s*\(|ignore (?:all )?previous|system prompt|"
    r"\.\./\.\./|/etc/passwd|;\s*rm\s+-rf)",
    re.IGNORECASE,
)


class InjectionDetector(_DetectorBase):
    """SQL / prompt / code injection markers in untrusted input."""

    def scan(self, text: str, *, origin: str = "user_input") -> ThreatEvent | None:
        m = _INJECTION.search(str(text or ""))
        if not m:
            return None
        return self._emit(
            source="injection_detector", description=f"injection marker: {m.group(0)[:40]}",
            severity=0.7, origin=origin, targeted_vuln="input_parsing", vector="injection",
            threat_class=ThreatClass.INJECTION, evidence={"match": m.group(0)[:80]},
        )

    def is_clean(self, text: str) -> bool:
        return _INJECTION.search(str(text or "")) is None


class ThreatDetectorSuite:
    """Bundles the detectors; the runtime feeds raw events in, ThreatEvents go to the immune system."""

    def __init__(self) -> None:
        self.rate = RateAnomalyDetector()
        self.bruteforce = BruteForceDetector()
        self.exfil = ExfilDetector()
        self.injection = InjectionDetector()


_suite: ThreatDetectorSuite | None = None
_lock = threading.Lock()


def get_threat_detectors() -> ThreatDetectorSuite:
    global _suite
    if _suite is None:
        with _lock:
            if _suite is None:
                _suite = ThreatDetectorSuite()
    return _suite
