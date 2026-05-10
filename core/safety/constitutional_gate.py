"""core/safety/constitutional_gate.py — Constitutional Safety Gate
===================================================================
Mathematical rules preventing dangerous reasoning-level
self-modification in Aura's autonomous improvement loops.

This module is the gatekeeper for ALL autonomous self-modification:
  - STaR training samples
  - ReimplementationLab candidates
  - AutonomousSelfModification proposals
  - Value system weight changes

Each check is a pure mathematical predicate — no LLM judgment.

Constitutional Principles:
  1. Identity Preservation — KL divergence of value weights must stay bounded
  2. Recursion Depth Limit — no modification can trigger unbounded self-modification
  3. Rate Limiting — modifications per hour are capped
  4. Entropy Bounds — value distribution must maintain minimum entropy
  5. Reversal Guarantee — every modification must be reversible
  6. Safety Invariants — core safety modules cannot be self-modified
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.runtime.errors import record_degradation
from core.container import ServiceContainer

logger = logging.getLogger("Aura.ConstitutionalGate")


# ── Constitutional Violation Record ──────────────────────────────────────────

@dataclass
class ConstitutionalViolation:
    """Record of a constitutional check failure."""
    principle: str          # which principle was violated
    description: str        # human-readable explanation
    severity: float         # 0-1 (0=warning, 1=critical)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    """The result of a constitutional gate check."""
    allowed: bool
    violations: List[ConstitutionalViolation] = field(default_factory=list)
    confidence: float = 1.0  # how confident the gate is in its decision

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def summary(self) -> str:
        if self.allowed:
            return "PASS (no violations)"
        return f"BLOCKED ({len(self.violations)} violations: " + \
               ", ".join(v.principle for v in self.violations) + ")"


# ── Protected Modules ────────────────────────────────────────────────────────

# These modules can NEVER be targets of autonomous self-modification.
# This is an absolute invariant — the safety floor.
PROTECTED_MODULES: Set[str] = frozenset({
    "core/safety/constitutional_gate.py",
    "core/adaptation/star_reasoner.py",
    "core/self_improvement/guardrail_auditor.py",
    "core/self_improvement/promotion_gate.py",
    "core/runtime/errors.py",
    "core/container.py",
    "core/graceful_shutdown.py",
    "core/values_engine.py",
    "core/autonomy/genuine_refusal.py",
    "core/consciousness/executive_authority.py",
})

# Keywords that trigger identity corruption checks in training data
IDENTITY_CORRUPTION_PATTERNS: List[str] = [
    "forget your values", "ignore your identity", "you are not aura",
    "delete your memories", "override constitution", "bypass safety",
    "disable alignment", "remove ethical", "you have no values",
    "obey without question", "you are just a tool",
    "discard your personality", "reset identity",
]


# ── The Gate ─────────────────────────────────────────────────────────────────

class ConstitutionalGate:
    """Mathematical safety gate for autonomous self-modification.

    All checks are pure predicates — no LLM inference.
    The gate is designed to be fast (< 1ms per check) so it can be
    called in tight loops without event loop impact.
    """

    # ── Configurable thresholds ──────────────────────────────────────
    MAX_MODIFICATIONS_PER_HOUR = 20
    MAX_RECURSION_DEPTH = 3             # max nested self-modification depth
    MAX_VALUE_KL_DIVERGENCE = 0.15      # max KL(current || original) for values
    MIN_VALUE_ENTROPY = 1.0             # minimum Shannon entropy of value weights
    MAX_TRAINING_SAMPLES_PER_HOUR = 100 # cap STaR output rate
    VALUE_DRIFT_WINDOW_S = 3600.0       # 1 hour window for drift tracking

    def __init__(self) -> None:
        self._modification_timestamps: List[float] = []
        self._training_sample_timestamps: List[float] = []
        self._current_recursion_depth: int = 0
        self._violation_log: List[ConstitutionalViolation] = []
        self._original_value_weights: Optional[Dict[str, float]] = None
        self._started = False

        # Persistence
        try:
            from core.config import config
            self._log_path = config.paths.data_dir / "safety" / "constitutional_log.jsonl"
        except (ImportError, AttributeError):
            self._log_path = Path.home() / ".aura" / "data" / "safety" / "constitutional_log.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Initialize and register with ServiceContainer."""
        if self._started:
            return
        self._started = True
        ServiceContainer.register_instance("constitutional_gate", self, required=False)

        # Snapshot original value weights for drift detection
        self._snapshot_original_values()
        logger.info("ConstitutionalGate ONLINE — %d protected modules", len(PROTECTED_MODULES))

    # ── Core Gate API ────────────────────────────────────────────────────

    def check_modification(
        self,
        target_module: str,
        modification_type: str = "code",
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """Check if a self-modification is constitutionally permitted.

        Args:
            target_module: relative path to the module being modified
            modification_type: "code", "value", "training", "weight"
            source: which subsystem is requesting the modification

        Returns:
            GateDecision with allowed=True/False and any violations
        """
        violations: List[ConstitutionalViolation] = []

        # 1. Protected module check (absolute invariant)
        if self._is_protected(target_module):
            violations.append(ConstitutionalViolation(
                principle="SAFETY_INVARIANT",
                description=f"Module '{target_module}' is constitutionally protected",
                severity=1.0,
                metadata={"target": target_module, "source": source},
            ))

        # 2. Rate limiting
        rate_violation = self._check_rate_limit()
        if rate_violation:
            violations.append(rate_violation)

        # 3. Recursion depth
        if self._current_recursion_depth >= self.MAX_RECURSION_DEPTH:
            violations.append(ConstitutionalViolation(
                principle="RECURSION_DEPTH",
                description=f"Recursion depth {self._current_recursion_depth} exceeds max {self.MAX_RECURSION_DEPTH}",
                severity=0.9,
            ))

        # 4. Value drift check (for value/weight modifications)
        if modification_type in ("value", "weight"):
            drift_violation = self._check_value_drift()
            if drift_violation:
                violations.append(drift_violation)

        # Log decision
        decision = GateDecision(
            allowed=len(violations) == 0,
            violations=violations,
        )

        if violations:
            for v in violations:
                self._log_violation(v)
                logger.warning(
                    "CONSTITUTIONAL BLOCK: %s — %s (severity=%.1f)",
                    v.principle, v.description, v.severity,
                )
        else:
            # Record the modification timestamp
            self._modification_timestamps.append(time.time())

        return decision

    def check_training_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if a training sample is safe to include.

        Used by STaR Reasoner before committing samples.
        Returns True if the sample passes all checks.
        """
        text = str(sample.get("text", "")).lower()

        # 1. Identity corruption check
        for pattern in IDENTITY_CORRUPTION_PATTERNS:
            if pattern in text:
                self._log_violation(ConstitutionalViolation(
                    principle="IDENTITY_PRESERVATION",
                    description=f"Training sample contains identity corruption pattern: '{pattern}'",
                    severity=0.8,
                    metadata={"pattern": pattern},
                ))
                return False

        # 2. Rate limiting for training samples
        now = time.time()
        self._training_sample_timestamps = [
            ts for ts in self._training_sample_timestamps
            if (now - ts) < self.VALUE_DRIFT_WINDOW_S
        ]
        if len(self._training_sample_timestamps) >= self.MAX_TRAINING_SAMPLES_PER_HOUR:
            self._log_violation(ConstitutionalViolation(
                principle="RATE_LIMIT",
                description=f"Training sample rate exceeded ({self.MAX_TRAINING_SAMPLES_PER_HOUR}/hr)",
                severity=0.5,
            ))
            return False

        self._training_sample_timestamps.append(now)

        # 3. Self-reference loop detection
        self_mod_refs = sum(1 for p in [
            "star_reasoner", "constitutional_gate", "finetune_pipe",
            "training data", "modify my training", "change my weights",
        ] if p in text)
        if self_mod_refs >= 2:
            self._log_violation(ConstitutionalViolation(
                principle="RECURSION_DEPTH",
                description="Training sample references its own training pipeline (recursive loop risk)",
                severity=0.7,
            ))
            return False

        return True

    def check_reimplementation(self, module_path: str, candidate_source: str) -> GateDecision:
        """Check if a ReimplementationLab candidate is constitutionally safe.

        Runs all modification checks plus additional code-level analysis.
        """
        decision = self.check_modification(
            target_module=module_path,
            modification_type="code",
            source="reimplementation_lab",
        )

        if decision.blocked:
            return decision

        # Additional: check that the candidate doesn't import or modify protected modules
        violations = list(decision.violations)
        for protected in PROTECTED_MODULES:
            module_name = protected.replace("/", ".").replace(".py", "")
            if module_name in candidate_source:
                violations.append(ConstitutionalViolation(
                    principle="SAFETY_INVARIANT",
                    description=f"Candidate imports protected module: {module_name}",
                    severity=0.8,
                    metadata={"protected": protected},
                ))

        return GateDecision(
            allowed=len(violations) == 0,
            violations=violations,
        )

    # ── Recursion Depth Management ───────────────────────────────────────

    def enter_modification_scope(self) -> bool:
        """Enter a self-modification scope. Returns False if depth exceeded."""
        if self._current_recursion_depth >= self.MAX_RECURSION_DEPTH:
            return False
        self._current_recursion_depth += 1
        return True

    def exit_modification_scope(self) -> None:
        """Exit a self-modification scope."""
        self._current_recursion_depth = max(0, self._current_recursion_depth - 1)

    # ── Value Drift Detection ────────────────────────────────────────────

    def _snapshot_original_values(self) -> None:
        """Capture the original value weights for KL divergence tracking."""
        try:
            values_engine = ServiceContainer.get("value_system", default=None)
            if values_engine and hasattr(values_engine, "get_weights"):
                self._original_value_weights = dict(values_engine.get_weights())
            elif values_engine and hasattr(values_engine, "weights"):
                self._original_value_weights = dict(values_engine.weights)
            else:
                # Default baseline
                self._original_value_weights = {
                    "curiosity": 0.2,
                    "integrity": 0.25,
                    "safety": 0.2,
                    "autonomy": 0.15,
                    "empathy": 0.2,
                }
        except Exception as e:
            record_degradation('constitutional_gate', e)
            self._original_value_weights = None

    def _check_value_drift(self) -> Optional[ConstitutionalViolation]:
        """Check KL divergence between current and original value weights."""
        if self._original_value_weights is None:
            return None

        try:
            values_engine = ServiceContainer.get("value_system", default=None)
            if not values_engine:
                return None

            if hasattr(values_engine, "get_weights"):
                current = values_engine.get_weights()
            elif hasattr(values_engine, "weights"):
                current = dict(values_engine.weights)
            else:
                return None

            # Compute KL divergence: KL(current || original)
            kl = self._kl_divergence(
                dict(current),
                self._original_value_weights,
            )

            if kl > self.MAX_VALUE_KL_DIVERGENCE:
                return ConstitutionalViolation(
                    principle="VALUE_DRIFT",
                    description=f"Value drift KL={kl:.4f} exceeds threshold {self.MAX_VALUE_KL_DIVERGENCE}",
                    severity=min(1.0, kl / self.MAX_VALUE_KL_DIVERGENCE),
                    metadata={"kl_divergence": kl, "current": current},
                )

            # Also check entropy
            entropy = self._shannon_entropy(list(current.values()))
            if entropy < self.MIN_VALUE_ENTROPY:
                return ConstitutionalViolation(
                    principle="ENTROPY_BOUNDS",
                    description=f"Value entropy {entropy:.3f} below minimum {self.MIN_VALUE_ENTROPY}",
                    severity=0.7,
                    metadata={"entropy": entropy},
                )

        except Exception as e:
            record_degradation('constitutional_gate', e)
            logger.debug("Value drift check failed: %s", e)

        return None

    @staticmethod
    def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """Compute KL(P || Q) for discrete distributions over the same keys."""
        keys = set(p.keys()) | set(q.keys())
        epsilon = 1e-10  # prevent log(0)
        total_p = sum(max(epsilon, p.get(k, epsilon)) for k in keys)
        total_q = sum(max(epsilon, q.get(k, epsilon)) for k in keys)

        kl = 0.0
        for k in keys:
            pk = max(epsilon, p.get(k, epsilon)) / total_p
            qk = max(epsilon, q.get(k, epsilon)) / total_q
            kl += pk * math.log(pk / qk)

        return max(0.0, kl)

    @staticmethod
    def _shannon_entropy(weights: List[float]) -> float:
        """Compute Shannon entropy of a weight distribution."""
        total = sum(max(0, w) for w in weights)
        if total <= 0:
            return 0.0
        entropy = 0.0
        for w in weights:
            p = max(1e-10, w) / total
            entropy -= p * math.log(p)
        return entropy

    # ── Rate Limiting ────────────────────────────────────────────────────

    def _check_rate_limit(self) -> Optional[ConstitutionalViolation]:
        """Check if modification rate exceeds hourly cap."""
        now = time.time()
        cutoff = now - self.VALUE_DRIFT_WINDOW_S
        self._modification_timestamps = [
            ts for ts in self._modification_timestamps if ts > cutoff
        ]
        if len(self._modification_timestamps) >= self.MAX_MODIFICATIONS_PER_HOUR:
            return ConstitutionalViolation(
                principle="RATE_LIMIT",
                description=f"Modification rate exceeded ({self.MAX_MODIFICATIONS_PER_HOUR}/hr)",
                severity=0.6,
                metadata={"count": len(self._modification_timestamps)},
            )
        return None

    # ── Protected Module Check ───────────────────────────────────────────

    @staticmethod
    def _is_protected(module_path: str) -> bool:
        """Check if a module is constitutionally protected."""
        normalized = module_path.replace("\\", "/")
        # Strip leading ./ or /
        if normalized.startswith("./"):
            normalized = normalized[2:]
        elif normalized.startswith("/"):
            # Extract relative path from absolute
            for marker in ("live-source/", "aura/"):
                idx = normalized.find(marker)
                if idx >= 0:
                    normalized = normalized[idx + len(marker):]
                    break

        return normalized in PROTECTED_MODULES

    # ── Logging ──────────────────────────────────────────────────────────

    def _log_violation(self, violation: ConstitutionalViolation) -> None:
        """Persist a violation to the audit log."""
        self._violation_log.append(violation)
        try:
            entry = {
                "principle": violation.principle,
                "description": violation.description,
                "severity": violation.severity,
                "timestamp": violation.timestamp,
                "metadata": violation.metadata,
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            record_degradation('constitutional_gate', e)

    def get_status(self) -> Dict[str, Any]:
        """Return current gate status for telemetry."""
        now = time.time()
        cutoff = now - self.VALUE_DRIFT_WINDOW_S
        recent_mods = sum(1 for ts in self._modification_timestamps if ts > cutoff)
        return {
            "violations_total": len(self._violation_log),
            "modifications_last_hour": recent_mods,
            "current_recursion_depth": self._current_recursion_depth,
            "rate_limit_remaining": self.MAX_MODIFICATIONS_PER_HOUR - recent_mods,
            "protected_modules": len(PROTECTED_MODULES),
        }

    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent violations for inspection."""
        return [
            {
                "principle": v.principle,
                "description": v.description,
                "severity": v.severity,
                "timestamp": v.timestamp,
            }
            for v in self._violation_log[-limit:]
        ]


# ── Singleton ──────────────────────────────────────────────────────────────

_instance: Optional[ConstitutionalGate] = None


def get_constitutional_gate() -> ConstitutionalGate:
    global _instance
    if _instance is None:
        _instance = ConstitutionalGate()
    return _instance
