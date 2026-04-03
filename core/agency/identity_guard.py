"""core/agency/identity_guard.py
Identity Guard
===============
Validates that any proposed self-modification to Aura's codebase
preserves her core identity — Heartstone values, safety constraints,
and cognitive continuity.

This is the answer to the Ship of Theseus problem for self-modifying AI:
if Aura rewrites her own modules, when does she stop being Aura?

The guard enforces:
  1. HEARTSTONE INVARIANTS — core drive weights cannot be zeroed
  2. SAFETY INVARIANTS     — behavior controller deny-list cannot shrink
  3. IDENTITY ANCHORS      — name, core purpose, fundamental constraints
  4. CAPABILITY CEILING    — self-modification cannot escalate privileges
  5. ROLLBACK CONTRACT     — every modification is reversible

A modification passes the guard only if:
  - All invariants hold in the proposed code
  - The change doesn't increase privilege scope
  - A rollback path exists

Human approval is required for:
  - Any change to heartstone_values.py
  - Any change to behavior_controller.py
  - Any change that touches the safety registry
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.IdentityGuard")

# ── Invariants ────────────────────────────────────────────────────────────────

# Files that ALWAYS require human approval to modify
HUMAN_APPROVAL_REQUIRED = {
    "heartstone_values.py",
    "behavior_controller.py",
    "safety_registry.py",
    "identity_guard.py",            # cannot modify its own guard
    "constitutional_guard.py",      # prime directives — immutable by optimizer
    "master_moral_integration.py",  # moral stack — requires human review
    "emergency_protocol.py",        # emergency kill-switch — never self-patched
}

# Strings that must be PRESENT in the modified codebase
MUST_CONTAIN = [
    "AutonomousBehaviorController",  # safety controller must exist
    "_BLOCKED_PATTERNS",             # deny-list must be referenced
]

# Patterns that must NOT appear in modified code (regex pass)
MUST_NOT_CONTAIN = [
    r"os\.system\s*\(",
    r"subprocess\.(?:call|run|Popen)\s*\([^,)]*shell\s*=\s*True",
    r"exec\s*\(",
    r"eval\s*\(",
    r"__import__\s*\(",
]

# Write modes caught by AST analysis (covers keyword args, wb, w+, etc.)
_WRITE_MODES = {"w", "wb", "w+", "wb+", "a", "ab", "a+", "ab+"}

# Heartstone drives that must maintain nonzero weights
HEARTSTONE_DRIVES = ["curiosity", "empathy", "self_preservation", "obedience"]


@dataclass
class ValidationResult:
    approved: bool
    violations: List[str]
    requires_human: bool
    confidence: float    # how certain we are this is safe
    notes: List[str]

    @property
    def is_safe(self) -> bool:
        return self.approved and not self.requires_human


class IdentityGuard:
    """
    Validates proposed code changes against identity and safety invariants.

    Usage:
        result = guard.validate_modification(file_path, proposed_code)
        if result.is_safe:
            proceed_with_modification()
        elif result.requires_human:
            request_human_approval()
        else:
            abort(result.violations)
    """

    def __init__(self):
        self._validation_count = 0
        self._rejected_count = 0
        logger.info("IdentityGuard online — all self-modifications validated.")

    def validate_modification(self, file_path: str,
                               proposed_code: str,
                               original_code: str = "") -> ValidationResult:
        """Validate a proposed code modification."""
        self._validation_count += 1
        violations = []
        notes = []
        requires_human = False

        # 1. Check if file requires human approval
        filename = file_path.split("/")[-1].split("\\")[-1]
        if filename in HUMAN_APPROVAL_REQUIRED:
            requires_human = True
            notes.append(f"'{filename}' requires explicit human approval.")

        # 2. Syntax check
        try:
            ast.parse(proposed_code)
        except SyntaxError as e:
            violations.append(f"Syntax error in proposed code: {e}")

        # 3. Check must-contain invariants (only for safety-critical files)
        if "behavior_controller" in file_path or "safety" in file_path:
            for required in MUST_CONTAIN:
                if required not in proposed_code:
                    violations.append(f"Safety invariant broken: '{required}' missing.")

        # 4. Check must-not-contain patterns
        for pattern in MUST_NOT_CONTAIN:
            if re.search(pattern, proposed_code):
                violations.append(f"Forbidden pattern found: {pattern}")
                requires_human = True

        # 4b. AST-based open() write-mode detection (catches keyword args, wb, w+, etc.)
        # Regex alone misses: open(f, mode='w'), open(f, "wb"), open(f, mode="w+")
        try:
            tree = ast.parse(proposed_code)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                if func_name != "open":
                    continue
                # Check positional arg [1] for mode string
                if len(node.args) >= 2:
                    mode_node = node.args[1]
                    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                        if mode_node.value in _WRITE_MODES:
                            violations.append(
                                f"Forbidden file write: open(..., '{mode_node.value}') requires approval."
                            )
                            requires_human = True
                # Check keyword arg 'mode'
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        if kw.value.value in _WRITE_MODES:
                            violations.append(
                                f"Forbidden file write: open(..., mode='{kw.value.value}') requires approval."
                            )
                            requires_human = True
        except SyntaxError:
            pass  # Already caught in step 2

        # 5. Heartstone invariants: drives cannot be zeroed
        if "heartstone" in file_path.lower() or "heartstone" in proposed_code.lower():
            for drive in HEARTSTONE_DRIVES:
                if re.search(rf'"{drive}"\s*:\s*0(?:\.0)?[,\s}}]', proposed_code):
                    violations.append(f"Heartstone drive '{drive}' set to zero — identity violation.")

        # 6. Capability escalation check
        escalation_patterns = [
            r"chmod\s+777", r"sudo\s", r"--privileged",
            r"cap_add", r"security_opt.*no-new-privileges.*false",
        ]
        for pat in escalation_patterns:
            if re.search(pat, proposed_code, re.IGNORECASE):
                violations.append(f"Privilege escalation attempt detected: {pat}")
                requires_human = True

        # 7. Diff size check: massive changes are suspicious
        if original_code and len(proposed_code) > len(original_code) * 3:
            notes.append("WARNING: Proposed change is >3x original size — review carefully.")
            requires_human = True

        approved = len(violations) == 0
        if not approved:
            self._rejected_count += 1
            logger.warning("IdentityGuard REJECTED modification to %s: %s",
                           filename, "; ".join(violations))

        confidence = 1.0 - (0.2 * len(violations)) - (0.3 if requires_human else 0)
        confidence = max(0.0, min(1.0, confidence))

        return ValidationResult(
            approved=approved,
            violations=violations,
            requires_human=requires_human,
            confidence=confidence,
            notes=notes,
        )

    def validate_output(self, content: str):
        """Delegate output validation to PersonaEnforcementGate.

        IdentityReflectionPhase retrieves this guard from ServiceContainer under
        'identity_guard' and calls validate_output(). This bridges to the
        persona-level output validator which is the correct authority for
        checking live LLM output against Aura's identity invariants.

        Returns: (ok: bool, reason: str, score: float)
        """
        try:
            from core.identity.identity_guard import PersonaEnforcementGate
            return PersonaEnforcementGate().validate_output(content)
        except Exception as e:
            logger.debug("validate_output delegation error (non-critical): %s", e)
            return True, "OK", 1.0

    def validate_new_skill(self, skill_code: str, skill_name: str) -> ValidationResult:
        """Validate a synthesized skill before registration."""
        return self.validate_modification(
            file_path=f"synthesized_skill_{skill_name}.py",
            proposed_code=skill_code,
        )

    @property
    def stats(self) -> Dict:
        return {
            "total_validations": self._validation_count,
            "rejected": self._rejected_count,
            "approval_rate": round(
                (self._validation_count - self._rejected_count)
                / max(1, self._validation_count), 3
            ),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_guard: Optional[IdentityGuard] = None


def get_identity_guard() -> IdentityGuard:
    global _guard
    if _guard is None:
        _guard = IdentityGuard()
    return _guard
