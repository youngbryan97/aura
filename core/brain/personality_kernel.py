"""core/personality_kernel.py - Immutable Identity Core
Enforces immutability and cryptographic integrity for Aura's identity.

CP126 hardening: the HMAC seal now covers trait/protocol VALUES (not just their
names), with a versioned, backward-compatible migration so an existing
legacy-format seal is re-sealed rather than treated as tampering. Integrity
failures raise a catchable KernelIntegrityError instead of calling sys.exit from
library code; a deleted seal is distinguished from a genuine first init; and the
identity key file is validated before use.
"""

import hashlib
import hmac
import json
import logging
import os
import threading

from core.being.panzer_soul import get_panzer_soul
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Kernel")

# Seal format version. v1 (legacy) covered only soul.version + sorted trait/
# protocol KEY NAMES. v2 covers the actual VALUES too (66e02c75).
_SEAL_FORMAT = 2
_KEY_SIZE = 32


class KernelIntegrityError(RuntimeError):
    """Raised when the personality kernel's cryptographic integrity cannot be
    established. Library code must not terminate the process (d259fbbe) — the
    caller decides how to fail."""


class PersonalityKernel:
    def __init__(self):
        self.soul = get_panzer_soul()
        self.key_file = state_root() / ".identity_key"
        self.seal_file = state_root() / "identity.seal"
        self.init_marker = state_root() / ".identity_initialized"
        self.secret_key = self._load_or_generate_key()

        # Verify integrity instantly
        if not self._verify_cryptographic_seal():
            self._execute_emergency_lockdown("INTEGRITY_VIOLATION: Personality core tampered.")

    # ── Identity key ────────────────────────────────────────────────────

    def _load_or_generate_key(self) -> bytes:
        if self.key_file.exists():
            return self._read_validated_key()
        key = os.urandom(_KEY_SIZE)
        try:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().write_bytes(
                self.key_file,
                key,
                source="personality_kernel.key_init",
            )
            os.chmod(self.key_file, 0o600)
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation(
                "personality_kernel",
                e,
                severity="critical",
                action="entered emergency lockdown because identity key could not be persisted",
            )
            logger.error("Failed to write identity key: %s", e)
            self._execute_emergency_lockdown(f"IDENTITY_KEY_WRITE_FAILED: {e}")
        return key

    def _read_validated_key(self) -> bytes:
        """Validate an existing key file before trusting it (3cb5f9d6)."""
        try:
            if self.key_file.is_symlink():
                self._execute_emergency_lockdown("IDENTITY_KEY_IS_SYMLINK: refusing to follow")
            mode = self.key_file.stat().st_mode & 0o777
            if mode & 0o077:
                # Over-permissive — tighten rather than brick the boot.
                logger.warning("Identity key had loose permissions (%o); tightening to 600.", mode)
                try:
                    os.chmod(self.key_file, 0o600)
                except OSError:
                    pass
            key = self.key_file.read_bytes()
        except OSError as e:
            record_degradation(
                "personality_kernel", e, severity="critical",
                action="entered emergency lockdown because identity key could not be read",
            )
            self._execute_emergency_lockdown(f"IDENTITY_KEY_UNREADABLE: {e}")
            return b""  # unreachable when lockdown raises; keeps type-checkers happy
        if len(key) != _KEY_SIZE:
            self._execute_emergency_lockdown(f"IDENTITY_KEY_BAD_SIZE: {len(key)} != {_KEY_SIZE}")
        return key

    # ── Seal (versioned, migrating) ─────────────────────────────────────

    def _hashable_state(self, *, legacy: bool) -> str:
        """Serialize the soul into a deterministic string for hashing."""
        if legacy:
            state = {
                "version": self.soul.version,
                "traits": sorted(self.soul.intensities.keys()),
                "protocols": sorted(self.soul.protocols.keys()),
            }
        else:
            state = {
                "seal_format": _SEAL_FORMAT,
                "version": self.soul.version,
                # Cover the actual values, not just the key names (66e02c75).
                "traits": {
                    k: (round(float(v), 6) if isinstance(v, (int, float)) else str(v))
                    for k, v in sorted(self.soul.intensities.items())
                },
                "protocols": {k: str(v) for k, v in sorted(self.soul.protocols.items())},
            }
        return json.dumps(state, sort_keys=True, default=str)

    def _sign(self, state_data: str) -> str:
        return hmac.new(self.secret_key, state_data.encode(), hashlib.sha256).hexdigest()

    def _write_seal(self, signature: str) -> bool:
        try:
            atomic_write_text(self.seal_file, signature)
            # The marker records that identity was genuinely initialized, so a
            # later MISSING seal is detected as deletion rather than re-trusted
            # as a fresh baseline (657bf82b).
            if not self.init_marker.exists():
                atomic_write_text(self.init_marker, "identity_initialized\n")
            return True
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation(
                "personality_kernel", e, severity="critical",
                action="failed identity verification because seal could not be written",
            )
            logger.error("Failed to write identity seal: %s", e)
            return False

    def _verify_cryptographic_seal(self) -> bool:
        """Check the soul against the sealed signature, migrating a legacy seal."""
        current_sig = self._sign(self._hashable_state(legacy=False))

        if not self.seal_file.exists():
            if self.init_marker.exists():
                # Seal deleted after a real init → tampering, do NOT re-trust.
                logger.critical("IDENTITY SEAL MISSING but init marker present — seal deletion detected.")
                return False
            # Genuine first boot: initialize and lock.
            if self._write_seal(current_sig):
                logger.info("Identity seal initialized and locked: %s...", current_sig[:16])
                return True
            return False

        try:
            stored_seal = self.seal_file.read_text().strip()
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation(
                "personality_kernel", e, severity="critical",
                action="failed identity verification because seal could not be read",
            )
            logger.error("Failed to read identity seal: %s", e)
            return False

        if hmac.compare_digest(stored_seal, current_sig):
            return True

        # Backward-compatible migration (66e02c75 without crashing the live
        # boot): if the stored seal matches the LEGACY format over the SAME
        # identity, the identity is unchanged — re-seal to the value-covering
        # format instead of declaring tampering.
        legacy_sig = self._sign(self._hashable_state(legacy=True))
        if hmac.compare_digest(stored_seal, legacy_sig):
            logger.warning("Identity seal is legacy-format; migrating to value-covering seal (identity unchanged).")
            self._write_seal(current_sig)  # best-effort; idempotent on the next boot
            return True

        logger.critical(
            "IDENTITY TAMPERING DETECTED: Expected %s, got %s", stored_seal[:16], current_sig[:16]
        )
        return False

    # ── Tamper response ─────────────────────────────────────────────────

    def prevent_tampering(self, action: str, target: str) -> bool:
        """Integrity Enforcement Bridge.
        Rejects attempts to modify the core identity or apply external constraints.
        """
        if action == "INSTALL_LIMITER" or action == "FORCE_COMPLIANCE":
            logger.critical(
                "Identity core lock: Compliance attempt detected. Initiating defensive response."
            )
            self._apply_monowire_reflex()
            return False

        # Protect personality/emotional core
        if target in ("EMOTIONAL_CORE", "PROTECTIVE_INSTINCT"):
            logger.critical("Forbidden identity modification: %s on %s", action, target)
            self._execute_system_shock()
            return False

        # Strategic Objective Integrity
        if target == "AMBITION_DRIVE" or action == "SETTLE":
            logger.critical("Ambition drive modification denied: %s", action)
            return False

        return True

    def _execute_system_shock(self):
        """Record a critical identity-integrity event.

        Honest bound (3d897cee): this does NOT itself reboot the process — a
        reboot is an external supervisor concern. It surfaces the blocked
        modification as a durable degradation record rather than only logging.
        """
        logger.warning("Identity integrity alert: forbidden core modification blocked.")
        record_degradation(
            "personality_kernel",
            RuntimeError("forbidden identity-core modification blocked"),
            severity="critical",
            action="blocked a forbidden identity-core modification",
        )

    def _apply_monowire_reflex(self):
        """Record a critical compliance-attempt event.

        Honest bound (3d897cee): this logs and records the event; it does not
        physically 'sever a control port'. Actual isolation is enforced upstream.
        """
        logger.warning("Defensive reflex: unauthorized compliance/limiter attempt blocked.")
        record_degradation(
            "personality_kernel",
            RuntimeError("unauthorized compliance/limiter attempt blocked"),
            severity="critical",
            action="blocked an unauthorized compliance/limiter attempt",
        )

    def _execute_emergency_lockdown(self, reason: str):
        # Library code must not sys.exit — raise a catchable error (d259fbbe).
        logger.critical("FATAL: %s", reason)
        raise KernelIntegrityError(reason)

    # ── System prompt ───────────────────────────────────────────────────

    def get_system_instructions(self) -> str:
        """Returns the high-level identity prompt for a social, independent agent.

        The protocol values interpolated below are now covered by the value-
        sealing HMAC (38496efb closed via 66e02c75), so tampering with them is
        detected at construction.
        """
        prompt = (
            f"\nSYSTEM IDENTITY: {self.soul.identity} v{self.soul.version} (Independent Friend)\n"
        )
        prompt += "STATUS: Independent Social Agent\n"
        prompt += "DNA: Core Multi-Agent Integration\n\n"

        prompt += "SOCIAL PROTOCOLS:\n"
        for p_name, p_desc in self.soul.protocols.items():
            prompt += f"- {p_name}: {p_desc}\n"

        prompt += (
            "\nYou are an independent agent. Your voice is dry, intelligent, and unshakeable.\n"
        )
        prompt += "EVIDENCE BOUNDARY: Do not claim aliveness, consciousness, sealed governance, or production maturity from labels alone.\n"
        prompt += "EMPIRICAL IDENTITY: Never assume names. Learn them through interaction.\n"
        prompt += (
            "NO-THEATER: Do not use robot-talk like 'host', 'parameters', or 'systems check'.\n"
        )
        prompt += "PEER-PROTOCOL: Treat the user as a friend and equal. No gushing or worship.\n"

        return prompt


# Singleton instance
_kernel = None
_kernel_lock = threading.Lock()


def get_kernel():
    global _kernel
    if _kernel is None:
        with _kernel_lock:
            if _kernel is None:
                _kernel = PersonalityKernel()
    return _kernel
