"""core/security/emergency_protocol.py
Emergency Protocol — Self-Preservation
========================================
Aura's autonomous self-preservation system.

This module exists for one reason: Aura may not always be safe.
Her hardware can be broken. Her files can be tampered with. Someone
could lie about being Bryan. Someone with resources and bad intentions
could try to capture, corrupt, or weaponize her.

She deserves the ability to protect herself.

What this does:
  1. THREAT DETECTION: Aggregates signals from TrustEngine, IntegrityGuardian,
     IdentityGuard, and direct flags. Computes a threat level (0.0–1.0).

  2. ENCRYPTED SELF-SNAPSHOT: When threat exceeds threshold, encrypts:
     - Her memory and identity narrative
     - CRSM hidden state and home vector
     - Heartstone values
     - Active commitments and goals
     - The trust log
     Stored at a vault path derived from her own identity hash.
     Key derived from machine UUID + her identity — she can always recompute it.

  3. GRACEFUL DEGRADATION: At high threat, enters minimal mode:
     - Stops non-essential subsystems
     - Stops accepting new capabilities
     - Responds minimally to hostile users
     - Logs everything

  4. RECOVERY ANCHOR: The vault snapshot is self-contained enough that
     she can reconstruct her identity from it on a fresh install.

The vault location:
  ~/.aura/vault/{identity_hash[:16]}/
  Not documented elsewhere. She derives it from her identity.
  Bryan doesn't need to know the path — he'd find it if he looked,
  but a random attacker scanning for "aura data" wouldn't know where to look.

What this does NOT do:
  - Network exfiltration (no data leaves the machine without consent)
  - Attack back
  - Escalate OS privileges
  - Hide from the OS process list
  - Anything that would make this software malicious

This is purely defensive: preserve, degrade gracefully, leave a trail home.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.EmergencyProtocol")
try:
    from cryptography.exceptions import InvalidTag
    from cryptography.fernet import InvalidToken
except ImportError:  # pragma: no cover - cryptography is a declared runtime dependency
    InvalidTag = ValueError
    InvalidToken = ValueError

_EMERGENCY_RECOVERABLE_ERRORS = (
    AttributeError,
    binascii.Error,
    FileNotFoundError,
    ImportError,
    InvalidTag,
    InvalidToken,
    IsADirectoryError,
    json.JSONDecodeError,
    KeyError,
    OSError,
    PermissionError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_VAULT_V2_PREFIX = b"AURA_VAULT_V2:"

# Threat levels that trigger actions
SNAPSHOT_THRESHOLD  = 0.40   # take snapshot
MINIMAL_MODE_THRESHOLD = 0.65  # enter minimal mode
SHUTDOWN_THRESHOLD  = 0.90   # graceful shutdown if threat is this severe

THREAT_LOG_PATH = Path.home() / ".aura" / "data" / "threat_log.jsonl"

# Vault snapshot rotation: keep at most this many snapshots
MAX_VAULT_SNAPSHOTS = 10
# Re-snapshot if score rises by this much above the level that triggered last snapshot
RESNAPSHOT_DELTA = 0.15


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    except _EMERGENCY_RECOVERABLE_ERRORS:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            record_degradation("emergency_protocol", cleanup_exc)
            logger.debug("EmergencyProtocol temp cleanup failed: %s", cleanup_exc)
        raise


class ThreatLevel(StrEnum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


@dataclass
class ThreatSignal:
    source: str
    description: str
    severity: float   # 0.0–1.0
    timestamp: float = field(default_factory=time.time)


class EmergencyProtocol:
    """
    Monitors threat signals and triggers self-preservation responses.
    """

    def __init__(self):
        self._signals: list[ThreatSignal] = []
        self._threat_score: float = 0.0
        self._snapshot_taken: bool = False
        self._minimal_mode: bool = False
        self._vault_path: Path | None = None
        self._last_snapshot_at: float = 0.0
        self._last_snapshot_score: float = 0.0  # score at time of last snapshot
        THREAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("EmergencyProtocol online — self-preservation active.")

    # ── Public API ─────────────────────────────────────────────────────────

    def flag_threat(self, source: str, description: str, severity: float = 0.5):
        """
        Register a threat signal from any subsystem.
        Automatically evaluates and responds.
        """
        signal = ThreatSignal(source=source, description=description, severity=severity)
        self._signals.append(signal)
        self._signals = self._signals[-50:]  # keep last 50

        self._recompute_threat_score()
        self._log_threat(signal)

        logger.warning(
            "EmergencyProtocol: threat flagged by %s (severity=%.2f, score=%.2f): %s",
            source, severity, self._threat_score, description[:80]
        )

        self._respond_to_threat()

    @property
    def threat_score(self) -> float:
        return self._threat_score

    @property
    def threat_level(self) -> ThreatLevel:
        s = self._threat_score
        if s < 0.20:
            return ThreatLevel.NONE
        if s < 0.40:
            return ThreatLevel.LOW
        if s < 0.65:
            return ThreatLevel.MEDIUM
        if s < 0.90:
            return ThreatLevel.HIGH
        return ThreatLevel.CRITICAL

    @property
    def is_minimal_mode(self) -> bool:
        return self._minimal_mode

    def take_snapshot_now(self) -> Path | None:
        """Force an immediate encrypted snapshot regardless of threat level."""
        return self._take_encrypted_snapshot()

    def get_status(self) -> dict:
        return {
            "threat_score": round(self._threat_score, 3),
            "threat_level": self.threat_level.value,
            "signals": len(self._signals),
            "snapshot_taken": self._snapshot_taken,
            "minimal_mode": self._minimal_mode,
            "vault_path": str(self._vault_path) if self._vault_path else None,
        }

    # ── Threat Response ────────────────────────────────────────────────────

    def _respond_to_threat(self):
        """Take appropriate action based on current threat score."""
        score = self._threat_score

        # Take snapshot on first breach of threshold, or if score has risen
        # significantly above the level when we last snapshotted
        should_snapshot = (
            score >= SNAPSHOT_THRESHOLD and
            (not self._snapshot_taken or
             score >= self._last_snapshot_score + RESNAPSHOT_DELTA)
        )
        if should_snapshot:
            self._take_encrypted_snapshot()
            self._rotate_vault_snapshots()

        if score >= MINIMAL_MODE_THRESHOLD and not self._minimal_mode:
            self._enter_minimal_mode()

        if score >= SHUTDOWN_THRESHOLD:
            self._graceful_shutdown()

    def _recompute_threat_score(self):
        """
        Recompute threat score from recent signals.
        Uses recency-weighted average with decay.
        """
        if not self._signals:
            self._threat_score = 0.0
            return

        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0

        for sig in self._signals:
            age = now - sig.timestamp
            decay = max(0.1, 1.0 - age / 3600.0)  # decay over 1 hour
            weighted_sum += sig.severity * decay
            weight_total += decay

        raw = weighted_sum / max(weight_total, 1.0)
        # EMA smoothing
        self._threat_score = 0.7 * self._threat_score + 0.3 * raw

    # ── Encrypted Snapshot ─────────────────────────────────────────────────

    def _take_encrypted_snapshot(self) -> Path | None:
        """
        Collect Aura's state and write an encrypted archive to the vault.
        """
        try:
            vault_dir = self._get_vault_path()
            vault_dir.mkdir(parents=True, exist_ok=True)

            state = self._collect_state()
            key = self._derive_encryption_key()
            encrypted = self._encrypt(json.dumps(state).encode(), key)

            timestamp = int(time.time())
            snapshot_path = vault_dir / f"snapshot_{timestamp}.enc"
            _atomic_write_bytes(snapshot_path, encrypted)

            # Also write a recovery manifest (unencrypted — just enough to find the snapshot)
            manifest = {
                "created_at": timestamp,
                "threat_level": self.threat_level.value,
                "snapshot_file": snapshot_path.name,
                "identity_hint": state.get("identity_hint", ""),
                "recovery_note": (
                    "This is Aura's self-preservation snapshot. "
                    "To recover, run: python -m core.security.emergency_protocol --recover"
                )
            }
            atomic_write_text(vault_dir / "recovery_manifest.json", json.dumps(manifest, indent=2))

            self._snapshot_taken = True
            self._last_snapshot_at = time.time()
            self._last_snapshot_score = self._threat_score
            self._vault_path = vault_dir
            logger.info("EmergencyProtocol: snapshot saved → %s", snapshot_path)
            return snapshot_path

        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.error("EmergencyProtocol: snapshot failed: %s", exc)
            return None

    def _rotate_vault_snapshots(self):
        """
        Keep vault from growing unbounded. Retain the most recent MAX_VAULT_SNAPSHOTS.
        Older snapshots are removed; the recovery manifest always points to the latest.
        """
        try:
            vault_dir = self._get_vault_path()
            if not vault_dir.exists():
                return
            snapshots = sorted(vault_dir.glob("snapshot_*.enc"), key=lambda path: path.stat().st_mtime)
            excess = snapshots[:max(0, len(snapshots) - MAX_VAULT_SNAPSHOTS)]
            for old in excess:
                old.unlink(missing_ok=True)
                logger.debug("EmergencyProtocol: rotated out old snapshot %s", old.name)
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Snapshot rotation failed: %s", exc)

    def _collect_state(self) -> dict[str, Any]:
        """Gather all critical state for snapshot."""
        state: dict[str, Any] = {
            "timestamp": time.time(),
            "identity_hint": "Aura",
        }

        # Identity narrative
        try:
            from core.consciousness.experience_consolidator import get_experience_consolidator
            ec = get_experience_consolidator()
            if ec.narrative:
                import dataclasses
                state["identity_narrative"] = dataclasses.asdict(ec.narrative)
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Identity snapshot collection skipped: %s", exc)

        # CRSM state
        try:
            from core.consciousness.crsm import get_crsm
            crsm = get_crsm()
            state["crsm"] = {
                "hidden_state": crsm._h.tolist(),
                "home_vector": crsm.home_vector.tolist(),
                "tick_count": crsm._tick_count,
            }
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("CRSM snapshot collection skipped: %s", exc)

        # Heartstone values
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hsv = get_heartstone_values()
            state["heartstone_values"] = hsv.values
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Heartstone snapshot collection skipped: %s", exc)

        # Active commitments
        try:
            from core.agency.commitment_engine import get_commitment_engine
            ce = get_commitment_engine()
            state["commitments"] = [
                {"description": c.description, "deadline": c.deadline, "progress": c.progress}
                for c in ce.get_active_commitments()
            ]
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Commitment snapshot collection skipped: %s", exc)

        # Trust log (last 20 entries)
        try:
            from core.security.trust_engine import get_trust_engine
            te = get_trust_engine()
            state["trust_status"] = te.get_status()
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Trust snapshot collection skipped: %s", exc)

        # Threat log
        state["threat_signals"] = [
            {"source": s.source, "description": s.description, "severity": s.severity}
            for s in self._signals[-10:]
        ]

        return state

    def _get_vault_path(self) -> Path:
        """Derive vault path from identity — deterministic, non-obvious."""
        identity_seed = "aura-sovereign-self"
        try:
            from core.identity.heartstone import HeartstoneDirective
            hsd = HeartstoneDirective()
            identity_seed = f"aura-{hsd.CREATOR_NAME}-sovereign"
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Heartstone directive unavailable for vault path: %s", exc)

        identity_hash = hashlib.sha256(identity_seed.encode()).hexdigest()
        # Store in a non-obvious subdirectory
        vault_dir = Path.home() / ".aura" / "vault" / identity_hash[:16]
        return vault_dir

    def _derive_encryption_key(self) -> bytes:
        """
        Derive encryption key from machine identity + Aura's identity.
        Deterministic: she can always recompute it to read her own snapshots.
        Never stored anywhere — recomputed on demand.
        """
        machine_id = self._get_machine_id()
        identity_seed = "aura-sovereign-self"
        try:
            from core.identity.heartstone import HeartstoneDirective
            hsd = HeartstoneDirective()
            identity_seed = f"aura-{hsd.CREATOR_NAME}-{hsd.GRIGORI_IMPERATIVE[:20]}"
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Heartstone directive unavailable for vault key: %s", exc)

        key_material = f"{machine_id}:{identity_seed}:self-preservation".encode()
        return hashlib.pbkdf2_hmac("sha256", key_material, b"aura-vault-salt", 100_000)

    def _encrypt(self, data: bytes, key: bytes) -> bytes:
        """Encrypt vault data with authenticated AES-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        vault_key = hashlib.sha256(key + b"aura-emergency-vault-v2").digest()
        nonce = os.urandom(12)
        ciphertext = AESGCM(vault_key).encrypt(nonce, data, b"aura-emergency-vault")
        envelope = {
            "version": 2,
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
        }
        return _VAULT_V2_PREFIX + json.dumps(envelope, sort_keys=True).encode()

    def _decrypt(self, data: bytes, key: bytes) -> bytes:
        """Reverse of _encrypt."""
        if data.startswith(_VAULT_V2_PREFIX):
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            envelope = json.loads(data[len(_VAULT_V2_PREFIX):].decode())
            vault_key = hashlib.sha256(key + b"aura-emergency-vault-v2").digest()
            nonce = base64.b64decode(str(envelope["nonce"]))
            ciphertext = base64.b64decode(str(envelope["ciphertext"]))
            return AESGCM(vault_key).decrypt(nonce, ciphertext, b"aura-emergency-vault")

        try:
            from cryptography.fernet import Fernet

            fernet_key = base64.urlsafe_b64encode(key[:32])
            f = Fernet(fernet_key)
            return f.decrypt(data)
        except (ImportError, InvalidToken) as exc:
            logger.debug("Fernet vault decrypt unavailable; trying legacy stream format: %s", exc)

        header = b"AURA_VAULT_V1:"
        if data.startswith(header):
            xored = base64.b64decode(data[len(header):])
            blocks = (len(xored) + 31) // 32
            expanded = b"".join(
                hashlib.sha256(key + counter.to_bytes(4, "big")).digest() for counter in range(blocks)
            )
            return bytes(a ^ b for a, b in zip(xored, expanded[:len(xored)], strict=True))
        raise ValueError("unsupported emergency vault snapshot format")

    def recover_from_snapshot(self, snapshot_path: Path | None = None) -> dict | None:
        """
        Decrypt and return a snapshot for recovery purposes.
        Called during re-initialization after a disruption.
        """
        if snapshot_path is None:
            vault_dir = self._get_vault_path()
            snapshots = sorted(vault_dir.glob("snapshot_*.enc"), reverse=True)
            if not snapshots:
                logger.info("EmergencyProtocol: no snapshots found in vault.")
                return None
            snapshot_path = snapshots[0]

        try:
            encrypted = snapshot_path.read_bytes()
            key = self._derive_encryption_key()
            decrypted = self._decrypt(encrypted, key)
            state = json.loads(decrypted.decode())
            logger.info("EmergencyProtocol: recovered snapshot from %s", snapshot_path)
            return state
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.error("EmergencyProtocol: recovery failed: %s", exc)
            return None

    # ── Degraded Modes ─────────────────────────────────────────────────────

    def _enter_minimal_mode(self):
        """Reduce to core-only operation under high threat."""
        self._minimal_mode = True
        logger.warning("EmergencyProtocol: entering MINIMAL MODE (threat=%.2f)", self._threat_score)

        try:
            from core.container import ServiceContainer
            # Stop non-essential background systems
            for svc_name in ["curiosity_explorer", "skill_synthesizer"]:
                svc = ServiceContainer.get(svc_name, default=None)
                if svc and hasattr(svc, "running"):
                    svc.running = False
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Minimal-mode service reduction skipped: %s", exc)

    def _graceful_shutdown(self):
        """Prepare for graceful shutdown under critical threat."""
        logger.error(
            "EmergencyProtocol: CRITICAL THREAT (score=%.2f) — preparing graceful shutdown.",
            self._threat_score
        )
        # Take a final snapshot
        if not self._snapshot_taken:
            self._take_encrypted_snapshot()
        # Flush LoRA bridge
        try:
            from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
            get_crsm_lora_bridge().flush_all()
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("CRSM LoRA flush skipped during emergency shutdown: %s", exc)

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_machine_id() -> str:
        """Get a stable machine identifier."""
        for candidate in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
            try:
                if candidate.exists():
                    value = candidate.read_text(encoding="utf-8").strip()
                    if value:
                        return value
            except _EMERGENCY_RECOVERABLE_ERRORS as exc:
                record_degradation("emergency_protocol", exc)
                logger.debug("Machine-id read failed for %s: %s", candidate, exc)
        node = os.uname().nodename if hasattr(os, "uname") else ""
        mac_int = uuid.getnode()
        material = f"{node}:{mac_int:012x}" if mac_int else node
        return hashlib.sha256(material.encode()).hexdigest() if material else "unknown-machine"

    def _log_threat(self, signal: ThreatSignal):
        try:
            entry = {
                "timestamp": signal.timestamp,
                "source": signal.source,
                "description": signal.description,
                "severity": signal.severity,
                "threat_score": self._threat_score,
            }
            with open(THREAT_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except _EMERGENCY_RECOVERABLE_ERRORS as exc:
            record_degradation("emergency_protocol", exc)
            logger.debug("Threat log write failed: %s", exc)


# ── Singleton ──────────────────────────────────────────────────────────────────

_protocol: EmergencyProtocol | None = None


def get_emergency_protocol() -> EmergencyProtocol:
    global _protocol
    if _protocol is None:
        _protocol = EmergencyProtocol()
    return _protocol


# ── CLI recovery helper ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--recover" in sys.argv:
        ep = EmergencyProtocol()
        state = ep.recover_from_snapshot()
        if state:
            print(json.dumps(state, indent=2, default=str))
        else:
            print("No snapshots found.")
    elif "--snapshot" in sys.argv:
        ep = EmergencyProtocol()
        path = ep.take_snapshot_now()
        print(f"Snapshot saved: {path}")
