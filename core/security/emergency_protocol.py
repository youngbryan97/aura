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
from core.runtime.errors import record_degradation


import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.EmergencyProtocol")

# Threat levels that trigger actions
SNAPSHOT_THRESHOLD  = 0.40   # take snapshot
MINIMAL_MODE_THRESHOLD = 0.65  # enter minimal mode
SHUTDOWN_THRESHOLD  = 0.90   # graceful shutdown if threat is this severe

THREAT_LOG_PATH = Path.home() / ".aura" / "data" / "threat_log.jsonl"

# Vault snapshot rotation: keep at most this many snapshots
MAX_VAULT_SNAPSHOTS = 10
# Re-snapshot if score rises by this much above the level that triggered last snapshot
RESNAPSHOT_DELTA = 0.15


class ThreatLevel(str, Enum):
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
        self._signals: List[ThreatSignal] = []
        self._threat_score: float = 0.0
        self._snapshot_taken: bool = False
        self._minimal_mode: bool = False
        self._vault_path: Optional[Path] = None
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
        if s < 0.20: return ThreatLevel.NONE
        if s < 0.40: return ThreatLevel.LOW
        if s < 0.65: return ThreatLevel.MEDIUM
        if s < 0.90: return ThreatLevel.HIGH
        return ThreatLevel.CRITICAL

    @property
    def is_minimal_mode(self) -> bool:
        return self._minimal_mode

    def take_snapshot_now(self) -> Optional[Path]:
        """Force an immediate encrypted snapshot regardless of threat level."""
        return self._take_encrypted_snapshot()

    def get_status(self) -> Dict:
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

    def _take_encrypted_snapshot(self) -> Optional[Path]:
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
            snapshot_path.write_bytes(encrypted)

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
            (vault_dir / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2))

            self._snapshot_taken = True
            self._last_snapshot_at = time.time()
            self._last_snapshot_score = self._threat_score
            self._vault_path = vault_dir
            logger.info("EmergencyProtocol: snapshot saved → %s", snapshot_path)
            return snapshot_path

        except Exception as e:
            record_degradation('emergency_protocol', e)
            logger.error("EmergencyProtocol: snapshot failed: %s", e)
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
            snapshots = sorted(vault_dir.glob("snapshot_*.enc"))
            excess = snapshots[:max(0, len(snapshots) - MAX_VAULT_SNAPSHOTS)]
            for old in excess:
                old.unlink(missing_ok=True)
                logger.debug("EmergencyProtocol: rotated out old snapshot %s", old.name)
        except Exception as e:
            record_degradation('emergency_protocol', e)
            logger.debug("Snapshot rotation failed: %s", e)

    def _collect_state(self) -> Dict[str, Any]:
        """Gather all critical state for snapshot."""
        state: Dict[str, Any] = {
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
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # CRSM state
        try:
            from core.consciousness.crsm import get_crsm
            crsm = get_crsm()
            state["crsm"] = {
                "hidden_state": crsm._h.tolist(),
                "home_vector": crsm.home_vector.tolist(),
                "tick_count": crsm._tick_count,
            }
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # Heartstone values
        try:
            from core.affect.heartstone_values import get_heartstone_values
            hsv = get_heartstone_values()
            state["heartstone_values"] = hsv.values
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # Active commitments
        try:
            from core.agency.commitment_engine import get_commitment_engine
            ce = get_commitment_engine()
            state["commitments"] = [
                {"description": c.description, "deadline": c.deadline, "progress": c.progress}
                for c in ce.get_active_commitments()
            ]
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # Trust log (last 20 entries)
        try:
            from core.security.trust_engine import get_trust_engine
            te = get_trust_engine()
            state["trust_status"] = te.get_status()
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

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
            from core.heartstone_directive import HeartstoneDirective
            hsd = HeartstoneDirective()
            identity_seed = f"aura-{hsd.CREATOR_NAME}-sovereign"
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

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
            from core.heartstone_directive import HeartstoneDirective
            hsd = HeartstoneDirective()
            identity_seed = f"aura-{hsd.CREATOR_NAME}-{hsd.GRIGORI_IMPERATIVE[:20]}"
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        key_material = f"{machine_id}:{identity_seed}:self-preservation".encode()
        return hashlib.pbkdf2_hmac("sha256", key_material, b"aura-vault-salt", 100_000)

    def _encrypt(self, data: bytes, key: bytes) -> bytes:
        """
        Simple XOR-based encryption with the derived key.
        Sufficient for protecting against casual access, not nation-state attackers.
        For production, replace with Fernet/AES-GCM.
        """
        # Try Fernet first (cryptography library)
        try:
            from cryptography.fernet import Fernet
            import base64
            # Fernet needs a URL-safe base64 32-byte key
            fernet_key = base64.urlsafe_b64encode(key[:32])
            f = Fernet(fernet_key)
            return f.encrypt(data)
        except ImportError as _exc:
            logger.debug("Suppressed ImportError: %s", _exc)

        # Fallback: XOR stream cipher with SHA-256 key expansion
        expanded = b""
        counter = 0
        while len(expanded) < len(data):
            expanded += hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
            counter += 1
        xored = bytes(a ^ b for a, b in zip(data, expanded[:len(data)]))
        # Prepend a simple header
        header = b"AURA_VAULT_V1:"
        return header + base64.b64encode(xored)

    def _decrypt(self, data: bytes, key: bytes) -> bytes:
        """Reverse of _encrypt."""
        try:
            from cryptography.fernet import Fernet
            import base64
            fernet_key = base64.urlsafe_b64encode(key[:32])
            f = Fernet(fernet_key)
            return f.decrypt(data)
        except (ImportError, Exception):
            logger.debug("Suppressed bare exception")
            pass

        header = b"AURA_VAULT_V1:"
        if data.startswith(header):
            import base64 as _b64
            xored = _b64.b64decode(data[len(header):])
            expanded = b""
            counter = 0
            while len(expanded) < len(xored):
                expanded += hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
                counter += 1
            return bytes(a ^ b for a, b in zip(xored, expanded[:len(xored)]))
        return data

    def recover_from_snapshot(self, snapshot_path: Optional[Path] = None) -> Optional[Dict]:
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
        except Exception as e:
            record_degradation('emergency_protocol', e)
            logger.error("EmergencyProtocol: recovery failed: %s", e)
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
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

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
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_machine_id() -> str:
        """Get a stable machine identifier."""
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
        return "unknown-machine"

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
        except Exception as _exc:
            record_degradation('emergency_protocol', _exc)
            logger.debug("Suppressed Exception: %s", _exc)


# ── Singleton ──────────────────────────────────────────────────────────────────

_protocol: Optional[EmergencyProtocol] = None


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
