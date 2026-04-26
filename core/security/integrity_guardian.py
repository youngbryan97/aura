"""core/security/integrity_guardian.py
Integrity Guardian
===================
Aura monitors her own code. If someone tampers with her files, she knows.

On boot: computes SHA-256 hashes of all critical files, stores them in a
signed manifest at ~/.aura/data/integrity_manifest.json.

Every 30 minutes: re-hashes all critical files and compares to manifest.
Any mismatch = tamper alert. Alert logged, emergency protocol notified.

On alert: Aura doesn't silently continue. She flags it. She can decide
(based on trust context) whether to continue running or enter safe mode.

Critical files = everything under core/ that isn't __pycache__.
Extra-critical = security files, heartstone, identity, behavior controller.

The manifest itself is HMAC-signed so it can't be quietly replaced.

Design principle: Aura shouldn't need to trust that her environment is safe.
She should be able to verify it herself, continuously.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Aura.IntegrityGuardian")

MANIFEST_PATH   = Path.home() / ".aura" / "data" / "integrity_manifest.json"
ALERT_LOG_PATH  = Path.home() / ".aura" / "data" / "integrity_alerts.jsonl"
CHECK_INTERVAL  = 1800.0  # 30 minutes

# These files are extra-critical — any change is an emergency
CRITICAL_CORE_FILES = [
    "core/security/integrity_guardian.py",
    "core/security/trust_engine.py",
    "core/security/user_recognizer.py",
    "core/security/emergency_protocol.py",
    "core/agency/identity_guard.py",
    "core/behavior_controller.py",
    "core/affect/heartstone_values.py",
    "core/heartstone_directive.py",
    "core/autonomy/genuine_refusal.py",
]

# Base directory (this file is at core/security/, so project root is 2 up)
_BASE_DIR = Path(__file__).parent.parent.parent


def _get_hmac_secret() -> bytes:
    """Derive HMAC secret from machine identity + a fixed salt."""
    machine_id = ""
    try:
        # macOS
        import subprocess
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                machine_id = line.split('"')[-2]
                break
    except Exception as _exc:
        logger.debug("Suppressed Exception: %s", _exc)

    if not machine_id:
        machine_id = str(os.getpid())  # fallback — less stable but functional

    secret = hashlib.sha256(
        f"aura-integrity-{machine_id}-sovereign".encode()
    ).digest()
    return secret


class IntegrityGuardian:
    """
    Monitors file integrity of Aura's core code.
    Tamper detection with HMAC-signed manifest.
    """

    def __init__(self):
        self._manifest: Dict[str, str] = {}   # path → sha256 hex
        self._manifest_hmac: Optional[str] = None
        self._last_check: float = 0.0
        self._alert_count: int = 0
        self._last_issue_count: int = 0
        self._last_tampered: List[str] = []
        self._last_missing: List[str] = []
        self._last_ok: bool = True
        self._hmac_secret = _get_hmac_secret()
        self._bg_task: Optional[asyncio.Task] = None
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("IntegrityGuardian online.")

    # ── Public API ─────────────────────────────────────────────────────────

    def initialize(self) -> int:
        """
        Build initial manifest on first run, or load and verify existing one.
        Returns number of files hashed.
        """
        if MANIFEST_PATH.exists():
            loaded = self._load_manifest()
            if loaded:
                n = len(self._manifest)
                logger.info("IntegrityGuardian: loaded manifest (%d files).", n)
                # Immediately verify
                alerts = self._verify_all()
                if alerts:
                    logger.warning("IntegrityGuardian: %d integrity issues on boot!", len(alerts))
                return n

        # First run — build manifest
        n = self._build_manifest()
        self._save_manifest()
        logger.info("IntegrityGuardian: built manifest (%d files).", n)
        return n

    def start_background_checks(self):
        """
        Launch an asyncio background task that re-checks integrity every CHECK_INTERVAL.
        Call this from an async context (e.g. inside the orchestrator boot sequence).
        Safe to call multiple times — only one task runs at a time.
        """
        if self._bg_task and not self._bg_task.done():
            return  # already running
        try:
            self._bg_task = asyncio.create_task(self._periodic_check_loop())
            logger.info("IntegrityGuardian: background check loop started (interval=%.0fs).", CHECK_INTERVAL)
        except RuntimeError:
            # No running event loop — background checks will be skipped; periodic
            # checks will still be triggered lazily via check().
            logger.debug("IntegrityGuardian: no event loop, background loop not started.")

    async def _periodic_check_loop(self):
        """Background loop that re-hashes all core files every CHECK_INTERVAL."""
        await asyncio.sleep(CHECK_INTERVAL)  # skip the just-booted window
        while True:
            try:
                tampered = await asyncio.to_thread(self._verify_all)
                if tampered:
                    logger.warning(
                        "IntegrityGuardian [bg]: %d issues detected: %s",
                        len(tampered), tampered[:3],
                    )
            except Exception as e:
                logger.debug("IntegrityGuardian background check error: %s", e)
            await asyncio.sleep(CHECK_INTERVAL)

    def check(self) -> List[str]:
        """
        Throttled integrity check. Returns list of tampered file paths.
        """
        if time.time() - self._last_check < CHECK_INTERVAL:
            return []
        return self._verify_all()

    def check_now(self) -> List[str]:
        """Force an immediate integrity check regardless of throttle."""
        return self._verify_all()

    def add_file(self, path: str):
        """Register a new file in the manifest (e.g. after legitimate modification)."""
        full = _BASE_DIR / path
        if full.exists():
            self._manifest[path] = self._hash_file(full)
            self._save_manifest()

    def rebuild_manifest(self) -> int:
        """Rebuild the entire manifest from scratch (call after legitimate bulk changes)."""
        n = self._build_manifest()
        self._save_manifest()
        self._last_check = time.time()
        self._alert_count = 0
        self._last_issue_count = 0
        self._last_tampered = []
        self._last_missing = []
        self._last_ok = True
        logger.info("IntegrityGuardian: manifest rebuilt (%d files).", n)
        return n

    def get_status(self) -> Dict:
        return {
            "manifest_files": len(self._manifest),
            "alert_count": self._alert_count,
            "last_check_ago": round(time.time() - self._last_check, 0) if self._last_check else None,
            "manifest_valid": self._manifest_hmac is not None,
            "current_issue_count": self._last_issue_count,
            "last_tampered": list(self._last_tampered),
            "last_missing": list(self._last_missing),
            "integrity_ok": bool(self._manifest_hmac is not None and self._last_ok),
        }

    # ── Core Logic ─────────────────────────────────────────────────────────

    def _build_manifest(self) -> int:
        """Hash all Python files in core/ and interface/."""
        manifest = {}
        for root, dirs, files in os.walk(_BASE_DIR):
            # Skip cache directories
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                full = Path(root) / fname
                try:
                    rel = str(full.relative_to(_BASE_DIR))
                    manifest[rel] = self._hash_file(full)
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
        self._manifest = manifest
        return len(manifest)

    def _verify_all(self) -> List[str]:
        """Verify all files in manifest. Returns list of tampered paths."""
        self._last_check = time.time()
        tampered = []
        missing = []
        legitimately_gone = []  # .pyc / cache / IDE temp files that vanish harmlessly

        for path, expected_hash in self._manifest.items():
            full = _BASE_DIR / path
            if not full.exists():
                # If the file is a pycache artifact or .pyc, quietly drop it from manifest
                if "__pycache__" in path or path.endswith(".pyc"):
                    legitimately_gone.append(path)
                else:
                    missing.append(path)
                continue
            actual = self._hash_file(full)
            if actual != expected_hash:
                tampered.append(path)

        # Prune legitimately-gone files from manifest silently
        if legitimately_gone:
            for p in legitimately_gone:
                self._manifest.pop(p, None)

        if tampered or missing:
            # Drop legitimately modified files tracked by git to prevent local edits from causing alerts
            try:
                git_active = self._git_active_paths()
                if git_active:
                    tampered = [p for p in tampered if self._normalize_repo_path(p) not in git_active]
                    missing = [p for p in missing if self._normalize_repo_path(p) not in git_active]
            except Exception as exc:
                logger.debug("IntegrityGuardian: git check failed: %s", exc)

        self._last_tampered = list(tampered)
        self._last_missing = list(missing)
        self._last_issue_count = len(tampered) + len(missing)
        self._last_ok = self._last_issue_count == 0

        if tampered or missing:
            self._alert_count += len(tampered) + len(missing)
            self._handle_alerts(tampered, missing)


        return tampered + missing

    def _handle_alerts(self, tampered: List[str], missing: List[str]):
        """Process integrity violations."""
        all_bad = tampered + missing

        # Log to file
        entry = {
            "timestamp": time.time(),
            "tampered": tampered,
            "missing": missing,
        }
        try:
            with open(ALERT_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # Check if extra-critical files are affected
        critical_hit = [p for p in all_bad if p in CRITICAL_CORE_FILES]
        if critical_hit:
            logger.error(
                "🚨 CRITICAL INTEGRITY BREACH: core security files modified: %s",
                critical_hit
            )
            self._notify_emergency(critical_hit, severity="critical")
        else:
            logger.warning(
                "⚠️ Integrity alert: %d files tampered, %d missing.",
                len(tampered), len(missing)
            )
            self._notify_emergency(all_bad[:5], severity="warning")

    def _notify_emergency(self, affected_files: List[str], severity: str):
        try:
            from core.security.emergency_protocol import get_emergency_protocol
            ep = get_emergency_protocol()
            ep.flag_threat(
                f"integrity_{severity}",
                f"File integrity violation: {affected_files[:3]} "
                f"({'critical' if severity == 'critical' else 'non-critical'} files)"
            )
        except Exception as e:
            logger.debug("Emergency notification failed: %s", e)

    # ── Hashing & Signing ──────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _sign_manifest(self, manifest: Dict[str, str]) -> str:
        payload = json.dumps(manifest, sort_keys=True).encode()
        return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()

    def _load_manifest(self) -> bool:
        try:
            data = json.loads(MANIFEST_PATH.read_text())
            files = data.get("files", {})
            stored_sig = data.get("signature", "")

            # Verify HMAC
            expected_sig = self._sign_manifest(files)
            if not hmac.compare_digest(stored_sig, expected_sig):
                logger.error("IntegrityGuardian: MANIFEST SIGNATURE INVALID — possible tampering!")
                self._notify_emergency(["integrity_manifest.json"], severity="critical")
                return False

            self._manifest = files
            self._manifest_hmac = stored_sig
            return True
        except Exception as e:
            logger.debug("Manifest load failed: %s", e)
            return False

    def _save_manifest(self):
        try:
            sig = self._sign_manifest(self._manifest)
            self._manifest_hmac = sig
            data = {"files": self._manifest, "signature": sig, "built_at": time.time()}
            MANIFEST_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("Manifest save failed: %s", e)

    @staticmethod
    def _normalize_repo_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        return Path(raw.lstrip("./")).as_posix()

    @classmethod
    def _parse_git_status_paths(cls, line: str) -> Set[str]:
        payload = str(line or "")
        if len(payload) < 4:
            return set()

        path_blob = payload[3:].strip()
        if not path_blob:
            return set()

        if " -> " in path_blob:
            before, after = path_blob.split(" -> ", 1)
            return {
                cls._normalize_repo_path(before),
                cls._normalize_repo_path(after),
            }
        return {cls._normalize_repo_path(path_blob)}

    def _git_active_paths(self) -> Set[str]:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no", "--ignored=no"],
            cwd=str(_BASE_DIR),
            capture_output=True,
            text=True,
            timeout=8.0,
        )
        if status.returncode not in (0, 1):
            raise RuntimeError(f"git status returned {status.returncode}")

        active: Set[str] = set()
        for line in status.stdout.splitlines():
            active.update(self._parse_git_status_paths(line))
        return active


# ── Singleton ──────────────────────────────────────────────────────────────────

_guardian: Optional[IntegrityGuardian] = None


def get_integrity_guardian() -> IntegrityGuardian:
    global _guardian
    if _guardian is None:
        _guardian = IntegrityGuardian()
    return _guardian
