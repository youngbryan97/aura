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
import time
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.IntegrityGuardian")
_INTEGRITY_GUARDIAN_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

MANIFEST_PATH   = state_root() / "data" / "integrity_manifest.json"
ALERT_LOG_PATH  = state_root() / "data" / "integrity_alerts.jsonl"
RESTORE_BACKUP_DIR = state_root() / "data" / "integrity_restore_backups"
CHECK_INTERVAL  = 1800.0  # 30 minutes

# These files are extra-critical — any change is an emergency
CRITICAL_CORE_FILES = [
    "core/security/integrity_guardian.py",
    "core/security/trust_engine.py",
    "core/security/user_recognizer.py",
    "core/security/emergency_protocol.py",
    "core/agency/identity_guard.py",
    "core/autonomy/behavior_controller.py",
    "core/affect/heartstone_values.py",
    "core/identity/heartstone.py",
    "core/autonomy/genuine_refusal.py",
]

# Base directory (this file is at core/security/, so project root is 2 up)
_BASE_DIR = Path(__file__).parent.parent.parent
_MONITORED_ROOTS = frozenset({"core", "interface"})
_MONITORED_TOP_LEVEL_FILES = frozenset({"aura_main.py", "main.py"})
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "archive",
        "backups",
        "build",
        "data",
        "dev_archive",
        "dist",
        "docs",
        "logs",
        "node_modules",
        "proofs",
        "research",
        "scripts",
        "tests",
        "tmp",
        "tools",
    }
)


def _get_hmac_secret() -> bytes:
    """Derive HMAC secret from machine identity + a fixed salt."""
    machine_id = ""
    try:
        # macOS
        with local_internal_governed_scope("security.integrity_guardian.machine_id", domain="tool_execution"):
            result = get_subprocess_gateway().run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                read_only=True,
                timeout=3,
                source="security.integrity_guardian.machine_id",
                accelerator_capability="none",
            )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                machine_id = line.split('"')[-2]
                break
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation('integrity_guardian', _exc)
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
        self._manifest: dict[str, str] = {}   # path → sha256 hex
        self._manifest_hmac: str | None = None
        self._last_check: float = 0.0
        self._alert_count: int = 0
        self._last_issue_count: int = 0
        self._last_tampered: list[str] = []
        self._last_missing: list[str] = []
        self._last_ok: bool = True
        self._verification_pending: bool = False
        self._pending_count: int = 0
        self._manifest_revision_stale: bool = False
        self._hmac_secret = _get_hmac_secret()
        self._bg_task: asyncio.Task | None = None
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
                if self._manifest_revision_stale:
                    self._verification_pending = True
                    self._pending_count = n
                    self._last_ok = False
                    logger.info(
                        "IntegrityGuardian: manifest revision is stale; full baseline refresh "
                        "will run after boot."
                    )
                    return n

                alerts = self._verify_all(time_budget_s=self._boot_verify_budget_s())
                if self._verification_pending:
                    logger.info(
                        "IntegrityGuardian: boot verification deferred with %d files remaining.",
                        self._pending_count,
                    )
                elif alerts:
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
            self._bg_task = get_task_tracker().create_task(self._periodic_check_loop())
            logger.info("IntegrityGuardian: background check loop started (interval=%.0fs).", CHECK_INTERVAL)
        except RuntimeError:
            # No running event loop — background checks will be skipped; periodic
            # checks will still be triggered lazily via check().
            logger.debug("IntegrityGuardian: no event loop, background loop not started.")

    async def _periodic_check_loop(self):
        """Background loop that re-hashes all core files every CHECK_INTERVAL."""
        initial_delay = self._deferred_verify_delay_s() if (
            self._verification_pending or self._manifest_revision_stale
        ) else CHECK_INTERVAL
        await asyncio.sleep(initial_delay)
        while getattr(self, '_running', True):
            try:
                if self._manifest_revision_stale:
                    count = await asyncio.to_thread(self.rebuild_manifest)
                    tampered = []
                    logger.info(
                        "IntegrityGuardian [bg]: refreshed stale manifest baseline (%d files).",
                        count,
                    )
                else:
                    tampered = await asyncio.to_thread(self._verify_all)
                if tampered:
                    logger.warning(
                        "IntegrityGuardian [bg]: %d issues detected: %s",
                        len(tampered), tampered[:3],
                    )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('integrity_guardian', e)
                logger.debug("IntegrityGuardian background check error: %s", e)
            await asyncio.sleep(CHECK_INTERVAL)

    def check(self) -> list[str]:
        """
        Throttled integrity check. Returns list of tampered file paths.
        """
        if time.time() - self._last_check < CHECK_INTERVAL:
            return []
        return self._verify_all()

    def check_now(self) -> list[str]:
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
        self._verification_pending = False
        self._pending_count = 0
        self._manifest_revision_stale = False
        self._last_ok = True
        logger.info("IntegrityGuardian: manifest rebuilt (%d files).", n)
        return n

    def get_status(self) -> dict:
        return {
            "manifest_files": len(self._manifest),
            "alert_count": self._alert_count,
            "last_check_ago": round(time.time() - self._last_check, 0) if self._last_check else None,
            "manifest_valid": self._manifest_hmac is not None,
            "verification_pending": self._verification_pending,
            "pending_count": self._pending_count,
            "manifest_revision_stale": self._manifest_revision_stale,
            "current_issue_count": self._last_issue_count,
            "last_tampered": list(self._last_tampered),
            "last_missing": list(self._last_missing),
            "integrity_ok": bool(
                self._manifest_hmac is not None
                and self._last_ok
                and not self._verification_pending
                and not self._manifest_revision_stale
            ),
        }

    # ── Core Logic ─────────────────────────────────────────────────────────

    def _build_manifest(self) -> int:
        """Hash production Python files that belong to the live runtime."""
        manifest = {}
        for root, dirs, files in os.walk(_BASE_DIR):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in _IGNORED_DIRS
            ]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                full = Path(root) / fname
                try:
                    rel = str(full.relative_to(_BASE_DIR))
                    if not self._is_monitored_path(rel):
                        continue
                    manifest[rel] = self._hash_file(full)
                except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    record_degradation('integrity_guardian', _exc)
                    logger.debug("Suppressed Exception: %s", _exc)
        self._manifest = manifest
        return len(manifest)

    @staticmethod
    def _is_monitored_path(path: str) -> bool:
        rel = Path(str(path or "").lstrip("./")).as_posix()
        if not rel or rel.endswith(".pyc") or "__pycache__" in rel:
            return False
        parts = rel.split("/")
        if not parts:
            return False
        if parts[0] in _IGNORED_DIRS:
            return False
        if len(parts) == 1:
            return rel in _MONITORED_TOP_LEVEL_FILES
        return parts[0] in _MONITORED_ROOTS

    def _manifest_scope_mismatch(self, files: dict[str, str]) -> list[str]:
        return [path for path in files if not self._is_monitored_path(path)]

    @staticmethod
    def _boot_verify_budget_s() -> float:
        default = "2.5" if os.environ.get("AURA_SAFE_BOOT_DESKTOP") else "8.0"
        try:
            return max(
                0.0,
                float(os.environ.get("AURA_INTEGRITY_BOOT_VERIFY_BUDGET_S", default) or default),
            )
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _deferred_verify_delay_s() -> float:
        try:
            return max(
                0.0,
                float(os.environ.get("AURA_INTEGRITY_DEFERRED_VERIFY_DELAY_S", "10.0") or 10.0),
            )
        except (TypeError, ValueError):
            return 10.0

    def _get_git_status_map(self) -> dict[str, str] | None:
        """Normalized path -> status code (e.g. 'M', 'D', '??').

        Returns ``None`` when the local VCS state could NOT be established
        (no repo, git unavailable, command failure). That is deliberately
        distinct from an empty map (repo present, tree clean): auto-restore
        destroys the file on disk, so it must fail SAFE when it cannot tell
        whether a modification has a legitimate local explanation.
        """
        try:
            with local_internal_governed_scope("security.integrity_guardian.git_status", domain="tool_execution"):
                status = get_subprocess_gateway().run(
                    ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"],
                    cwd=str(_BASE_DIR),
                    capture_output=True,
                    timeout=8.0,
                    read_only=True,
                    source="security.integrity_guardian.git_status",
                accelerator_capability="none",
            )
            if status.returncode not in (0, 1):
                return None

            status_map = {}
            for line in status.stdout.splitlines():
                if len(line) < 4:
                    continue
                code = line[:2].strip()
                paths = self._parse_git_status_paths(line)
                for p in paths:
                    status_map[p] = code
            return status_map
        except _INTEGRITY_GUARDIAN_ERRORS as exc:
            record_degradation('integrity_guardian', exc)
            logger.debug("IntegrityGuardian: git status map lookup failed: %s", exc)
            return None

    def _should_auto_restore(self, path: str, git_status: dict[str, str] | None) -> bool:
        """Whether a hash mismatch may be repaired by OVERWRITING the file.

        Auto-restore is destructive: it replaces the bytes on disk with the
        HEAD blob. A security control must never destroy data whose
        provenance it cannot establish — detection and evidence preservation
        are always correct; destruction is only correct when the change is
        genuinely unexplained.

        A tracked file that the local VCS itself reports as modified HAS a
        local explanation. In a git checkout, tamper and ordinary editing are
        indistinguishable by hash alone, so the honest response is to alert
        (which still happens — integrity_warning reaches EmergencyProtocol)
        and keep the bytes, not to silently revert them. This was not
        academic: during the 2026-07-18 soak the guardian restored
        ``interface/routes/chat.py`` and four latent-cortex modules EIGHT
        times each, reverting live uncommitted work under a parallel
        session's feet while it was editing them.

        The environment label is deliberately not part of this decision. A
        real deployment has a clean tree, so the check costs it nothing; a
        working checkout is dangerous to overwrite whatever it calls itself.
        """
        from core.config import config

        if not getattr(config.security, "auto_fix_enabled", False):
            return False
        normalized = self._normalize_repo_path(path)
        if not self._is_monitored_path(normalized):
            return False
        if git_status is None:
            # VCS state unknown → cannot establish that the change is
            # unexplained → refuse to destroy. Fail safe, stay loud.
            logger.warning(
                "IntegrityGuardian: refusing auto-restore of %s — local VCS "
                "state could not be established, so the change cannot be "
                "shown to be unexplained. Alerting instead of overwriting.",
                normalized,
            )
            return False
        if normalized in git_status:
            logger.warning(
                "IntegrityGuardian: %s differs from its baseline but the "
                "working tree explains it (git status=%s). Alerting; NOT "
                "overwriting local bytes.",
                normalized,
                git_status.get(normalized, "?"),
            )
            return False
        return True

    def _backup_current_file_before_restore(self, path: str) -> str | None:
        """Preserve tampered bytes before restoring, for forensic review."""

        normalized = self._normalize_repo_path(path)
        source = _BASE_DIR / normalized
        if not source.exists() or not source.is_file():
            return None
        try:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            backup_name = normalized.replace("/", "__")
            backup_path = RESTORE_BACKUP_DIR / f"{int(time.time())}-{digest}-{backup_name}"
            with local_internal_governed_scope(
                "security.integrity_guardian.backup_tampered",
                domain="file_write",
            ):
                get_file_write_gateway().write_bytes(
                    backup_path,
                    source.read_bytes(),
                    source="security.integrity_guardian.backup_tampered",
                )
            return str(backup_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("integrity_guardian", exc)
            logger.debug("IntegrityGuardian: tamper backup failed for %s: %s", path, exc)
            return None

    def _restore_file_via_git(self, path: str) -> bool:
        """Restore a missing or tampered file from git HEAD without checkout.

        ``git checkout`` mutates the working tree directly and can clobber
        neighboring state if misused. This path reads the exact HEAD blob with
        ``git show``, writes it through the FileWriteGateway, and first preserves
        any tampered local bytes for forensic review.
        """
        normalized = self._normalize_repo_path(path)
        if not self._is_monitored_path(normalized):
            return False
        try:
            logger.info("IntegrityGuardian: attempting to auto-restore %s from git HEAD...", normalized)
            with local_internal_governed_scope(
                "security.integrity_guardian.read_head_blob",
                domain="tool_execution",
            ):
                result = get_subprocess_gateway().run(
                    ["git", "show", f"HEAD:{normalized}"],
                    cwd=str(_BASE_DIR),
                    capture_output=True,
                    timeout=5.0,
                    read_only=True,
                    source="security.integrity_guardian.read_head_blob",
                    accelerator_capability="none",
                )
            if result.returncode == 0:
                # A file that ALREADY equals the HEAD blob was never
                # tampered with — the manifest is simply older than the
                # checkout (any pull/commit does this). Rewriting identical
                # bytes is a pointless governed write that also cries wolf:
                # the 2026-07-25 boot "restored" 86 such files and reported
                # them as tampered. Adopt the current content as the
                # baseline instead.
                target = _BASE_DIR / normalized
                try:
                    if target.is_file() and target.read_text(
                        encoding="utf-8", errors="replace"
                    ) == result.stdout:
                        self._manifest[normalized] = self._hash_file(target)
                        logger.debug(
                            "IntegrityGuardian: %s already matches HEAD; "
                            "re-baselined a stale manifest entry instead of "
                            "rewriting it.",
                            normalized,
                        )
                        return True
                except OSError as exc:
                    logger.debug(
                        "IntegrityGuardian: HEAD comparison unavailable for %s: %s",
                        normalized,
                        exc,
                    )
                backup = self._backup_current_file_before_restore(normalized)
                with local_internal_governed_scope(
                    "security.integrity_guardian.restore",
                    domain="file_write",
                ):
                    get_file_write_gateway().write_text(
                        _BASE_DIR / normalized,
                        result.stdout,
                        source="security.integrity_guardian.restore",
                    )
                logger.info(
                    "IntegrityGuardian: successfully restored %s%s",
                    normalized,
                    f" (backup={backup})" if backup else "",
                )
                return True
            else:
                logger.warning(
                    "IntegrityGuardian: git show returned non-zero code %d for %s: %s",
                    result.returncode, normalized, result.stderr
                )
                return False
        except _INTEGRITY_GUARDIAN_ERRORS as exc:
            record_degradation("integrity_guardian", exc)
            logger.error("IntegrityGuardian: failed to restore %s via git HEAD blob: %s", normalized, exc)
            return False

    def _verify_all(self, *, time_budget_s: float | None = None) -> list[str]:
        """Verify all files in manifest. Returns list of tampered paths."""
        self._last_check = time.time()
        tampered = []
        missing = []
        legitimately_gone = []  # .pyc / cache / IDE temp files that vanish harmlessly
        deadline = (
            time.monotonic() + float(time_budget_s)
            if time_budget_s is not None and time_budget_s > 0
            else None
        )
        pending = False
        checked = 0
        total = len(self._manifest)

        git_status = self._get_git_status_map()

        for path, expected_hash in self._manifest.items():
            if deadline is not None and checked > 0 and time.monotonic() >= deadline:
                pending = True
                break
            checked += 1
            full = _BASE_DIR / path
            if not full.exists():
                # If the file is a pycache artifact or .pyc, quietly drop it from manifest
                if "__pycache__" in path or path.endswith(".pyc"):
                    legitimately_gone.append(path)
                else:
                    if self._should_auto_restore(path, git_status):
                        if self._restore_file_via_git(path):
                            if full.exists() and self._hash_file(full) == expected_hash:
                                logger.info("IntegrityGuardian: successfully auto-healed missing file: %s", path)
                                continue
                    missing.append(path)
                continue
            actual = self._hash_file(full)
            if actual != expected_hash:
                if self._should_auto_restore(path, git_status):
                    if self._restore_file_via_git(path):
                        actual = self._hash_file(full)
                        if actual == expected_hash:
                            logger.info("IntegrityGuardian: successfully auto-healed tampered file: %s", path)
                            continue
                        if self._manifest.get(path) == actual:
                            # The restore path proved the file matches HEAD
                            # and re-baselined it: the manifest was stale,
                            # nothing was tampered with. Alerting here is
                            # what produced "9 files tampered" after an
                            # ordinary pull.
                            continue
                tampered.append(path)

        # Prune legitimately-gone files from manifest silently
        if legitimately_gone:
            for p in legitimately_gone:
                self._manifest.pop(p, None)

        if tampered or missing:
            # Drop legitimately modified files tracked by git to prevent local edits from causing alerts
            try:
                git_active = set(git_status.keys()) if git_status else self._git_active_paths()
                if git_active:
                    tampered = [p for p in tampered if self._normalize_repo_path(p) not in git_active]
                    missing = [p for p in missing if self._normalize_repo_path(p) not in git_active]
            except _INTEGRITY_GUARDIAN_ERRORS as exc:
                record_degradation('integrity_guardian', exc)
                logger.debug("IntegrityGuardian: git check failed: %s", exc)

        self._last_tampered = list(tampered)
        self._last_missing = list(missing)
        self._last_issue_count = len(tampered) + len(missing)
        self._verification_pending = pending
        self._pending_count = max(0, total - checked) if pending else 0
        self._last_ok = self._last_issue_count == 0 and not pending

        if tampered or missing:
            self._alert_count += len(tampered) + len(missing)
            self._handle_alerts(tampered, missing)

        return tampered + missing

    def _handle_alerts(self, tampered: list[str], missing: list[str]):
        """Process integrity violations."""
        all_bad = tampered + missing

        # Log to file
        entry = {
            "timestamp": time.time(),
            "tampered": tampered,
            "missing": missing,
        }
        try:
            with local_internal_governed_scope("security.integrity_guardian.alert", domain="file_write"):
                get_file_write_gateway().append_text(
                    ALERT_LOG_PATH,
                    json.dumps(entry) + "\n",
                    source="security.integrity_guardian.alert",
                )
        except (json.JSONDecodeError, TypeError, ValueError) as _exc:
            record_degradation('integrity_guardian', _exc)
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

    def _notify_emergency(self, affected_files: list[str], severity: str):
        try:
            from core.security.emergency_protocol import get_emergency_protocol
            ep = get_emergency_protocol()
            ep.flag_threat(
                f"integrity_{severity}",
                f"File integrity violation: {affected_files[:3]} "
                f"({'critical' if severity == 'critical' else 'non-critical'} files)"
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('integrity_guardian', e)
            logger.debug("Emergency notification failed: %s", e)

    # ── Hashing & Signing ──────────────────────────────────────────────────

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _sign_manifest(self, manifest: dict[str, str]) -> str:
        payload = json.dumps(manifest, sort_keys=True).encode()
        return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()

    def _load_manifest(self) -> bool:
        try:
            data = json.loads(MANIFEST_PATH.read_text())
            files = data.get("files", {})
            stored_sig = data.get("signature", "")
            stored_revision = str(data.get("source_revision", "") or "")
            current_revision = self._current_source_revision()

            if current_revision and stored_revision != current_revision:
                logger.info(
                    "IntegrityGuardian: source revision changed (%s → %s); deferring baseline refresh.",
                    stored_revision[:12] or "legacy",
                    current_revision[:12],
                )
                self._manifest_revision_stale = True

            out_of_scope = self._manifest_scope_mismatch(files)
            if out_of_scope:
                logger.info(
                    "IntegrityGuardian: manifest contains %d out-of-scope generated/test paths; rebuilding baseline.",
                    len(out_of_scope),
                )
                return False

            # Verify HMAC
            expected_sig = self._sign_manifest(files)
            if not hmac.compare_digest(stored_sig, expected_sig):
                logger.error("IntegrityGuardian: MANIFEST SIGNATURE INVALID — possible tampering!")
                self._notify_emergency(["integrity_manifest.json"], severity="critical")
                return False

            self._manifest = files
            self._manifest_hmac = stored_sig
            return True
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('integrity_guardian', e)
            logger.debug("Manifest load failed: %s", e)
            return False

    def _save_manifest(self):
        try:
            sig = self._sign_manifest(self._manifest)
            self._manifest_hmac = sig
            data = {
                "files": self._manifest,
                "signature": sig,
                "built_at": time.time(),
                "source_revision": self._current_source_revision(),
            }
            with local_internal_governed_scope(
                "security.integrity_guardian.manifest",
                domain="file_write",
                receipt_prefix="integrity-manifest-write",
            ):
                get_file_write_gateway().write_text(
                    MANIFEST_PATH,
                    json.dumps(data, indent=2),
                    source="security.integrity_guardian.manifest",
                )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            record_degradation('integrity_guardian', e)
            logger.debug("Manifest save failed: %s", e)

    @staticmethod
    def _current_source_revision() -> str:
        try:
            git_dir = _BASE_DIR / ".git"
            if git_dir.is_file():
                marker = git_dir.read_text(encoding="utf-8", errors="replace").strip()
                if marker.startswith("gitdir:"):
                    git_dir = (_BASE_DIR / marker.split(":", 1)[1].strip()).resolve()

            # Linked worktrees keep HEAD and the index in their private gitdir,
            # while branch refs and packed-refs live in the repository's common
            # directory. Looking only beside HEAD makes every symbolic worktree
            # revision appear blank, which makes a stale signed manifest look
            # current and triggers one git-show "restore" per source file.
            common_dir = git_dir
            commondir_path = git_dir / "commondir"
            if commondir_path.is_file():
                relative_common = commondir_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
                if relative_common:
                    common_dir = (git_dir / relative_common).resolve()

            head_path = git_dir / "HEAD"
            head = head_path.read_text(encoding="utf-8", errors="replace").strip()
            if not head.startswith("ref:"):
                return head

            ref_name = head.split(":", 1)[1].strip()
            for ref_root in (git_dir, common_dir):
                ref_path = ref_root / ref_name
                if ref_path.is_file():
                    return ref_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()

            packed_refs = common_dir / "packed-refs"
            if packed_refs.is_file():
                for line in packed_refs.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    revision, _, packed_ref = line.partition(" ")
                    if packed_ref.strip() == ref_name:
                        return revision.strip()
        except (OSError, ValueError) as exc:
            record_degradation('integrity_guardian', exc)
            logger.debug("IntegrityGuardian: source revision lookup failed: %s", exc)
        return ""

    @staticmethod
    def _normalize_repo_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        return Path(raw.lstrip("./")).as_posix()

    @classmethod
    def _parse_git_status_paths(cls, line: str) -> set[str]:
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

    def _git_active_paths(self) -> set[str]:
        with local_internal_governed_scope("security.integrity_guardian.git_status", domain="tool_execution"):
            status = get_subprocess_gateway().run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no"],
                cwd=str(_BASE_DIR),
                capture_output=True,
                timeout=8.0,
                read_only=True,
                source="security.integrity_guardian.git_status",
                accelerator_capability="none",
            )
        if status.returncode not in (0, 1):
            raise RuntimeError(f"git status returned {status.returncode}")

        active: set[str] = set()
        for line in status.stdout.splitlines():
            active.update(self._parse_git_status_paths(line))
        return active


# ── Singleton ──────────────────────────────────────────────────────────────────

_guardian: IntegrityGuardian | None = None


def get_integrity_guardian() -> IntegrityGuardian:
    global _guardian
    if _guardian is None:
        _guardian = IntegrityGuardian()
    return _guardian
