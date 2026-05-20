"""
core/self_modification/shadow_ast_healer.py
============================================
SHADOW AST HEALER — Governed Self-Repair via AST Manipulation

Implements zero-token self-repair via AST manipulation. When the system
detects a subsystem failure due to code errors, the ShadowHealer analyzes
the AST, identifies common patterns (missing imports, type mismatches),
and applies patches — but ONLY after governance approval.

GOVERNANCE CONTRACT:
    - Every file write MUST pass through _check_governance()
    - Only files within the codebase root can be modified
    - Only imports from a known-safe allowlist can be injected
    - The safe list explicitly EXCLUDES dangerous modules (os, subprocess,
      shutil, socket, ctypes, etc.)
    - All repair attempts are logged with before/after content hashes

This module is named "shadow" because it operates in the background
during error recovery, not because it hides from governance.
"""

import ast
import asyncio
import hashlib
import inspect
import logging
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path

logger = logging.getLogger("Aura.ShadowHealer")


def _record_shadow_healer_degradation(
    error: BaseException,
    *,
    action: str,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "shadow_ast_healer",
        error,
        severity="warning",
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


class ShadowASTHealer:
    """Autonomously repairs source code via AST manipulation.

    All file modifications are gated through governance — the healer
    cannot write to disk without approval from the ConstitutionalCore
    or UnifiedWill. If governance is unavailable, repairs are REFUSED
    (fail-closed, not fail-open).
    """

    # Known-safe imports that can be auto-injected. This list explicitly
    # EXCLUDES modules that provide filesystem, network, or process access.
    # Adding to this list requires a deliberate decision.
    _SAFE_IMPORTS = {
        "asyncio": "import asyncio",
        "json": "import json",
        "logging": "import logging",
        "time": "import time",
        "math": "import math",
        "re": "import re",
        "Path": "from pathlib import Path",
        "Any": "from typing import Any",
        "Dict": "from typing import Dict",
        "List": "from typing import List",
        "Optional": "from typing import Optional",
    }

    def __init__(self, codebase_root: Path | None = None):
        self.root = (codebase_root or Path.cwd()).resolve()

    def _check_governance(self, file_path: Path, action: str) -> bool:
        """Check with governance before modifying a file.

        Returns True if the repair is authorized, False otherwise.
        If the governance system is unavailable, returns False (fail-closed).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._check_governance_async(file_path, action))
        return self._check_governance_async(file_path, action)

    async def _check_governance_async(self, file_path: Path, action: str) -> bool:
        """Async governance check used by the repair coroutine."""
        try:
            from core.governance.will_client import WillClient, WillRequest
            from core.will import ActionDomain

            decision = await WillClient().decide_async(
                WillRequest(
                    content=action,
                    source="shadow_ast_healer",
                    domain=ActionDomain.STATE_MUTATION,
                    context={"file": str(file_path)},
                    priority=0.6,
                )
            )
            approved = WillClient.is_approved(decision)
            if not approved:
                logger.warning(
                    "ShadowHealer: Governance DENIED repair of %s: %s",
                    file_path.name,
                    getattr(decision, "reason", "no reason"),
                )
            return approved
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_shadow_healer_degradation(
                exc,
                action="Denied AST repair because governance decision was unavailable",
                extra={"file": str(file_path), "action": action},
            )
            # Governance check failed — fail closed
            logger.warning("ShadowHealer: Governance check failed (denying): %s", exc)
            return False

    def _is_within_root(self, file_path: Path) -> bool:
        """Verify the target file is within the codebase root."""
        try:
            resolved = file_path.resolve()
            resolved.relative_to(self.root)
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            return False

    async def propose_repair(self, file_path: Path, error_msg: str) -> dict[str, Any] | None:
        """Build an AST repair proposal without mutating source files."""
        logger.info("ShadowHealer: Analyzing %s: %s", file_path.name, error_msg)

        try:
            # Ensure path is absolute and within root
            if not file_path.is_absolute():
                file_path = self.root / file_path

            # Path containment check — refuse files outside root
            if not self._is_within_root(file_path):
                logger.warning(
                    "ShadowHealer: REFUSED — %s is outside codebase root %s",
                    file_path,
                    self.root,
                )
                return None
            try:
                rel_path = file_path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                rel_path = file_path.as_posix()
            tier = classify_mutation_path(rel_path)
            if tier.tier in {MutationTier.SEALED, MutationTier.PROPOSE_ONLY}:
                logger.warning(
                    "ShadowHealer: REFUSED — %s is %s (%s)",
                    rel_path,
                    tier.tier.label,
                    tier.reason,
                )
                return None

            content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
            content_hash_before = hashlib.sha256(content.encode()).hexdigest()[:16]
            tree = ast.parse(content)

            repaired = False
            # Pattern 1: Missing Import
            if "name" in error_msg.lower() and "is not defined" in error_msg.lower():
                try:
                    missing_name = error_msg.split("'")[1]
                except IndexError:
                    missing_name = error_msg.split()[-1]
                repaired = self._inject_missing_import(tree, missing_name)

            if repaired:
                new_content = ast.unparse(tree)
                content_hash_after = hashlib.sha256(new_content.encode()).hexdigest()[:16]
                return {
                    "target_file": rel_path,
                    "original_content": content,
                    "repaired_content": new_content,
                    "content_hash_before": content_hash_before,
                    "content_hash_after": content_hash_after,
                    "explanation": f"Zero-token AST repair for {error_msg}",
                }

            return None
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as e:
            _record_shadow_healer_degradation(
                e,
                action="Rejected AST repair proposal and left source file unchanged",
                extra={"file": str(file_path), "error": error_msg[:250]},
            )
            logger.error("ShadowHealer: AST repair failed: %s", e)
            return None

    async def attempt_repair(self, file_path: Path, error_msg: str) -> bool:
        """Attempts to repair a specific error in a target file.

        GOVERNANCE: Will not modify any file without explicit governance
        approval. Will not modify files outside the codebase root.
        """
        proposal = await self.propose_repair(file_path, error_msg)
        if not proposal:
            return False

        target_path = self.root / proposal["target_file"]
        governance_result = self._check_governance(
            target_path,
            f"ast_repair:{target_path.name}",
        )
        if inspect.isawaitable(governance_result):
            governance_result = await governance_result
        if not governance_result:
            logger.info("ShadowHealer: Repair blocked by governance for %s", target_path.name)
            return False

        await asyncio.to_thread(
            atomic_write_text,
            target_path,
            proposal["repaired_content"],
            encoding="utf-8",
        )
        logger.info(
            "ShadowHealer: Repaired %s (before=%s after=%s)",
            target_path.name,
            proposal["content_hash_before"],
            proposal["content_hash_after"],
        )
        return True

    def _inject_missing_import(self, tree: ast.AST, name: str) -> bool:
        """Injects a common missing import into the AST.

        Only imports from the known-safe allowlist are permitted. Dangerous
        modules (os, subprocess, shutil, socket, ctypes, etc.) are explicitly
        excluded and cannot be auto-injected.
        """
        if name in self._SAFE_IMPORTS:
            import_node = ast.parse(self._SAFE_IMPORTS[name]).body[0]
            tree.body.insert(0, import_node)
            return True
        logger.debug("ShadowHealer: '%s' not in safe import allowlist — skipping", name)
        return False

    def validate_syntax(self, file_path: Path) -> bool:
        """Validates that a file has valid Python syntax."""
        try:
            ast.parse(file_path.read_text(encoding="utf-8"))
            return True
        except SyntaxError:
            return False
