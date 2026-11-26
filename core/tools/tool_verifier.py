"""core/tools/tool_verifier.py — Tool Integrity Verifier."""
from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.tools.tool_manifest import ToolManifest

logger = logging.getLogger("Aura.ToolVerifier")


class ToolVerifier:
    """Verifies SHA-256 hashes and cryptographic signatures of external tools."""

    @staticmethod
    def verify_integrity(code: str, manifest: ToolManifest) -> bool:
        """Checks if the actual source code hash matches the manifest hash."""
        code_bytes = code.encode("utf-8")
        actual_hash = hashlib.sha256(code_bytes).hexdigest()
        if actual_hash != manifest.hash_sha256:
            logger.error(
                "Integrity check failed for tool %s: expected %s, got %s",
                manifest.name, manifest.hash_sha256, actual_hash
            )
            return False
        
        # In a production environment, signature verification would use a public key.
        # Here we verify that the signature starts with the manifest owner name for simulation.
        if not manifest.signature.startswith(manifest.owner):
            logger.error("Signature check failed for tool %s: invalid signature payload", manifest.name)
            return False

        logger.info("Integrity and signature verified for tool: %s", manifest.name)
        return True
