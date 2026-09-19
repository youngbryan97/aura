"""How a container key is sealed, and what the seal is over.

Lifted whole out of `container`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import os
from pathlib import Path


class _SealsItsKeys:
    """Lifted whole out of ServiceContainer; see container.py."""

    @classmethod
    def _seal_path(cls) -> Path:
        from .container import (
            logger,
            record_degradation,
            state_root,
        )

        try:
            from core.config import config

            return Path(config.paths.data_dir) / "sovereignty_seal.json"
        except (ImportError, AttributeError, RuntimeError, OSError) as exc:
            record_degradation("container", exc)
            logger.debug("Falling back to default sovereignty seal path after config lookup failed: %s", exc)
            return Path(state_root()) / "data" / "sovereignty_seal.json"

    @classmethod
    def _seal_key(cls) -> bytes | None:
        """Local HMAC key for the sovereignty seal, created on first use.

        None when unavailable, which the verifier treats as unsigned rather
        than valid.
        """
        from .container import (
            get_file_write_gateway,
            local_internal_governed_scope,
        )

        path = cls._seal_path().with_name(".sovereignty_seal.key")
        try:
            if path.exists():
                key: bytes | None = path.read_bytes()
                return key if key is not None and len(key) == 32 else None
            candidate = os.urandom(32)
            with local_internal_governed_scope(
                "service_container.sovereignty_seal_key",
                domain="file_write",
            ):
                # Annotated above because the gateway is not followed by the
                # ratchet's mypy, so its return reads as Any here.
                key = get_file_write_gateway().provision_private_bytes(
                    path,
                    candidate,
                    expected_size=32,
                    mode=0o600,
                    source="service_container.sovereignty_seal_key",
                )
            return key
        except (OSError, ValueError):
            return None

    @classmethod
    def _seal_signature(cls, digest: str, service_count: int) -> str:
        from .container import (
            hashlib,
            hmac,
        )

        key = cls._seal_key()
        if key is None:
            return ""
        body = f"{digest}:{service_count}".encode()
        return hmac.new(key, body, hashlib.sha256).hexdigest()

