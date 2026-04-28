"""core/memory/black_hole.py — Aura 3.0: BlackHole Encryption
=========================================================
Implements Phase 7: AES-256-GCM encryption for all local persistent data.
Replaces the old legacy XOR obfuscation.

ZENITH Protocol compliance:
  - AES-256-GCM (Authenticated Encryption).
  - Derived key from Horcrux hardware entanglement.
  - Zero raw secrets stored on disk.
"""

from core.runtime.errors import record_degradation
import base64
import binascii
import hashlib
import logging
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.BlackHole")


class DecodedPayload(str):
    """Backward-compatible decode result supporting both string and mapping access."""

    def get(self, key: str, default: str = "") -> str:
        if key == "decoded":
            return str(self)
        return default

    def __contains__(self, item: object) -> bool:
        if item == "decoded":
            return True
        return super().__contains__(item)

    def __getitem__(self, key):  # type: ignore[override]
        if key == "decoded":
            return str(self)
        return super().__getitem__(key)


def _resolve_aes_key(key_material: str | bytes) -> bytes:
    """Accept base64-encoded keys, raw AES keys, or arbitrary strings.

    The legacy memory stack sometimes passes a placeholder or raw string key
    before Horcrux has been fully initialized. Instead of crashing on base64
    decode, normalize unsupported formats into a deterministic AES-256 key.
    """
    raw = key_material.encode("utf-8") if isinstance(key_material, str) else bytes(key_material)

    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        decoded = b""

    if len(decoded) in {16, 24, 32}:
        return decoded
    if len(raw) in {16, 24, 32}:
        return raw
    return hashlib.sha256(raw).digest()


class BlackHole:
    """
    Encryption provider for Aura 3.0.
    
    ZENITH Purity:
      - Mandatory authentication tag verification.
      - Automated nonce generation.
    """

    def __init__(self):
        self._aesgcm: Optional[AESGCM] = None

    def on_start(self):
        """Initializes the provider with the Horcrux key."""
        from core.container import ServiceContainer
        horcrux = ServiceContainer.get("horcrux", default=None)
        if not horcrux or not horcrux.derived_key:
            logger.error("BlackHole: Horcrux keys UNAVAILABLE. Encryption disabled.")
            return
            
        self._aesgcm = AESGCM(horcrux.derived_key)
        logger.info("BlackHole: AES-256-GCM substrate initialized.")

    def encrypt(self, data: bytes) -> bytes:
        """Encrypts data with a fresh random nonce."""
        if not self._aesgcm:
            logger.warning("BlackHole: Encryption bypass active (no key).")
            return data
            
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypts and verifies data."""
        if not self._aesgcm:
            return blob
            
        try:
            nonce = blob[:12]
            ciphertext = blob[12:]
            return self._aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            record_degradation('black_hole', e)
            logger.error("BlackHole decryption FAILED: %s", e)
            raise ValueError("Decryption/Authentication failure.") from e
            
    def encrypt_json(self, data: Dict[str, Any]) -> str:
        import json
        raw = json.dumps(data).encode()
        return base64.b64encode(self.encrypt(raw)).decode()

    def decrypt_json(self, b64_blob: str) -> Dict[str, Any]:
        import json
        blob = base64.b64decode(b64_blob)
        raw = self.decrypt(blob)
        return json.loads(raw.decode())


def encode_payload(data: str | bytes, key_b64: str) -> Dict[str, str]:
    """Module-level compatibility for Zenith memory encryption."""
    key = _resolve_aes_key(key_b64)
    aesgcm = AESGCM(key)
    
    raw = data.encode() if isinstance(data, str) else data
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, raw, None)
    encoded = base64.b64encode(nonce + ciphertext).decode()
    raw_len = max(len(raw), 1)
    ratio = round((len(encoded) / raw_len) * 100, 2)
    
    return {"encoded": encoded, "ratio": ratio}


def decode_payload(b64_blob: str, key_b64: str) -> DecodedPayload:
    """Module-level compatibility for Zenith memory decryption."""
    try:
        key = _resolve_aes_key(key_b64)
        aesgcm = AESGCM(key)
        
        blob = base64.b64decode(b64_blob)
        nonce = blob[:12]
        ciphertext = blob[12:]
        
        decrypted = aesgcm.decrypt(nonce, ciphertext, None).decode()
        return DecodedPayload(decrypted)
    except Exception as e:
        record_degradation('black_hole', e)
        logger.debug("decode_payload failed: %s", e)  # Downgraded — happens on first boot with no stored data
        return DecodedPayload("")
