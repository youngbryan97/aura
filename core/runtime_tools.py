import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict
from core.config import config

logger = logging.getLogger("Aura.Runtime")

# Prefer Ed25519 signature if available, else HMAC-SHA256 fallback
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

# Key path (create securely; gitignore it)
KEY_DIR = str(config.paths.home_dir / "keys")
ED25519_PRIV_PATH = os.path.join(KEY_DIR, "ed25519_priv.pem")
HMAC_KEY_PATH = os.path.join(KEY_DIR, "hmac_key.bin")
_EPHEMERAL_HMAC_KEY = os.urandom(32)

def _prepare_key_dir() -> bool:
    try:
        os.makedirs(KEY_DIR, mode=0o700, exist_ok=True)
        if os.name != 'nt':
            os.chmod(KEY_DIR, 0o700)
        return True
    except OSError as exc:
        logger.warning("Runtime key storage unavailable at %s: %s. Using ephemeral signatures.", KEY_DIR, exc)
        return False

_KEY_STORAGE_AVAILABLE = _prepare_key_dir()

def _ensure_keys():
    if not _KEY_STORAGE_AVAILABLE:
        return
    if CRYPTO_AVAILABLE:
        if not os.path.exists(ED25519_PRIV_PATH):
            # generate and write private key
            priv = Ed25519PrivateKey.generate()
            fd = os.open(ED25519_PRIV_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(priv.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
    else:
        if not os.path.exists(HMAC_KEY_PATH):
            # generate random 32 byte key
            fd = os.open(HMAC_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(os.urandom(32))

try:
    _ensure_keys()
except OSError as exc:
    logger.warning("Failed to materialize runtime signing keys: %s. Using ephemeral signatures.", exc)

def _compute_sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _sign_payload(payload_bytes: bytes) -> str:
    if CRYPTO_AVAILABLE and _KEY_STORAGE_AVAILABLE and os.path.exists(ED25519_PRIV_PATH):
        with open(ED25519_PRIV_PATH, "rb") as f:
            priv = serialization.load_pem_private_key(f.read(), password=None)
        sig = priv.sign(payload_bytes)
        return sig.hex()
    elif _KEY_STORAGE_AVAILABLE and os.path.exists(HMAC_KEY_PATH):
        with open(HMAC_KEY_PATH, "rb") as f:
            key = f.read()
    else:
        key = _EPHEMERAL_HMAC_KEY
    sig = hmac.new(key, payload_bytes, hashlib.sha256).hexdigest()
    return sig

def get_runtime_state() -> Dict[str, Any]:
    """Trusted function that samples live runtime pieces and returns:
    { "state": {...}, "sha256": "...", "signature": "..." }
    """
    # Pull live state from the service container (graceful fallback if not ready)
    drive_data = {"energy": 0.0}
    affect_data = {"valence": 0.0, "arousal": 0.0, "engagement": 0.0}
    self_model_data = {"version": "8.2", "identity_index": 0.0}

    try:
        import asyncio

        from core.container import ServiceContainer
        
        # Helper to safely get status
        def _safe_status(service, fallback_data):
            if not service: return fallback_data
            try:
                if hasattr(service, "get_status"):
                    res = service.get_status()
                    if asyncio.iscoroutine(res):
                        res.close() # Clean up unawaited coroutine
                        return fallback_data
                    return res
                return fallback_data
            except Exception:
                return fallback_data

        drive = ServiceContainer.get("drive_engine", default=None)
        # Try sync properties if get_status is async/missing
        drive_data = _safe_status(drive, drive_data)
        if drive_data == {"energy": 0.0} and drive and hasattr(drive, "energy"):
             drive_data = {"energy": float(drive.energy)}

        affect = ServiceContainer.get("affect_engine", default=None)
        affect_data = _safe_status(affect, affect_data)
        if affect_data == {"valence": 0.0, "arousal": 0.0, "engagement": 0.0} and affect:
            affect_data = {
                "valence": float(getattr(affect, "valence", 0)), 
                "arousal": float(getattr(affect, "arousal", 0))
            }

        sm = ServiceContainer.get("self_model", default=None)
        self_model_data = _safe_status(sm, self_model_data)
        
    except Exception as e:
        logger.debug("Runtime state sampling skipped (pre-init): %s", e)

    state = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "heartbeat_tick": int(time.time()),
        "self_model": self_model_data,
        "drive_engine": drive_data,
        "affect": affect_data,
        "process_id": os.getpid(),
    }
    
    from core.resilience.state_manager import _SafeEncoder
    state_json = json.dumps(state, sort_keys=True, ensure_ascii=False, cls=_SafeEncoder)
    sha256 = _compute_sha256_hex(state_json)
    signature = _sign_payload(state_json.encode("utf-8"))
    
    return {"state": state, "sha256": sha256, "signature": signature}
