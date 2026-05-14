from core.runtime.errors import record_degradation
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict
from core.config import config
from core.runtime.service_access import (
    optional_service,
    resolve_canonical_self,
    resolve_identity_model,
    resolve_state_repository,
)

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


_USER_FACING_STATUS_ORIGINS = {
    "user",
    "api",
    "chat",
    "desktop",
    "gui",
    "voice",
    "web",
    "websocket",
    "ws",
    "direct",
    "external",
}


def _is_user_facing_status_origin(origin: Any) -> bool:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    if not normalized:
        return False
    tokens = {token for token in normalized.split("_") if token}
    return normalized in _USER_FACING_STATUS_ORIGINS or bool(tokens & _USER_FACING_STATUS_ORIGINS)


def _looks_like_stale_user_prompt(text: str) -> bool:
    lowered = text.lower()
    if not text:
        return False
    if "?" in text and any(marker in lowered for marker in ("aura", "you", "your", "what", "why", "how", "can ", "could ", "please")):
        return True
    return len(text) > 120 and any(
        marker in lowered
        for marker in (
            "what is actually on your mind",
            "tell me",
            "answer like",
            "why does",
            "can you",
            "could you",
        )
    )


def _clean_current_intention_for_status(intention: Any, live_objective: Any = "", live_origin: Any = "") -> str:
    text = " ".join(str(intention or "").split())
    objective = " ".join(str(live_objective or "").split())
    origin = str(live_origin or "").strip()
    if objective and origin.lower() not in {"system", "unknown"} and not _is_user_facing_status_origin(origin):
        return objective[:260]
    lowered = text.lower()
    if not text:
        return objective[:260] if objective and not _is_user_facing_status_origin(live_origin) else ""
    if "[referential anchor]" in lowered or len(text) > 320 or _looks_like_stale_user_prompt(text):
        return objective[:260] if objective and not _is_user_facing_status_origin(live_origin) else "idle"
    return text[:260]

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
    self_model_data = {"version": 0, "identity_index": 0.0, "name": "Aura", "current_intention": ""}

    try:
        import asyncio

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

        drive = optional_service("drive_engine", default=None)
        # Try sync properties if get_status is async/missing
        drive_data = _safe_status(drive, drive_data)
        if drive_data == {"energy": 0.0} and drive and hasattr(drive, "energy"):
             drive_data = {"energy": float(drive.energy)}

        affect = optional_service("affect_engine", default=None)
        affect_data = _safe_status(affect, affect_data)
        if affect_data == {"valence": 0.0, "arousal": 0.0, "engagement": 0.0} and affect:
            affect_data = {
                "valence": float(getattr(affect, "valence", 0)), 
                "arousal": float(getattr(affect, "arousal", 0))
            }

        repo = resolve_state_repository(default=None)
        live_state = getattr(repo, "_current", None) if repo is not None else None
        live_cognition = getattr(live_state, "cognition", None) if live_state is not None else None
        live_objective = str(
            getattr(live_cognition, "current_objective", "") or ""
        )
        live_origin = str(getattr(live_cognition, "current_origin", "") or "")

        canonical_self = resolve_canonical_self(default=None)
        if canonical_self is not None:
            self_model_data.update(
                {
                    "version": int(getattr(canonical_self, "version", 0) or 0),
                    "name": str(getattr(getattr(canonical_self, "identity", None), "name", "") or "Aura"),
                    "current_intention": _clean_current_intention_for_status(
                        getattr(canonical_self, "current_intention", "") or "",
                        live_objective,
                        live_origin,
                    ),
                }
            )

        sm = resolve_identity_model(default=None)
        reported_self = _safe_status(sm, {}) if sm is not None else {}
        if isinstance(reported_self, dict):
            merged_self = dict(reported_self)
            merged_self.setdefault("name", str(getattr(sm, "name", "Aura") or "Aura"))
            if canonical_self is not None:
                merged_self["version"] = int(getattr(canonical_self, "version", merged_self.get("version", 0)) or 0)
                merged_self["name"] = str(
                    getattr(getattr(canonical_self, "identity", None), "name", merged_self.get("name", "Aura"))
                    or merged_self.get("name", "Aura")
                )
                merged_self["current_intention"] = _clean_current_intention_for_status(
                    getattr(canonical_self, "current_intention", "") or "",
                    live_objective,
                    live_origin,
                )
            self_model_data = {**self_model_data, **merged_self}

    except Exception as e:
        record_degradation('runtime_tools', e)
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
