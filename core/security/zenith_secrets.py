"""
Secrets management for Aura.
Hierarchy (highest priority first):
  1. Environment variable
  2. macOS Keychain (production)
  3. .env file (development)
"""
import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Secrets")

_KEYCHAIN_SERVICE = "AuraAutonomyEngine"
_NO_ERR = 0
_ERR_SEC_ITEM_NOT_FOUND = -25300
_ERR_SEC_DUPLICATE_ITEM = -25299
_KEYCHAIN_UNAVAILABLE = object()
_KEYCHAIN_BACKEND: "KeychainBackend | object | None" = None
_DEFAULT_KEYCHAIN_CALL_TIMEOUT_S = 0.75
_KEYCHAIN_RECOVERABLE_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


class KeychainBackend(Protocol):
    def get_password(self, service: str, account: str) -> str | None: ...

    def set_password(self, service: str, account: str, password: str) -> bool: ...


class KeychainUnavailableError(RuntimeError):
    """Raised when the native macOS Keychain API is unavailable or rejects a call."""


class _SecurityFrameworkKeychain:
    """Tiny ctypes adapter over macOS Security.framework generic passwords."""

    def __init__(self) -> None:
        security_path = ctypes.util.find_library("Security")
        if not security_path:
            raise KeychainUnavailableError("Security.framework not found")
        self._security = ctypes.CDLL(security_path)

        core_foundation_path = ctypes.util.find_library("CoreFoundation")
        self._core_foundation = ctypes.CDLL(core_foundation_path) if core_foundation_path else None
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._security.SecKeychainFindGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecKeychainFindGenericPassword.restype = ctypes.c_int32

        self._security.SecKeychainAddGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecKeychainAddGenericPassword.restype = ctypes.c_int32

        self._security.SecKeychainItemModifyAttributesAndData.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        self._security.SecKeychainItemModifyAttributesAndData.restype = ctypes.c_int32

        self._security.SecKeychainItemFreeContent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._security.SecKeychainItemFreeContent.restype = ctypes.c_int32

        if self._core_foundation is not None:
            self._core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
            self._core_foundation.CFRelease.restype = None

    def get_password(self, service: str, account: str) -> str | None:
        password_length = ctypes.c_uint32(0)
        password_data = ctypes.c_void_p()
        item_ref = ctypes.c_void_p()
        status = self._find_generic_password(
            service,
            account,
            ctypes.byref(password_length),
            ctypes.byref(password_data),
            ctypes.byref(item_ref),
        )
        if status == _ERR_SEC_ITEM_NOT_FOUND:
            return None
        if status != _NO_ERR:
            raise KeychainUnavailableError(f"Keychain read failed with status {status}")
        try:
            if not password_data.value:
                return ""
            raw = ctypes.string_at(password_data.value, password_length.value)
            return raw.decode("utf-8")
        finally:
            self._free_keychain_content(password_data)
            self._release_item(item_ref)

    def set_password(self, service: str, account: str, password: str) -> bool:
        password_bytes = password.encode("utf-8")
        service_bytes = service.encode("utf-8")
        account_bytes = account.encode("utf-8")
        item_ref = ctypes.c_void_p()
        password_buffer = ctypes.create_string_buffer(password_bytes)
        status = self._security.SecKeychainAddGenericPassword(
            None,
            len(service_bytes),
            service_bytes,
            len(account_bytes),
            account_bytes,
            len(password_bytes),
            ctypes.cast(password_buffer, ctypes.c_void_p),
            ctypes.byref(item_ref),
        )
        try:
            if status == _NO_ERR:
                return True
            if status != _ERR_SEC_DUPLICATE_ITEM:
                raise KeychainUnavailableError(f"Keychain write failed with status {status}")
            existing_ref = self._find_item_ref(service, account)
            try:
                update_status = self._security.SecKeychainItemModifyAttributesAndData(
                    existing_ref,
                    None,
                    len(password_bytes),
                    ctypes.cast(password_buffer, ctypes.c_void_p),
                )
                if update_status != _NO_ERR:
                    raise KeychainUnavailableError(f"Keychain update failed with status {update_status}")
                return True
            finally:
                self._release_item(existing_ref)
        finally:
            self._release_item(item_ref)

    def _find_item_ref(self, service: str, account: str) -> ctypes.c_void_p:
        password_length = ctypes.c_uint32(0)
        password_data = ctypes.c_void_p()
        item_ref = ctypes.c_void_p()
        status = self._find_generic_password(
            service,
            account,
            ctypes.byref(password_length),
            ctypes.byref(password_data),
            ctypes.byref(item_ref),
        )
        self._free_keychain_content(password_data)
        if status != _NO_ERR or not item_ref.value:
            raise KeychainUnavailableError(f"Existing Keychain item lookup failed with status {status}")
        return item_ref

    def _find_generic_password(
        self,
        service: str,
        account: str,
        password_length: ctypes.c_void_p,
        password_data: ctypes.c_void_p,
        item_ref: ctypes.c_void_p,
    ) -> int:
        service_bytes = service.encode("utf-8")
        account_bytes = account.encode("utf-8")
        return int(
            self._security.SecKeychainFindGenericPassword(
                None,
                len(service_bytes),
                service_bytes,
                len(account_bytes),
                account_bytes,
                password_length,
                password_data,
                item_ref,
            )
        )

    def _free_keychain_content(self, password_data: ctypes.c_void_p) -> None:
        if password_data.value:
            self._security.SecKeychainItemFreeContent(None, password_data)

    def _release_item(self, item_ref: ctypes.c_void_p) -> None:
        if item_ref.value and self._core_foundation is not None:
            self._core_foundation.CFRelease(item_ref)


def _configured_keychain_call_timeout_s() -> float:
    raw = os.environ.get(
        "AURA_KEYCHAIN_CALL_TIMEOUT_S",
        str(_DEFAULT_KEYCHAIN_CALL_TIMEOUT_S),
    )
    try:
        configured = float(raw)
    except (TypeError, ValueError):
        configured = _DEFAULT_KEYCHAIN_CALL_TIMEOUT_S
    return max(0.05, min(5.0, configured))


class _BoundedKeychainBackend:
    """Keep native Security.framework IPC off callers and bound its blast radius.

    Security.framework can wait indefinitely for a locked or unavailable
    security daemon. A daemon worker keeps that wait off Aura's event loop and
    cannot hold process shutdown open. Single-flight admission guarantees that
    one stuck OS call cannot accumulate more threads over a long-lived runtime.
    """

    def __init__(
        self,
        backend_factory: Callable[[], KeychainBackend] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self._backend_factory = backend_factory or _SecurityFrameworkKeychain
        self._timeout_s = (
            _configured_keychain_call_timeout_s()
            if timeout_s is None
            else max(0.01, float(timeout_s))
        )
        self._state_lock = threading.Lock()
        self._idle = threading.Condition(self._state_lock)
        self._backend: KeychainBackend | None = None
        self._active_thread: threading.Thread | None = None

    def get_password(self, service: str, account: str) -> str | None:
        result = self._invoke("read", service, account, None)
        return result if isinstance(result, str) else None

    def set_password(self, service: str, account: str, password: str) -> bool:
        return bool(self._invoke("write", service, account, password))

    def _invoke(
        self,
        operation: str,
        service: str,
        account: str,
        password: str | None,
    ) -> object:
        done = threading.Event()
        result_lock = threading.Lock()
        state: dict[str, object] = {
            "abandoned": False,
            "result": None,
            "error": None,
        }

        def _worker() -> None:
            result: object = None
            error: BaseException | None = None
            try:
                backend = self._backend
                if backend is None:
                    candidate = self._backend_factory()
                    with self._state_lock:
                        if self._backend is None:
                            self._backend = candidate
                        backend = self._backend
                if operation == "read":
                    result = backend.get_password(service, account)
                else:
                    result = backend.set_password(service, account, password or "")
            except BaseException as exc:  # noqa: BLE001 - final daemon-worker containment boundary
                error = exc

            with result_lock:
                if not bool(state["abandoned"]):
                    state["result"] = result
                    state["error"] = error
            with self._idle:
                if self._active_thread is threading.current_thread():
                    self._active_thread = None
                self._idle.notify_all()
            done.set()

        worker = threading.Thread(
            target=_worker,
            name=f"aura-keychain-{operation}",
            daemon=True,
        )
        admission_deadline = time.monotonic() + self._timeout_s
        with self._idle:
            while self._active_thread is not None:
                remaining = admission_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise KeychainUnavailableError(
                        "Keychain operation deferred because a previous native call "
                        f"is still in progress after waiting {self._timeout_s:.2f}s"
                    )
                self._idle.wait(timeout=remaining)
            if self._active_thread is not None:
                raise KeychainUnavailableError(
                    "Keychain operation deferred because native-call admission did not converge"
                )
            self._active_thread = worker
            worker.start()

        if not done.wait(self._timeout_s):
            with result_lock:
                state["abandoned"] = True
                state["result"] = None
            raise KeychainUnavailableError(
                f"Keychain {operation} timed out after {self._timeout_s:.2f}s"
            )

        with result_lock:
            error = state["error"]
            result = state["result"]
            state["result"] = None
        if isinstance(error, BaseException):
            if isinstance(error, KeychainUnavailableError):
                raise error
            raise KeychainUnavailableError(
                f"Keychain {operation} failed: {type(error).__name__}: {error}"
            ) from error
        return result


def _native_keychain_disabled_for_test() -> bool:
    if os.environ.get("AURA_ALLOW_NATIVE_KEYCHAIN_IN_TESTS") == "1":
        return False
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("AURA_TEST_MODE") == "1"
    )


def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Retrieve a secret by key. Never logs the value.
    """
    # 1. Environment variable (also catches .env loaded by pydantic-settings)
    value = os.environ.get(key)
    if value:
        return value

    # 2. macOS Keychain
    value = _keychain_get(key)
    if value:
        return value

    if default is None:
        logger.debug("Secret '%s' not found in any store.", key)
    return default


def set_secret(key: str, value: str, store: str = "keychain") -> bool:
    """
    Persist a secret. Default target is macOS Keychain.
    """
    if store == "keychain":
        success = _keychain_set(key, value)
        if success:
            logger.info("Secret '%s' stored in Keychain.", key)
            return True
        logger.warning("Keychain unavailable — storing '%s' in environment only.", key)

    os.environ[key] = value
    return store != "keychain"


def _keychain_get(key: str) -> str | None:
    """Retrieve from macOS Keychain using Security.framework."""
    backend = _keychain_backend()
    if backend is None:
        return None
    try:
        return backend.get_password(_KEYCHAIN_SERVICE, key)
    except _KEYCHAIN_RECOVERABLE_ERRORS as exc:
        record_degradation("zenith_secrets", exc)
        logger.debug("Keychain read unavailable for key '%s': %s", key, exc)
    return None


def _keychain_set(key: str, value: str) -> bool:
    """Store in macOS Keychain."""
    backend = _keychain_backend()
    if backend is None:
        return False
    try:
        return backend.set_password(_KEYCHAIN_SERVICE, key, value)
    except _KEYCHAIN_RECOVERABLE_ERRORS as exc:
        record_degradation("zenith_secrets", exc)
        logger.debug("Keychain write unavailable for key '%s': %s", key, exc)
        return False


def _keychain_backend() -> KeychainBackend | None:
    global _KEYCHAIN_BACKEND
    if _KEYCHAIN_BACKEND is _KEYCHAIN_UNAVAILABLE:
        return None
    if _KEYCHAIN_BACKEND is not None:
        return _KEYCHAIN_BACKEND
    # Unit and proof runs must never read or mutate the operator's real
    # Keychain. Explicit injected backends above remain available to tests.
    if _native_keychain_disabled_for_test():
        return None
    if sys.platform != "darwin":
        _KEYCHAIN_BACKEND = _KEYCHAIN_UNAVAILABLE
        return None
    try:
        backend = _BoundedKeychainBackend()
    except _KEYCHAIN_RECOVERABLE_ERRORS as exc:
        record_degradation("zenith_secrets", exc)
        logger.debug("Native Keychain backend unavailable: %s", exc)
        _KEYCHAIN_BACKEND = _KEYCHAIN_UNAVAILABLE
        return None
    _KEYCHAIN_BACKEND = backend
    return backend


def require_keychain_backend() -> KeychainBackend:
    """Return the native/injected Keychain backend or fail without fallback."""

    backend = _keychain_backend()
    if backend is None:
        raise KeychainUnavailableError("A strict macOS Keychain backend is required")
    return backend


def load_dotenv(path: str | None = None) -> None:
    """Load a .env file into environment variables. Dev convenience only."""
    dot_env = Path(path or ".env")
    if not dot_env.exists():
        return
    logger.info("Loading .env from %s (dev mode)", dot_env)
    with open(dot_env, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ── Service-Namespaced Credential Management ─────────────────────────
# These helpers provide a clean interface for skills that need login
# credentials (email, Reddit, etc.). Values are NEVER logged.

_CREDENTIAL_KEYS = {
    "email": {
        "address": "AURA_EMAIL_ADDRESS",
        "password": "AURA_EMAIL_PASSWORD",
    },
    "reddit": {
        "username": "AURA_REDDIT_USERNAME",
        "password": "AURA_REDDIT_PASSWORD",
        # Her registered application, for reading (core/skills/reddit_reading.py).
        "client_id": "AURA_REDDIT_CLIENT_ID",
        "client_secret": "AURA_REDDIT_CLIENT_SECRET",
    },
    "owner": {
        "email": "AURA_OWNER_EMAIL",
    },
}


def get_credential(service: str, field: str = "password") -> str | None:
    """Retrieve a credential for a named service.

    Examples:
        get_credential("email", "address")   -> "auraluna.cog@gmail.com"
        get_credential("email", "password")  -> the Gmail password
        get_credential("reddit", "username") -> "AuraLuna_Cog"
        get_credential("owner", "email")     -> "youngbryan97@gmail.com"

    Returns None if the credential is not found. NEVER logs the value.
    """
    keys = _CREDENTIAL_KEYS.get(service, {})
    key = keys.get(field)
    if not key:
        logger.debug("No credential key mapping for service=%s field=%s", service, field)
        return None
    return get_secret(key)


def store_credential(service: str, field: str, value: str) -> None:
    """Store a credential for a named service in macOS Keychain.

    This is the ONLY approved way to persist credentials.
    The value is NEVER logged.
    """
    keys = _CREDENTIAL_KEYS.get(service)
    if keys is None:
        _CREDENTIAL_KEYS[service] = {}
        keys = _CREDENTIAL_KEYS[service]
    key_name = f"AURA_{service.upper()}_{field.upper()}"
    keys[field] = key_name
    set_secret(key_name, value, store="keychain")
