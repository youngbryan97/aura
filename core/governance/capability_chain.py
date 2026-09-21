"""core/governance/capability_chain.py
=====================================
The cryptographic join between the Will and every consequential sink.

Aura had all the pieces of a constitutional authority system — the Will could
make and sign a legitimate decision, sinks could require a token, receipts
could record an operation — but the pieces were never *joined*. The sink could
not establish that the token it accepted came from the Will. A token was an
8-character uuid in a process-global dict (``core/agency/capability_system.py``),
mintable by anyone who imported the module, and callers could skip even that by
setting ``ctx["_capability_token_verified"] = True``. That broke the invariant
underlying the strongest claims Aura makes about itself.

This module closes the chain. A :class:`SignedCapability` is an immutable,
signed grant that binds, in one signature:

    outcome        the Will's actual decision (only approving outcomes issue)
    domain         the ActionDomain the decision was made in
    action_digest  a digest of the concrete action + its parameters
    issuer         who minted it
    key_id         which key minted it
    nonce          single-use, replay-defeating
    expires_at     wall-clock expiry
    receipt_id     back-link to the Will's provenance record

Verification is fail-closed and rejects, at the moment of execution:

    fabricated       — no valid signature over the payload
    altered          — any field mutated after issue (signature no longer binds)
    refused          — an outcome the Will never approves
    expired          — past ``expires_at`` (or not yet valid)
    replayed         — a nonce already consumed
    domain-mismatch  — issued for one ActionDomain, presented at another
    action-mismatch  — issued for one action digest, presented for another
    revoked          — explicitly revoked before use

Threat model — stated honestly
------------------------------
Under Ed25519 (the default when ``cryptography`` is importable) the asymmetry is
real and load-bearing: :class:`CapabilityVerifier` loads **only the public key**,
so a sink is *structurally incapable* of minting a capability it would accept.
Only :class:`WillCapabilityIssuer`, which holds the private key, can mint.

What this defends against is the actual failure that was present: structural
bypass, self-asserted verification flags, fabricated or LLM-authored governance
contexts, replay across turns, and domain/action confusion. What it does **not**
defend against is malicious in-process code that reads the private key off disk
and mints deliberately. Aura is a single trust domain; a capability chain cannot
manufacture a boundary the process does not have. The honest claim is: forgery
is no longer reachable by accident, refactor, or a fabricated dict — it now
requires deliberate key theft, which is auditable. Do not claim more than that.

On the HMAC fallback that asymmetry is lost (verifiers can mint), so the fallback
is a degraded mode: it is recorded in ``key_id`` as ``hmac-*``, surfaced in
:func:`capability_chain_status`, and callers that require the strong property can
assert :func:`issuer_is_asymmetric`.

Dependency discipline
---------------------
This module deliberately does not import ``core.governance.will`` (the Will
issues capabilities, so that would be circular) and does not route its nonce
ledger through ``core/runtime/file_write_gateway.py`` (the gateway is itself a
consequential sink and will be capability-gated; routing the ledger through it
would make the authority system depend on the thing it authorizes). Key and
nonce state use the lower-level canonical atomic writer with interprocess
serialization so authority durability does not depend on its own sink.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import stat
import threading
import time
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import (
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
    interprocess_file_lock,
)
from core.runtime.flags import FlagKind, declare
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Governance.CapabilityChain")

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _ED25519_AVAILABLE = True
except (ImportError, AttributeError):  # pragma: no cover - declared runtime dep
    _ED25519_AVAILABLE = False

    class InvalidSignature(Exception):  # type: ignore[no-redef]
        pass


CAPABILITY_SCHEMA_VERSION = 1

# Domain-separation tag. Signatures are computed over this tag + payload so a
# signature minted for some other Aura structure can never be replayed as a
# capability, and vice versa.
_SIGNING_TAG = b"aura.governance.capability.v1\x00"
_WILL_RECEIPT_SIGNING_TAG = b"aura.governance.will-receipt.v1\x00"

# Outcomes the Will considers authorizing. Anything else must never mint, and a
# forged capability carrying a non-approving outcome is rejected at verify time
# even if its signature were somehow valid.
APPROVING_OUTCOMES: frozenset[str] = frozenset({"proceed", "constrain", "critical"})

DEFAULT_TTL_S = 300.0
MAX_TTL_S = 3600.0
# Small tolerance for clock jitter between issue and verify on the same host.
_CLOCK_SKEW_TOLERANCE_S = 2.0

_NONCE_LEDGER_CAP = 20_000
_NONCE_LEDGER_SCHEMA = "aura.governance.capability_nonce_ledger"
_NONCE_LEDGER_VERSION = 1
_NONCE_LEDGER_MAX_BYTES = 4 * 1024 * 1024


class CapabilityDenial(StrEnum):
    """Why a capability was refused. Every value is a hard denial."""

    MISSING = "missing"
    MALFORMED = "malformed"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNKNOWN_ISSUER = "unknown_issuer"
    BAD_SIGNATURE = "bad_signature"
    NOT_APPROVED = "not_approved"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REPLAYED = "replayed"
    REVOKED = "revoked"
    DOMAIN_MISMATCH = "domain_mismatch"
    ACTION_MISMATCH = "action_mismatch"
    LEDGER_UNAVAILABLE = "ledger_unavailable"


class CapabilityViolation(Exception):
    """Raised at a consequential sink when authority cannot be established.

    This is the fail-closed path. It carries the machine-readable denial so
    sinks can record a truthful receipt rather than a generic error.
    """

    def __init__(self, denial: CapabilityDenial, detail: str = "", *, sink: str = ""):
        self.denial = denial
        self.detail = detail
        self.sink = sink
        super().__init__(
            f"Capability denied at sink '{sink or 'unknown'}': {denial.value}"
            + (f" ({detail})" if detail else "")
        )


# ---------------------------------------------------------------------------
# Action digests
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    """Deterministic JSON. Both issuer and sink must produce identical bytes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_fallback,
    )


def _json_fallback(obj: Any) -> str:
    """Stable rendering for values json cannot encode natively.

    Falls back to the repr of the type rather than the value for unknown
    objects: a digest must never depend on a memory address.
    """
    if isinstance(obj, (set, frozenset)):
        return _canonical_json(sorted(str(o) for o in obj))
    if isinstance(obj, bytes):
        return hashlib.sha256(obj).hexdigest()
    if isinstance(obj, Path):
        return str(obj)
    return f"<{type(obj).__name__}>"


def compute_action_digest(action: str, payload: Any = None) -> str:
    """Digest binding a capability to one concrete action and its parameters.

    The digest is what makes a capability non-transferable between actions: a
    grant minted for ``read_file(/etc/hosts)`` cannot be presented for
    ``shell_command(rm -rf /)`` even though both are tool executions in the same
    domain.
    """
    body = _canonical_json({"action": str(action or ""), "payload": payload})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The capability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignedCapability:
    """An immutable, signed grant of authority from the Will to one action.

    Frozen because a capability that can be mutated after issue is not a
    capability. Any mutation must go through :meth:`replace`-style
    reconstruction, which invalidates the signature — which is the point.
    """

    capability_id: str
    schema_version: int
    outcome: str
    domain: str
    action_digest: str
    issuer: str
    key_id: str
    nonce: str
    receipt_id: str
    scope: str
    constraints: tuple[str, ...]
    issued_at: float
    expires_at: float
    signature: str = ""

    # -- serialization ----------------------------------------------------
    def signing_payload(self) -> bytes:
        """Canonical bytes covered by the signature (everything but the sig)."""
        return _SIGNING_TAG + _canonical_json(
            {
                "capability_id": self.capability_id,
                "schema_version": self.schema_version,
                "outcome": self.outcome,
                "domain": self.domain,
                "action_digest": self.action_digest,
                "issuer": self.issuer,
                "key_id": self.key_id,
                "nonce": self.nonce,
                "receipt_id": self.receipt_id,
                "scope": self.scope,
                "constraints": list(self.constraints),
                "issued_at": round(float(self.issued_at), 6),
                "expires_at": round(float(self.expires_at), 6),
            }
        ).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "domain": self.domain,
            "action_digest": self.action_digest,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "nonce": self.nonce,
            "receipt_id": self.receipt_id,
            "scope": self.scope,
            "constraints": list(self.constraints),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Any) -> SignedCapability:
        """Rebuild from transport. Raises ValueError on anything malformed.

        Deliberately strict: a capability that arrives with a missing or
        wrong-typed field is a fabrication attempt, not a compatibility
        problem to paper over.
        """
        if not isinstance(data, dict):
            raise ValueError(f"capability must be a mapping, got {type(data).__name__}")
        try:
            return cls(
                capability_id=str(data["capability_id"]),
                schema_version=int(data["schema_version"]),
                outcome=str(data["outcome"]),
                domain=str(data["domain"]),
                action_digest=str(data["action_digest"]),
                issuer=str(data["issuer"]),
                key_id=str(data["key_id"]),
                nonce=str(data["nonce"]),
                receipt_id=str(data["receipt_id"]),
                scope=str(data["scope"]),
                constraints=tuple(str(c) for c in (data.get("constraints") or ())),
                issued_at=float(data["issued_at"]),
                expires_at=float(data["expires_at"]),
                signature=str(data.get("signature") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed capability: {exc}") from exc

    def redacted(self) -> dict[str, Any]:
        """Log-safe view. Never log the full nonce or signature."""
        return {
            "capability_id": self.capability_id,
            "outcome": self.outcome,
            "domain": self.domain,
            "action_digest": self.action_digest[:12],
            "issuer": self.issuer,
            "key_id": self.key_id,
            "receipt_id": self.receipt_id,
            "expires_in_s": round(self.expires_at - time.time(), 3),
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    denial: CapabilityDenial | None = None
    detail: str = ""
    capability: SignedCapability | None = None

    def raise_if_denied(self, sink: str = "") -> SignedCapability:
        if not self.ok or self.capability is None:
            raise CapabilityViolation(
                self.denial or CapabilityDenial.MALFORMED, self.detail, sink=sink
            )
        return self.capability


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


_KEY_DIR_FLAG = declare(
    "AURA_CAPABILITY_KEY_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for the Will's capability signing keys",
    owner="core.governance.capability_chain",
)
_FORCE_HMAC_FLAG = declare(
    "AURA_CAPABILITY_FORCE_HMAC", kind=FlagKind.BOOL, default=False,
    description=(
        "Force the degraded symmetric HMAC mode instead of Ed25519 "
        "(verifiers can then mint — test seam only)"
    ),
    owner="core.governance.capability_chain",
)
_ENFORCEMENT_FLAG = declare(
    "AURA_CAPABILITY_ENFORCEMENT", kind=FlagKind.STRING, default="",
    description="How hard sinks enforce the capability chain: strict | warn | off",
    owner="core.governance.capability_chain",
)


def _key_dir() -> Path:
    override = str(_KEY_DIR_FLAG.value() or "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        return Path(config.paths.home_dir) / "keys"
    except (ImportError, AttributeError, RuntimeError):
        return state_root() / "keys"


def _priv_path() -> Path:
    return _key_dir() / "will_capability_ed25519_priv.pem"


def _pub_path() -> Path:
    return _key_dir() / "will_capability_ed25519_pub.pem"


def _hmac_path() -> Path:
    return _key_dir() / "will_capability_hmac.key"


def _key_id_for(material: bytes, kind: str) -> str:
    return f"{kind}-{hashlib.sha256(material).hexdigest()[:16]}"


def _read_private_material(path: Path, *, max_bytes: int = 64 * 1024) -> bytes:
    """Read one owner-private regular file without following a symlink."""

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"private material is not a regular file: {path}")
    if before.st_mode & 0o077:
        raise PermissionError(f"private material is not owner-only: {path}")
    if hasattr(os, "getuid") and before.st_uid != os.getuid():
        raise PermissionError(f"private material is not owned by this user: {path}")
    if before.st_size > max_bytes:
        raise ValueError(f"private material exceeds {max_bytes} bytes: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"private material is not a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"private material changed while opening: {path}")
        if opened.st_mode & 0o077:
            raise PermissionError(f"private material is not owner-only: {path}")
        if hasattr(os, "getuid") and opened.st_uid != os.getuid():
            raise PermissionError(f"private material is not owned by this user: {path}")
        chunks: list[bytes] = []
        total = 0
        remaining = opened.st_size
        while remaining > 0:
            chunk = os.read(fd, min(64 * 1024, remaining))
            if not chunk:
                raise RuntimeError(f"private material truncated while reading: {path}")
            chunks.append(chunk)
            total += len(chunk)
            remaining -= len(chunk)
            if total > max_bytes:
                raise ValueError(f"private material exceeds {max_bytes} bytes: {path}")
        after = os.fstat(fd)
        stable_attributes = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            stat.S_IMODE(after.st_mode),
            after.st_uid,
            after.st_gid,
        )
        opened_attributes = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            stat.S_IMODE(opened.st_mode),
            opened.st_uid,
            opened.st_gid,
        )
        if stable_attributes != opened_attributes:
            raise RuntimeError(f"private material changed while reading: {path}")
        if after.st_ctime_ns != opened.st_ctime_ns and not (
            opened.st_nlink == 2 and after.st_nlink == 1
        ):
            raise RuntimeError(f"private material metadata changed while reading: {path}")
        current = path.lstat()
        current_attributes = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            stat.S_IMODE(current.st_mode),
            current.st_uid,
            current.st_gid,
        )
        if not stat.S_ISREG(current.st_mode) or current_attributes != stable_attributes:
            raise RuntimeError(f"private material path changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


class _KeyMaterial:
    """Loads/creates the Will's capability keys.

    The private key is created 0600 inside a 0700 directory. If key storage is
    unavailable we fall back to a process-ephemeral key rather than to no
    signing at all: an ephemeral key still defeats fabrication and replay
    within the process lifetime, and capabilities do not outlive the process
    because their TTL is minutes.
    """

    _lock = threading.RLock()
    _cache: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def _ensure_dir(cls) -> bool:
        try:
            ensure_private_directory(_key_dir())
            return True
        except OSError as exc:
            logger.warning(
                "Capability key storage unavailable at %s: %s — using an "
                "ephemeral in-process key for this run.",
                _key_dir(),
                exc,
            )
            return False

    @classmethod
    def load(cls) -> dict[str, Any]:
        with cls._lock:
            if cls._cache:
                return cls._cache
            cls._cache = cls._load_uncached()
            return cls._cache

    @classmethod
    def _load_uncached(cls) -> dict[str, Any]:
        storage = cls._ensure_dir()
        forced_hmac = bool(_FORCE_HMAC_FLAG.value())

        if _ED25519_AVAILABLE and not forced_hmac:
            try:
                priv, persisted = cls._load_or_create_ed25519(storage)
                pub = priv.public_key()
                pub_raw = pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
                if storage:
                    cls._write_public(pub)
                return {
                    "algorithm": "ed25519",
                    "private": priv,
                    "public": pub,
                    "key_id": _key_id_for(pub_raw, "ed25519"),
                    "asymmetric": True,
                    "persisted": persisted,
                }
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                logger.error(
                    "Ed25519 capability key unusable (%s) — using an ephemeral "
                    "asymmetric key for this process without replacing persisted material.",
                    exc,
                )
                priv = Ed25519PrivateKey.generate()
                pub = priv.public_key()
                pub_raw = pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
                return {
                    "algorithm": "ed25519",
                    "private": priv,
                    "public": pub,
                    "key_id": _key_id_for(pub_raw, "ed25519-ephemeral"),
                    "asymmetric": True,
                    "persisted": False,
                }

        secret, persisted = cls._load_or_create_hmac(storage)
        return {
            "algorithm": "hmac-sha256",
            "private": secret,
            "public": secret,
            "key_id": _key_id_for(secret, "hmac"),
            "asymmetric": False,
            "persisted": persisted,
        }

    @classmethod
    def _load_or_create_ed25519(cls, storage: bool) -> tuple[Any, bool]:
        path = _priv_path()
        if storage and os.path.lexists(path):
            loaded = serialization.load_pem_private_key(
                _read_private_material(path), password=None
            )
            if not isinstance(loaded, Ed25519PrivateKey):
                raise ValueError(f"{path} is not an Ed25519 private key")
            return loaded, True

        priv = Ed25519PrivateKey.generate()
        if not storage:
            return priv, False
        data = priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        if atomic_write_bytes_if_absent(path, data, mode=0o600):
            logger.info("Minted a new Will capability signing key at %s", path)
            return priv, True
        loaded = serialization.load_pem_private_key(
            _read_private_material(path), password=None
        )
        if not isinstance(loaded, Ed25519PrivateKey):
            raise ValueError(f"{path} is not an Ed25519 private key")
        return loaded, True

    @classmethod
    def _write_public(cls, pub: Any) -> None:
        """Publish the public key, once, and off the loop.

        Every load rewrote it, fsync and all, and the load happens during
        boot from a coroutine (LIVE 2026-09-21: `_async_init_subsystems`).
        A key that is already on disk unchanged is not written again, and
        one that is goes behind the loop.
        """
        try:
            data = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not serialise capability public key: %s", exc)
            return
        path = _pub_path()
        try:
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    if handle.read(len(data) + 1) == data:
                        return
        except OSError as exc:
            logger.debug("Could not compare the published capability key: %s", exc)

        def publish() -> None:
            try:
                atomic_write_bytes(path, data, mode=0o644)
            except OSError as exc:
                logger.warning("Could not publish capability public key: %s", exc)

        from core.runtime.executors import behind_the_loop

        behind_the_loop("capability_chain.public_key", publish)

    @classmethod
    def _load_or_create_hmac(cls, storage: bool) -> tuple[bytes, bool]:
        path = _hmac_path()
        if storage and os.path.lexists(path):
            try:
                secret = _read_private_material(path, max_bytes=4096)
                if 32 <= len(secret) <= 4096:
                    return secret, True
                logger.error(
                    "Capability HMAC key at %s is too short; refusing to replace it",
                    path,
                )
            except (OSError, RuntimeError) as exc:
                logger.error("Capability HMAC key unreadable (%s)", exc)
            return os.urandom(32), False

        secret = os.urandom(32)
        if not storage:
            return secret, False
        try:
            if atomic_write_bytes_if_absent(path, secret, mode=0o600):
                return secret, True
            winner = _read_private_material(path, max_bytes=4096)
            if not 32 <= len(winner) <= 4096:
                raise ValueError("persisted capability HMAC key has an invalid length")
            return winner, True
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Could not persist capability HMAC key: %s", exc)
            return secret, False


def issuer_is_asymmetric() -> bool:
    """True when sinks verify with a public key they cannot mint with.

    Call this in any test or gate that needs the *strong* property rather than
    merely "a signature was checked".
    """
    return bool(_KeyMaterial.load().get("asymmetric"))


def capability_chain_status() -> dict[str, Any]:
    """Operator-facing truth about the authority chain's current strength."""
    keys = _KeyMaterial.load()
    ledger = get_nonce_ledger()
    ledger_status = ledger.status()
    asymmetric = bool(keys["asymmetric"])
    persisted = bool(keys["persisted"])
    durable = asymmetric and persisted and ledger_status["healthy"]
    return {
        "algorithm": keys["algorithm"],
        "key_id": keys["key_id"],
        "asymmetric": asymmetric,
        "keys_persisted": persisted,
        "nonce_ledger_size": ledger_status["size"],
        "nonce_ledger_healthy": ledger_status["healthy"],
        "nonce_ledger_error": ledger_status["error"],
        "authority_durable": durable,
        "degraded": not durable,
        "note": (
            "Sinks verify with a persisted public key and accepted nonces are durable."
            if durable
            else (
                "DEGRADED: symmetric HMAC — any holder of the key can mint."
                if not asymmetric
                else (
                    "DEGRADED: signing identity is ephemeral and changes on restart."
                    if not persisted
                    else "DEGRADED: durable nonce replay protection is unavailable."
                )
            )
        ),
    }


_VALID_ENFORCEMENT_MODES = frozenset({"strict", "warn", "off"})


def capability_enforcement_mode(default: str = "strict") -> str:
    """How hard sinks enforce the chain. See ``_capability_chain_denial``.

    An unrecognized value resolves to ``strict`` rather than to the caller's
    default: a typo in an env var must not silently disable governance.
    """
    raw = str(_ENFORCEMENT_FLAG.value() or "").strip().lower()
    if not raw:
        return default if default in _VALID_ENFORCEMENT_MODES else "strict"
    if raw not in _VALID_ENFORCEMENT_MODES:
        logger.error(
            "AURA_CAPABILITY_ENFORCEMENT=%r is not one of %s — falling back to "
            "strict rather than disabling governance on a typo.",
            raw,
            sorted(_VALID_ENFORCEMENT_MODES),
        )
        return "strict"
    return raw


def _sign(payload: bytes) -> str:
    keys = _KeyMaterial.load()
    if keys["asymmetric"]:
        return keys["private"].sign(payload).hex()
    return hmac.new(keys["private"], payload, hashlib.sha256).hexdigest()


def _verify_signature(payload: bytes, signature: str, key_id: str) -> bool:
    keys = _KeyMaterial.load()
    if not signature:
        return False
    if key_id != keys["key_id"]:
        # A capability minted under a key we do not hold. Never accept it: an
        # unknown key_id is indistinguishable from an attacker naming their own.
        return False
    if keys["asymmetric"]:
        try:
            keys["public"].verify(bytes.fromhex(signature), payload)
            return True
        except (InvalidSignature, ValueError):
            return False
    expected = hmac.new(keys["private"], payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def sign_will_receipt_payload(payload: bytes) -> dict[str, Any]:
    """Sign canonical Will receipt bytes under the durable governance root.

    The distinct domain tag prevents a receipt signature from being replayed
    as a capability signature.  The returned key identity lets a persisted
    receipt remain independently verifiable after the Will audit deque has
    been evicted or the process has restarted.
    """

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Will receipt payload must be non-empty bytes")
    keys = _KeyMaterial.load()
    return {
        "signature": _sign(_WILL_RECEIPT_SIGNING_TAG + payload),
        "signature_scheme": str(keys["algorithm"]),
        "signature_key_id": str(keys["key_id"]),
        "trust_root_durable": bool(keys["asymmetric"] and keys["persisted"]),
    }


def verify_will_receipt_payload(
    payload: bytes,
    *,
    signature: str,
    signature_scheme: str,
    signature_key_id: str,
    require_durable: bool = False,
) -> bool:
    """Verify persisted Will receipt bytes without consulting live history."""

    if not isinstance(payload, bytes) or not payload:
        return False
    if not all(
        isinstance(value, str) and bool(value)
        for value in (signature, signature_scheme, signature_key_id)
    ):
        return False
    keys = _KeyMaterial.load()
    if signature_scheme != str(keys["algorithm"]):
        return False
    if require_durable and not bool(keys["asymmetric"] and keys["persisted"]):
        return False
    if require_durable and signature_scheme != "ed25519":
        return False
    return _verify_signature(
        _WILL_RECEIPT_SIGNING_TAG + payload,
        signature,
        signature_key_id,
    )


# ---------------------------------------------------------------------------
# Nonce ledger (replay defence)
# ---------------------------------------------------------------------------


class NonceLedger:
    """Single-use enforcement for capability nonces.

    Persisted so a capability captured before a restart cannot be replayed
    after one while still inside its TTL. Bounded, and pruned of entries whose
    capabilities have expired anyway — an expired capability is already
    rejected on the expiry check, so retaining its nonce buys nothing.
    """

    def __init__(self, path: Path | None = None):
        self._lock = threading.RLock()
        self._seen: dict[str, float] = {}
        self._path = path or (_key_dir().parent / "governance" / "capability_nonces.json")
        self._lock_path = self._path.with_name(f".{self._path.name}.lock")
        self._healthy = True
        self._error = ""
        self._load()

    def _load(self) -> None:
        try:
            with self._lock, interprocess_file_lock(self._lock_path):
                self._seen = self._read_disk_locked(time.time())
                self._mark_healthy_locked()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                self._mark_unhealthy_locked(exc)

    @staticmethod
    def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def _read_bytes_locked(self) -> bytes | None:
        try:
            before = self._path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"nonce ledger is not a regular file: {self._path}")
        if before.st_size > _NONCE_LEDGER_MAX_BYTES:
            raise ValueError(
                f"nonce ledger exceeds {_NONCE_LEDGER_MAX_BYTES} bytes: {self._path}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(str(self._path), flags)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError(f"nonce ledger is not a regular file: {self._path}")
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError("nonce ledger changed while opening")
            if hasattr(os, "getuid") and opened.st_uid != os.getuid():
                raise PermissionError("nonce ledger is not owned by the current user")
            if opened.st_mode & 0o077:
                os.fchmod(fd, 0o600)
                logger.warning("Restricted nonce ledger permissions to 0600 at %s", self._path)
                opened = os.fstat(fd)
            chunks: list[bytes] = []
            total = 0
            remaining = opened.st_size
            while remaining > 0:
                chunk = os.read(fd, min(64 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("nonce ledger truncated while reading")
                chunks.append(chunk)
                total += len(chunk)
                remaining -= len(chunk)
                if total > _NONCE_LEDGER_MAX_BYTES:
                    raise ValueError("nonce ledger grew beyond its maximum while reading")
            after = os.fstat(fd)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ):
                raise RuntimeError("nonce ledger changed while reading")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _read_disk_locked(self, now: float) -> dict[str, float]:
        encoded = self._read_bytes_locked()
        if encoded is None:
            return {}
        try:
            data = json.loads(encoded, object_pairs_hook=self._json_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"nonce ledger is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("nonce ledger root must be an object")

        schema = data.get("schema")
        version = data.get("version")
        if schema is not None or version is not None:
            if schema != _NONCE_LEDGER_SCHEMA or version != _NONCE_LEDGER_VERSION:
                raise ValueError(f"unsupported nonce ledger schema/version: {schema!r}/{version!r}")
        saved_at = data.get("saved_at")
        if saved_at is not None and (
            isinstance(saved_at, bool)
            or not isinstance(saved_at, (int, float))
            or not math.isfinite(float(saved_at))
        ):
            raise ValueError("nonce ledger saved_at must be finite")
        raw_nonces = data.get("nonces")
        if not isinstance(raw_nonces, dict):
            raise ValueError("nonce ledger nonces must be an object")

        retained: dict[str, float] = {}
        for nonce, raw_expiry in raw_nonces.items():
            if not isinstance(nonce, str) or not nonce or len(nonce) > 512:
                raise ValueError("nonce ledger contains an invalid nonce")
            if (
                isinstance(raw_expiry, bool)
                or not isinstance(raw_expiry, (int, float))
                or not math.isfinite(float(raw_expiry))
            ):
                raise ValueError(f"nonce {nonce[:12]!r} has an invalid expiry")
            expiry = float(raw_expiry)
            if expiry > now:
                retained[nonce] = expiry
        if len(retained) > _NONCE_LEDGER_CAP:
            raise ValueError(
                f"nonce ledger has {len(retained)} live entries; cap is {_NONCE_LEDGER_CAP}"
            )
        return retained

    def _persist_locked(self, nonces: dict[str, float]) -> None:
        payload = json.dumps(
            {
                "schema": _NONCE_LEDGER_SCHEMA,
                "version": _NONCE_LEDGER_VERSION,
                "nonces": nonces,
                "saved_at": time.time(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if len(payload) > _NONCE_LEDGER_MAX_BYTES:
            raise ValueError("serialized nonce ledger exceeds its maximum size")
        atomic_write_bytes(self._path, payload, durable=True, mode=0o600)

    def _mark_healthy_locked(self) -> None:
        self._healthy = True
        self._error = ""

    def _mark_unhealthy_locked(self, exc: BaseException | str) -> str:
        detail = str(exc) or type(exc).__name__
        if self._healthy or detail != self._error:
            logger.error(
                "Capability nonce ledger at %s unavailable (%s); refusing all "
                "capability consumption until durable replay protection recovers.",
                self._path,
                detail,
            )
        self._healthy = False
        self._error = detail
        return detail

    def _prune(self, now: float) -> None:
        for k in [nonce for nonce, expiry in self._seen.items() if expiry <= now]:
            self._seen.pop(k, None)

    def consume(self, nonce: str, expires_at: float) -> bool:
        """Claim a nonce. False if it was already claimed (a replay)."""
        accepted, _error = self.consume_with_reason(nonce, expires_at)
        return accepted

    def consume_with_reason(self, nonce: str, expires_at: float) -> tuple[bool, str | None]:
        """Durably claim ``nonce`` before returning success.

        ``(False, None)`` means a genuine replay. A non-empty reason means the
        ledger could not establish durable single use and execution must fail
        closed as an infrastructure denial.
        """

        now = time.time()
        if not isinstance(nonce, str) or not nonce or len(nonce) > 512:
            return False, "nonce is empty or exceeds 512 characters"
        if (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float))
            or not math.isfinite(float(expires_at))
            or float(expires_at) <= now
        ):
            return False, "nonce expiry is invalid or not in the future"
        try:
            with self._lock, interprocess_file_lock(self._lock_path):
                self._seen = self._read_disk_locked(now)
                self._prune(now)
                if nonce in self._seen:
                    self._mark_healthy_locked()
                    return False, None
                if len(self._seen) >= _NONCE_LEDGER_CAP:
                    return False, self._mark_unhealthy_locked(
                        f"live nonce capacity {_NONCE_LEDGER_CAP} reached"
                    )
                candidate = dict(self._seen)
                candidate[nonce] = float(expires_at)
                self._persist_locked(candidate)
                self._seen = candidate
                self._mark_healthy_locked()
                return True, None
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                return False, self._mark_unhealthy_locked(exc)

    def seen(self, nonce: str) -> bool:
        try:
            with self._lock, interprocess_file_lock(self._lock_path):
                self._seen = self._read_disk_locked(time.time())
                self._mark_healthy_locked()
                return nonce in self._seen
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                self._mark_unhealthy_locked(exc)
                return True

    def size(self) -> int:
        return int(self.status()["size"])

    def status(self) -> dict[str, Any]:
        try:
            with self._lock, interprocess_file_lock(self._lock_path):
                self._seen = self._read_disk_locked(time.time())
                self._mark_healthy_locked()
                return {"healthy": True, "error": "", "size": len(self._seen)}
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            with self._lock:
                self._mark_unhealthy_locked(exc)
                return {
                    "healthy": False,
                    "error": self._error,
                    "size": len(self._seen),
                }

    def flush(self) -> None:
        """Compatibility no-op: every successful consumption is already durable."""
        self.status()

    def reset(self) -> None:
        """Clear only this object's cache; never erase persisted replay history."""
        with self._lock:
            self._seen.clear()
            self._healthy = True
            self._error = ""


_ledger: NonceLedger | None = None
_ledger_lock = threading.RLock()


def get_nonce_ledger() -> NonceLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = NonceLedger()
        return _ledger


def reset_capability_chain() -> None:
    """Test seam. Drops cached keys and the nonce ledger."""
    global _ledger
    with _ledger_lock:
        _ledger = None
    _KeyMaterial.reset()


# ---------------------------------------------------------------------------
# Issuer — only the Will
# ---------------------------------------------------------------------------


class WillCapabilityIssuer:
    """Mints signed capabilities from real Will decisions.

    There is deliberately no ``issue(domain, action)`` convenience overload that
    skips the decision. A capability exists only as the signed shadow of a
    decision the Will actually made; if you have no decision, you get no
    capability. That is the invariant this whole module exists to enforce.
    """

    def __init__(self, issuer_name: str = "UnifiedWill"):
        self._issuer = issuer_name
        self._revoked: set[str] = set()
        self._lock = threading.RLock()

    # -- issue ------------------------------------------------------------
    def issue_from_decision(
        self,
        decision: Any,
        *,
        action: str,
        payload: Any = None,
        action_digest: str | None = None,
        scope: str = "",
        ttl_s: float = DEFAULT_TTL_S,
    ) -> SignedCapability:
        """Mint a capability bound to ``decision`` and this exact action.

        Raises:
            CapabilityViolation: if the decision did not approve. A refusal can
                never become a capability — that is the whole point of asking.
        """
        outcome = _outcome_str(decision)
        if outcome not in APPROVING_OUTCOMES:
            raise CapabilityViolation(
                CapabilityDenial.NOT_APPROVED,
                f"Will returned '{outcome}' for action '{action}' — refusing to mint",
                sink=action,
            )

        domain = _domain_str(decision)
        if not domain:
            raise CapabilityViolation(
                CapabilityDenial.MALFORMED,
                f"decision for '{action}' carries no domain",
                sink=action,
            )

        digest = action_digest or compute_action_digest(action, payload)
        ttl = max(1.0, min(float(ttl_s), MAX_TTL_S))
        now = time.time()
        keys = _KeyMaterial.load()

        unsigned = SignedCapability(
            capability_id=f"cap-{uuid.uuid4()}",
            schema_version=CAPABILITY_SCHEMA_VERSION,
            outcome=outcome,
            domain=domain,
            action_digest=digest,
            issuer=self._issuer,
            key_id=str(keys["key_id"]),
            nonce=uuid.uuid4().hex,
            receipt_id=str(getattr(decision, "receipt_id", "") or ""),
            scope=str(scope or ""),
            constraints=tuple(str(c) for c in (getattr(decision, "constraints", ()) or ())),
            issued_at=now,
            expires_at=now + ttl,
        )
        signed = replace(unsigned, signature=_sign(unsigned.signing_payload()))
        logger.debug("Issued capability %s", signed.redacted())
        return signed

    # -- revoke -----------------------------------------------------------
    def revoke(self, capability_id: str) -> None:
        with self._lock:
            self._revoked.add(str(capability_id))

    def is_revoked(self, capability_id: str) -> bool:
        with self._lock:
            return str(capability_id) in self._revoked

    @property
    def issuer_name(self) -> str:
        return self._issuer


def _outcome_str(decision: Any) -> str:
    raw = getattr(decision, "outcome", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


def _domain_str(decision: Any) -> str:
    raw = getattr(decision, "domain", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


_issuer: WillCapabilityIssuer | None = None
_issuer_lock = threading.RLock()


def get_capability_issuer() -> WillCapabilityIssuer:
    global _issuer
    with _issuer_lock:
        if _issuer is None:
            _issuer = WillCapabilityIssuer()
        return _issuer


# ---------------------------------------------------------------------------
# Verifier — every consequential sink
# ---------------------------------------------------------------------------


class CapabilityVerifier:
    """Verifies capabilities at a sink. Holds no minting authority.

    Under Ed25519 this object literally cannot produce a capability it would
    accept: it touches only the public key. That is the structural property the
    old dict-lookup token could not offer.
    """

    def verify(
        self,
        capability: Any,
        *,
        expected_domain: Any = None,
        expected_action_digest: str | None = None,
        consume: bool = True,
        now: float | None = None,
    ) -> VerificationResult:
        now = time.time() if now is None else now

        # -- shape --------------------------------------------------------
        if capability is None:
            return VerificationResult(False, CapabilityDenial.MISSING, "no capability presented")
        if isinstance(capability, dict):
            try:
                cap = SignedCapability.from_dict(capability)
            except ValueError as exc:
                return VerificationResult(False, CapabilityDenial.MALFORMED, str(exc))
        elif isinstance(capability, SignedCapability):
            cap = capability
        else:
            return VerificationResult(
                False,
                CapabilityDenial.MALFORMED,
                f"unsupported capability type {type(capability).__name__}",
            )

        if cap.schema_version != CAPABILITY_SCHEMA_VERSION:
            return VerificationResult(
                False,
                CapabilityDenial.SCHEMA_MISMATCH,
                f"schema {cap.schema_version} != {CAPABILITY_SCHEMA_VERSION}",
            )

        # -- signature first ---------------------------------------------
        # Everything below trusts the field values, so nothing below may run
        # until the signature proves those values are the Will's and unaltered.
        if not _verify_signature(cap.signing_payload(), cap.signature, cap.key_id):
            return VerificationResult(
                False,
                CapabilityDenial.BAD_SIGNATURE,
                f"signature does not verify under key {cap.key_id}",
            )

        # -- authority ----------------------------------------------------
        if cap.outcome not in APPROVING_OUTCOMES:
            return VerificationResult(
                False, CapabilityDenial.NOT_APPROVED, f"outcome '{cap.outcome}'"
            )

        if get_capability_issuer().is_revoked(cap.capability_id):
            return VerificationResult(False, CapabilityDenial.REVOKED, cap.capability_id)

        # -- validity window ----------------------------------------------
        if now + _CLOCK_SKEW_TOLERANCE_S < cap.issued_at:
            return VerificationResult(
                False,
                CapabilityDenial.NOT_YET_VALID,
                f"issued {cap.issued_at - now:.3f}s in the future",
            )
        if now >= cap.expires_at:
            return VerificationResult(
                False,
                CapabilityDenial.EXPIRED,
                f"expired {now - cap.expires_at:.3f}s ago",
            )

        # -- binding ------------------------------------------------------
        if expected_domain is not None:
            want = str(getattr(expected_domain, "value", expected_domain)).strip().lower()
            if want != cap.domain:
                return VerificationResult(
                    False,
                    CapabilityDenial.DOMAIN_MISMATCH,
                    f"issued for '{cap.domain}', presented at '{want}'",
                )

        if expected_action_digest is not None:
            if not hmac.compare_digest(str(expected_action_digest), cap.action_digest):
                return VerificationResult(
                    False,
                    CapabilityDenial.ACTION_MISMATCH,
                    f"issued for {cap.action_digest[:12]}…, "
                    f"presented for {str(expected_action_digest)[:12]}…",
                )

        # -- single use ---------------------------------------------------
        # Last, so a capability is not burned by a request that fails an
        # earlier check — otherwise a domain typo would consume real authority.
        if consume:
            consumed, ledger_error = get_nonce_ledger().consume_with_reason(
                cap.nonce, cap.expires_at
            )
            if not consumed and ledger_error:
                return VerificationResult(
                    False,
                    CapabilityDenial.LEDGER_UNAVAILABLE,
                    f"durable replay protection unavailable: {ledger_error}",
                )
            if not consumed:
                return VerificationResult(
                    False, CapabilityDenial.REPLAYED, f"nonce already used ({cap.capability_id})"
                )

        return VerificationResult(True, None, "", cap)


_verifier: CapabilityVerifier | None = None
_verifier_lock = threading.RLock()


def get_capability_verifier() -> CapabilityVerifier:
    global _verifier
    with _verifier_lock:
        if _verifier is None:
            _verifier = CapabilityVerifier()
        return _verifier


# ---------------------------------------------------------------------------
# The sink-side enforcement point
# ---------------------------------------------------------------------------

_CAPABILITY_CTX_KEY = "signed_capability"


def attach_capability(ctx: dict[str, Any], cap: SignedCapability) -> dict[str, Any]:
    """Put a capability into an execution context for a downstream sink."""
    ctx[_CAPABILITY_CTX_KEY] = cap.to_dict()
    ctx["capability_id"] = cap.capability_id
    ctx["will_receipt_id"] = cap.receipt_id
    return ctx


def capability_from_context(ctx: Any) -> Any:
    if not isinstance(ctx, dict):
        return None
    return ctx.get(_CAPABILITY_CTX_KEY)


def enforce_capability(
    ctx: Any,
    *,
    sink: str,
    domain: Any,
    action: str,
    payload: Any = None,
    action_digest: str | None = None,
    consume: bool = True,
) -> SignedCapability:
    """Fail-closed authority check for a consequential sink.

    This is the function that closes the chain. Call it at the moment of
    execution — not at plan time, not at admission time — so that what is
    authorized is exactly what runs.

    Raises:
        CapabilityViolation: always, when authority cannot be established.
    """
    digest = action_digest or compute_action_digest(action, payload)
    result = get_capability_verifier().verify(
        capability_from_context(ctx),
        expected_domain=domain,
        expected_action_digest=digest,
        consume=consume,
    )
    if not result.ok:
        logger.warning(
            "🔒 Capability DENIED at sink '%s' for action '%s': %s (%s)",
            sink,
            action,
            (result.denial.value if result.denial else "unknown"),
            result.detail,
        )
    return result.raise_if_denied(sink=sink)


__all__ = [
    "APPROVING_OUTCOMES",
    "CAPABILITY_SCHEMA_VERSION",
    "CapabilityDenial",
    "CapabilityVerifier",
    "CapabilityViolation",
    "NonceLedger",
    "SignedCapability",
    "VerificationResult",
    "WillCapabilityIssuer",
    "attach_capability",
    "capability_chain_status",
    "capability_enforcement_mode",
    "capability_from_context",
    "compute_action_digest",
    "enforce_capability",
    "get_capability_issuer",
    "get_capability_verifier",
    "get_nonce_ledger",
    "issuer_is_asymmetric",
    "reset_capability_chain",
    "sign_will_receipt_payload",
    "verify_will_receipt_payload",
]
"""
    core.governance.capability_chain — end of module
"""
