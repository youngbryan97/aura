"""Keychain-only contact identities for Aura's private communication surfaces.

Destinations never enter source, configuration files, logs, prompts, receipts,
or public status payloads. macOS Keychain is the sole persistence boundary. A
per-install HMAC key gives runtime components a stable, non-enumerable endpoint
reference for journals and telemetry without publishing a raw destination hash.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock
from core.security.zenith_secrets import KeychainBackend, require_keychain_backend

DEFAULT_MESSAGES_CONTACT_ALIAS = "primary_operator"
_KEYCHAIN_SERVICE = "AuraMessagesContacts.v1"
_DIGEST_KEY_ACCOUNT = "__endpoint_digest_key__"
_CONTACT_ACCOUNT_PREFIX = "contact."
_SCHEMA = "aura.messages_contact.v1"
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_E164_RE = re.compile(r"^\+[1-9][0-9]{7,14}$")
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_SERVICE_PREFERENCES = frozenset({"auto", "imessage", "sms"})
_RECORD_KEYS = frozenset(
    {
        "alias",
        "allow_inbound",
        "allow_outbound",
        "created_at",
        "destination",
        "destination_kind",
        "endpoint_ref",
        "record_mac",
        "schema",
        "service_preference",
        "updated_at",
    }
)


class ContactDirectoryError(RuntimeError):
    """A contact record was missing, malformed, or could not be persisted."""


class ContactNotConfiguredError(ContactDirectoryError):
    """The requested alias does not have a Keychain record."""


@dataclass(frozen=True, slots=True)
class MessagesContact:
    alias: str
    destination: str = field(repr=False)
    destination_kind: str
    endpoint_ref: str
    service_preference: str
    allow_inbound: bool
    allow_outbound: bool
    created_at: float
    updated_at: float

    def public_status(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "endpoint_ref": self.endpoint_ref,
            "destination_kind": self.destination_kind,
            "service_preference": self.service_preference,
            "allow_inbound": self.allow_inbound,
            "allow_outbound": self.allow_outbound,
            "configured": True,
        }


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validated_alias(value: Any) -> str:
    alias = str(value or "").strip().lower()
    if not _ALIAS_RE.fullmatch(alias):
        raise ValueError("contact alias must be a lowercase symbolic identifier")
    return alias


def normalize_messages_destination(value: Any) -> tuple[str, str]:
    """Return a canonical Messages handle and its non-secret kind.

    Ten-digit North American numbers are accepted only at provisioning time and
    normalized to E.164. Persisted records are always canonical.
    """

    raw = str(value or "").strip()
    if not raw or any(ord(character) < 32 for character in raw):
        raise ValueError("Messages destination is required")
    if "@" in raw:
        candidate = raw.casefold()
        if len(candidate) > 254 or not _EMAIL_RE.fullmatch(candidate):
            raise ValueError("Messages email handle is invalid")
        return candidate, "email"
    digits = "".join(character for character in raw if character.isdigit())
    if raw.startswith("+"):
        candidate = "+" + digits
    elif len(digits) == 10:
        candidate = "+1" + digits
    elif len(digits) == 11 and digits.startswith("1"):
        candidate = "+" + digits
    else:
        raise ValueError("Messages phone handle must be canonical E.164")
    if not _E164_RE.fullmatch(candidate):
        raise ValueError("Messages phone handle must be canonical E.164")
    return candidate, "phone"


class KeychainContactDirectory:
    """Strict contact directory backed only by a macOS Keychain backend."""

    def __init__(
        self,
        backend: KeychainBackend | None = None,
        *,
        backend_factory: Callable[[], KeychainBackend] = require_keychain_backend,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._backend = backend
        self._backend_factory = backend_factory
        self._clock = clock
        self._lock = checked_lock("contact_directory", reentrant=True)

    def _resolved_backend(self) -> KeychainBackend:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def _digest_key(self, *, create: bool) -> bytes:
        backend = self._resolved_backend()
        encoded = backend.get_password(_KEYCHAIN_SERVICE, _DIGEST_KEY_ACCOUNT)
        if not encoded:
            if not create:
                raise ContactDirectoryError("Messages contact integrity key is unavailable")
            key = secrets.token_bytes(32)
            encoded = base64.urlsafe_b64encode(key).decode("ascii")
            if not backend.set_password(_KEYCHAIN_SERVICE, _DIGEST_KEY_ACCOUNT, encoded):
                raise ContactDirectoryError("Messages contact integrity key was not persisted")
            confirmed = backend.get_password(_KEYCHAIN_SERVICE, _DIGEST_KEY_ACCOUNT)
            if not confirmed or not hmac.compare_digest(confirmed, encoded):
                raise ContactDirectoryError("Messages contact integrity key write was not confirmed")
            return key
        try:
            key = base64.urlsafe_b64decode(encoded.encode("ascii"))
        except (ValueError, UnicodeError) as exc:
            raise ContactDirectoryError("Messages contact integrity key is malformed") from exc
        if len(key) != 32:
            raise ContactDirectoryError("Messages contact integrity key has an invalid length")
        return key

    @staticmethod
    def _endpoint_ref(key: bytes, destination: str) -> str:
        digest = hmac.new(
            key,
            b"aura.messages.endpoint.v1\0" + destination.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return "msg_" + digest[:32]

    @staticmethod
    def _record_mac(key: bytes, body: dict[str, Any]) -> str:
        return hmac.new(
            key,
            b"aura.messages.contact.v1\0" + _canonical_json(body),
            hashlib.sha256,
        ).hexdigest()

    def provision(
        self,
        alias: str,
        destination: str,
        *,
        service_preference: str = "auto",
        allow_inbound: bool = True,
        allow_outbound: bool = True,
    ) -> MessagesContact:
        normalized_alias = _validated_alias(alias)
        normalized_destination, destination_kind = normalize_messages_destination(destination)
        preference = str(service_preference or "auto").strip().lower()
        if preference not in _SERVICE_PREFERENCES:
            raise ValueError("Messages service preference must be auto, imessage, or sms")
        if not bool(allow_inbound) and not bool(allow_outbound):
            raise ValueError("Messages contact must permit inbound or outbound communication")
        with self._lock:
            key = self._digest_key(create=True)
            backend = self._resolved_backend()
            now = float(self._clock())
            created_at = now
            try:
                existing = self.load(normalized_alias)
                created_at = existing.created_at
            except ContactNotConfiguredError:
                pass
            body: dict[str, Any] = {
                "alias": normalized_alias,
                "allow_inbound": bool(allow_inbound),
                "allow_outbound": bool(allow_outbound),
                "created_at": created_at,
                "destination": normalized_destination,
                "destination_kind": destination_kind,
                "endpoint_ref": self._endpoint_ref(key, normalized_destination),
                "schema": _SCHEMA,
                "service_preference": preference,
                "updated_at": now,
            }
            payload = {**body, "record_mac": self._record_mac(key, body)}
            encoded = _canonical_json(payload).decode("utf-8")
            account = _CONTACT_ACCOUNT_PREFIX + normalized_alias
            if not backend.set_password(_KEYCHAIN_SERVICE, account, encoded):
                raise ContactDirectoryError("Messages contact was not persisted")
            confirmed = backend.get_password(_KEYCHAIN_SERVICE, account)
            if not confirmed or not hmac.compare_digest(confirmed, encoded):
                raise ContactDirectoryError("Messages contact write was not confirmed")
            return self._decode_record(confirmed, alias=normalized_alias, key=key)

    async def provision_async(self, *args: Any, **kwargs: Any) -> MessagesContact:
        return await asyncio.to_thread(self.provision, *args, **kwargs)

    def load(self, alias: str = DEFAULT_MESSAGES_CONTACT_ALIAS) -> MessagesContact:
        normalized_alias = _validated_alias(alias)
        with self._lock:
            backend = self._resolved_backend()
            encoded = backend.get_password(
                _KEYCHAIN_SERVICE,
                _CONTACT_ACCOUNT_PREFIX + normalized_alias,
            )
            if not encoded:
                raise ContactNotConfiguredError(
                    f"Messages contact alias is not configured: {normalized_alias}"
                )
            key = self._digest_key(create=False)
            return self._decode_record(encoded, alias=normalized_alias, key=key)

    async def load_async(
        self,
        alias: str = DEFAULT_MESSAGES_CONTACT_ALIAS,
    ) -> MessagesContact:
        return await asyncio.to_thread(self.load, alias)

    def _decode_record(self, encoded: str, *, alias: str, key: bytes) -> MessagesContact:
        try:
            payload = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise ContactDirectoryError("Messages contact record is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != _RECORD_KEYS:
            raise ContactDirectoryError("Messages contact record has an invalid schema")
        if payload.get("schema") != _SCHEMA or payload.get("alias") != alias:
            raise ContactDirectoryError("Messages contact record identity does not match its alias")
        destination, destination_kind = normalize_messages_destination(payload.get("destination"))
        if destination_kind != payload.get("destination_kind"):
            raise ContactDirectoryError("Messages contact destination kind is inconsistent")
        preference = str(payload.get("service_preference") or "").strip().lower()
        if preference not in _SERVICE_PREFERENCES:
            raise ContactDirectoryError("Messages contact service preference is invalid")
        body = {key_name: value for key_name, value in payload.items() if key_name != "record_mac"}
        expected_mac = self._record_mac(key, body)
        supplied_mac = str(payload.get("record_mac") or "")
        if not hmac.compare_digest(expected_mac, supplied_mac):
            raise ContactDirectoryError("Messages contact integrity verification failed")
        endpoint_ref = self._endpoint_ref(key, destination)
        if not hmac.compare_digest(endpoint_ref, str(payload.get("endpoint_ref") or "")):
            raise ContactDirectoryError("Messages contact endpoint binding is invalid")
        try:
            created_at = float(payload["created_at"])
            updated_at = float(payload["updated_at"])
        except (TypeError, ValueError) as exc:
            raise ContactDirectoryError("Messages contact timestamps are invalid") from exc
        if created_at <= 0.0 or updated_at < created_at:
            raise ContactDirectoryError("Messages contact timestamps are inconsistent")
        allow_inbound = payload.get("allow_inbound")
        allow_outbound = payload.get("allow_outbound")
        if not isinstance(allow_inbound, bool) or not isinstance(allow_outbound, bool):
            raise ContactDirectoryError("Messages contact permissions are invalid")
        if not allow_inbound and not allow_outbound:
            raise ContactDirectoryError("Messages contact has no permitted direction")
        return MessagesContact(
            alias=alias,
            destination=destination,
            destination_kind=destination_kind,
            endpoint_ref=endpoint_ref,
            service_preference=preference,
            allow_inbound=allow_inbound,
            allow_outbound=allow_outbound,
            created_at=created_at,
            updated_at=updated_at,
        )


__all__ = [
    "ContactDirectoryError",
    "ContactNotConfiguredError",
    "DEFAULT_MESSAGES_CONTACT_ALIAS",
    "KeychainContactDirectory",
    "MessagesContact",
    "normalize_messages_destination",
]
