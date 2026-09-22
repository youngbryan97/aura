"""A durable entity key, so identity stops being a function of transient ids.

State lineage in this codebase is strong: every AuraState knows its parent, the
version is monotonic, commit admission is serialized, and the continuity hash
fingerprints identity, affect, cognition and world without the volatile fields.

The identity ANCHOR was not. ``IdentityAnchor.get_identity`` returned

    state.identity.name + "-" + state.state_id[:8]

and ``state_id`` is a fresh uuid4 on every derived state. So the one object
whose job is to say "this is the same entity across restarts and evolutions"
changed several times a second, and changed completely on restart. The lineage
was signed to nothing.

This is the anchor the lineage deserved:

    K_Aura → Sign(state_t, parent_t) → Sign(state_{t+1}, state_t)

An Ed25519 keypair, generated once, persisted under the state root, and
independent of every state version. Its fingerprint is the entity id. Each
state link is signed over (entity, state, parent, continuity hash, previous
signature), which makes the chain tamper-evident: altering a historic link
invalidates every signature after it, not just its own.

**Migration and recovery are explicit**, because a key with no rotation story
is a key that eventually gets replaced by silently starting a new identity.
Rotation writes a succession record signed by the OUTGOING key, so a verifier
holding only the original public key can still follow the chain forward. A new
key that cannot produce that signature is a new entity, and this says so rather
than pretending continuity.

What this does NOT settle: nothing here is a claim about personal identity, and
a signed chain proves custody of a key rather than sameness of a self. It makes
the technical claim technically true, which is the part that was false.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

__all__ = [
    "EntityIdentity",
    "LineageLink",
    "SuccessionRecord",
    "entity_identity",
    "reset_entity_identity_for_test",
]

logger = logging.getLogger(__name__)

#: Ed25519 over the raw link payload. Named so a stored record says which
#: scheme produced it rather than leaving a future reader to guess.
_SCHEME = "ed25519-sha256-v1"

#: The stored FILE's schema version, which is a different thing from the
#: signature scheme above and is what the write gateway envelopes on.
_KEY_FILE_SCHEMA_VERSION = 1

_LOCK = checked_lock("entity_identity")
_SINGLETON: EntityIdentity | None = None


@dataclass(frozen=True, slots=True)
class LineageLink:
    """One signed step of the state chain."""

    entity_id: str
    state_id: str
    version: int
    parent_state_id: str
    continuity_hash: str
    previous_signature: str
    signature: str
    scheme: str = _SCHEME
    at: float = field(default_factory=lambda: time.time())

    def payload(self) -> bytes:
        """Exactly the bytes that were signed. One definition, both directions.

        Signing and verification derive the payload from the same method, so
        the two cannot drift into a state where every historic signature is
        unverifiable because the field order changed.
        """

        return json.dumps(
            {
                "entity_id": self.entity_id,
                "state_id": self.state_id,
                "version": int(self.version),
                "parent_state_id": self.parent_state_id,
                "continuity_hash": self.continuity_hash,
                "previous_signature": self.previous_signature,
                "scheme": self.scheme,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "state_id": self.state_id,
            "version": self.version,
            "parent_state_id": self.parent_state_id,
            "continuity_hash": self.continuity_hash,
            "previous_signature": self.previous_signature,
            "signature": self.signature,
            "scheme": self.scheme,
            "at": self.at,
        }


@dataclass(frozen=True, slots=True)
class SuccessionRecord:
    """An old key attesting to a new one. The only legitimate rotation."""

    predecessor_entity_id: str
    successor_entity_id: str
    successor_public_key: str
    reason: str
    signature: str
    scheme: str = _SCHEME
    at: float = field(default_factory=lambda: time.time())

    def payload(self) -> bytes:
        return json.dumps(
            {
                "predecessor_entity_id": self.predecessor_entity_id,
                "successor_entity_id": self.successor_entity_id,
                "successor_public_key": self.successor_public_key,
                "reason": self.reason,
                "scheme": self.scheme,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "predecessor_entity_id": self.predecessor_entity_id,
            "successor_entity_id": self.successor_entity_id,
            "successor_public_key": self.successor_public_key,
            "reason": self.reason,
            "signature": self.signature,
            "scheme": self.scheme,
            "at": self.at,
        }


class EntityIdentity:
    """The key, its id, and the chain it signs.

    Deliberately holds no reference to any AuraState. The whole defect was that
    identity was derived from state; a class that keeps one would reintroduce
    it by a slower route.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else _default_root()
        self._lock = checked_lock("entity_identity_instance")
        self._private_key: Any = None
        self._public_bytes: bytes = b""
        self._entity_id: str = ""
        self._chain_head: str = ""
        self._links: list[LineageLink] = []
        self._successions: list[SuccessionRecord] = []
        #: Persisting fsyncs, and fsync under a lock is how this runtime
        #: freezes — lockdep says so by name. Key material is therefore written
        #: outside every lock, via :meth:`flush`, and this flag is what carries
        #: the intent across the boundary.
        self._needs_persist = False
        self._load_or_create()
        self.flush()

    # ── key material ───────────────────────────────────────────────────────

    @property
    def key_path(self) -> Path:
        return self._root / "entity_key.json"

    @property
    def entity_id(self) -> str:
        """The stable anchor. A fingerprint of the public key, nothing else."""
        return self._entity_id

    @property
    def public_key_hex(self) -> str:
        return self._public_bytes.hex()

    @property
    def chain_head(self) -> str:
        """The most recent link signature, or "" before anything was signed."""
        return self._chain_head

    def _load_or_create(self) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        stored = self._read_key_file()
        if stored is not None:
            try:
                self._private_key = Ed25519PrivateKey.from_private_bytes(
                    bytes.fromhex(stored["private_key"])
                )
                self._public_bytes = bytes.fromhex(stored["public_key"])
                self._entity_id = str(stored["entity_id"])
                self._successions = [
                    SuccessionRecord(**record) for record in stored.get("successions", [])
                ]
                return
            except (KeyError, TypeError, ValueError) as exc:
                record_degradation(
                    "entity_identity",
                    exc,
                    severity="critical",
                    action=(
                        "refused to reuse an unreadable entity key and minted a "
                        "new identity; continuity across this boundary is broken"
                    ),
                )

        self._private_key = Ed25519PrivateKey.generate()
        self._public_bytes = _public_bytes(self._private_key)
        self._entity_id = _fingerprint(self._public_bytes)
        self._needs_persist = True

    def _read_key_file(self) -> dict[str, Any] | None:
        """Read the key back out of the gateway's versioned envelope.

        The gateway wraps every JSON write as ``{schema, schema_version,
        payload}``. Reading the file as though it were the payload silently
        found no private key, minted a fresh one, and reported a critical
        degradation on every boot — an identity module that broke identity on
        exactly the boundary it exists to survive.
        """

        try:
            if not self.key_path.exists():
                return None
            from core.runtime.atomic_writer import read_json_envelope

            envelope = read_json_envelope(self.key_path)
            payload = envelope.get("payload")
            if isinstance(payload, str):
                payload = json.loads(payload)
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, ImportError, json.JSONDecodeError) as exc:
            record_degradation(
                "entity_identity",
                exc,
                severity="error",
                action="could not read the stored entity key",
            )
            return None

    def flush(self) -> None:
        """Write pending key material through the governed gateway.

        Called only from outside this object's lock. The write fsyncs, and a
        blocking op under a lock is the shape that froze the live event loop
        for twenty minutes once already; lockdep flags it by name.
        """

        if not self._needs_persist:
            return
        self._needs_persist = False
        self._persist()

    def _persist(self) -> None:
        """Write the key through the governed gateway, never with open()."""

        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        payload = {
            "scheme": _SCHEME,
            "entity_id": self._entity_id,
            "public_key": self._public_bytes.hex(),
            "private_key": self._private_key.private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption()
            ).hex(),
            "successions": [record.to_dict() for record in self._successions],
            "created_at": time.time(),
        }
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.atomic_writer import ensure_private_directory
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("entity_identity_key"):
                gateway = get_file_write_gateway()
                # The containing directory is the boundary, not a chmod on the
                # file afterwards. A raw `key_path.chmod(0o600)` was outside
                # every write owner, and it tightened the key only AFTER it had
                # already existed at the default mode — a window, and an
                # unaccounted mutation. `ensure_private_directory` makes the
                # directory 0o700 BEFORE the key is written, so there is no
                # moment at which the file is reachable by another user.
                ensure_private_directory(self._root)
                gateway.ensure_directory(self._root, source="entity_identity")
                gateway.write_json(
                    self.key_path,
                    payload,
                    source="entity_identity",
                    schema_version=_KEY_FILE_SCHEMA_VERSION,
                )
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "entity_identity",
                exc,
                severity="critical",
                action=(
                    "entity key was not persisted; identity will not survive "
                    "this process and lineage continuity is broken at restart"
                ),
            )

    # ── the chain ──────────────────────────────────────────────────────────

    def sign_state_link(
        self,
        *,
        state_id: str,
        version: int,
        parent_state_id: str = "",
        continuity_hash: str = "",
    ) -> LineageLink:
        """Sign one state's link to its parent, extending the chain.

        The previous signature is part of the payload, which is what makes the
        chain tamper-evident rather than merely signed: editing an old link
        breaks every signature downstream of it.
        """

        with self._lock:
            link = LineageLink(
                entity_id=self._entity_id,
                state_id=str(state_id or ""),
                version=int(version or 0),
                parent_state_id=str(parent_state_id or ""),
                continuity_hash=str(continuity_hash or ""),
                previous_signature=self._chain_head,
                signature="",
            )
            signature = self._private_key.sign(link.payload()).hex()
            signed = LineageLink(
                entity_id=link.entity_id,
                state_id=link.state_id,
                version=link.version,
                parent_state_id=link.parent_state_id,
                continuity_hash=link.continuity_hash,
                previous_signature=link.previous_signature,
                signature=signature,
                at=link.at,
            )
            self._chain_head = signature
            self._links.append(signed)
            # Bounded: this is a live continuity check, not an archive. The
            # durable record of state history is the state database.
            if len(self._links) > 4096:
                del self._links[: len(self._links) - 4096]
            return signed

    def verify_link(self, link: LineageLink) -> bool:
        """Whether this link was signed by this entity, over these exact bytes."""

        from cryptography.exceptions import InvalidSignature

        try:
            self._public_key().verify(bytes.fromhex(link.signature), link.payload())
            return True
        except (InvalidSignature, ValueError, TypeError) as exc:
            logger.debug(
                "the lineage link did not verify against this entity (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

    def verify_chain(self, links: list[LineageLink] | None = None) -> dict[str, Any]:
        """Verify a run of links: each signature, and each link to the last.

        Returns a report rather than a bool, because "the chain broke" and
        "the chain broke at index 41 on the previous-signature field" are
        different amounts of information and only one of them is useful.
        """

        chain = list(links if links is not None else self._links)
        expected_previous = chain[0].previous_signature if chain else ""
        for index, link in enumerate(chain):
            if not self.verify_link(link):
                return {
                    "valid": False,
                    "broken_at": index,
                    "reason": "signature does not verify",
                    "length": len(chain),
                }
            if link.previous_signature != expected_previous:
                return {
                    "valid": False,
                    "broken_at": index,
                    "reason": "link does not follow its predecessor",
                    "length": len(chain),
                }
            expected_previous = link.signature
        return {"valid": True, "length": len(chain), "head": expected_previous}

    def recent_links(self, limit: int = 16) -> list[LineageLink]:
        with self._lock:
            return list(self._links[-max(1, int(limit)) :])

    # ── rotation ───────────────────────────────────────────────────────────

    def rotate(self, *, reason: str) -> SuccessionRecord:
        """Replace the key, with the outgoing one attesting to its successor.

        A verifier holding only the original public key can follow the chain
        across this. A new key that cannot produce this signature is a new
        entity, and the honest report of that situation is a broken chain, not
        a quietly continued one.
        """

        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        with self._lock:
            successor = Ed25519PrivateKey.generate()
            successor_public = _public_bytes(successor)
            successor_id = _fingerprint(successor_public)
            record = SuccessionRecord(
                predecessor_entity_id=self._entity_id,
                successor_entity_id=successor_id,
                successor_public_key=successor_public.hex(),
                reason=str(reason or "unspecified"),
                signature="",
            )
            signature = self._private_key.sign(record.payload()).hex()
            signed = SuccessionRecord(
                predecessor_entity_id=record.predecessor_entity_id,
                successor_entity_id=record.successor_entity_id,
                successor_public_key=record.successor_public_key,
                reason=record.reason,
                signature=signature,
                at=record.at,
            )
            self._successions.append(signed)
            self._private_key = successor
            self._public_bytes = successor_public
            self._entity_id = successor_id
            # The chain head is carried across deliberately: the succession
            # record is what bridges the keys, and resetting the head here
            # would make the rotation look like a fresh start.
            self._needs_persist = True
        self.flush()
        return signed

    def verify_succession(
        self, record: SuccessionRecord, predecessor_public_key_hex: str
    ) -> bool:
        """Check a succession against the key it claims to come from."""

        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            public = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(predecessor_public_key_hex)
            )
            public.verify(bytes.fromhex(record.signature), record.payload())
            return True
        except (InvalidSignature, ValueError, TypeError) as exc:
            logger.debug(
                "the succession record did not verify against the predecessor key (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

    def successions(self) -> list[SuccessionRecord]:
        return list(self._successions)

    def report(self) -> dict[str, Any]:
        """The identity position, from the key rather than from a state field."""

        return {
            "entity_id": self._entity_id,
            "public_key": self.public_key_hex,
            "scheme": _SCHEME,
            "key_path": str(self.key_path),
            "chain_head": self._chain_head,
            "links_held": len(self._links),
            "successions": [record.to_dict() for record in self._successions],
            "anchored_to": (
                "an Ed25519 keypair persisted under the state root, independent "
                "of state_id and state version"
            ),
        }

    def _public_key(self) -> Any:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        return Ed25519PublicKey.from_public_bytes(self._public_bytes)


def _public_bytes(private_key: Any) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _fingerprint(public_bytes: bytes) -> str:
    """The entity id: a truncated digest of the public key, prefixed.

    Prefixed so it is recognisable in a log line as an entity id rather than as
    an arbitrary hash, and truncated because it is an identifier rather than a
    security boundary — the signature is the security boundary.
    """

    return "aura:" + hashlib.sha256(public_bytes).hexdigest()[:32]


def _default_root() -> Path:
    try:
        from core.runtime.state_ownership import state_root

        return Path(state_root()) / "identity"
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        record_degradation(
            "entity_identity",
            exc,
            severity="error",
            action="fell back to a local identity directory for the entity key",
        )
        return Path("data") / "identity"


def entity_identity(root: Path | None = None) -> EntityIdentity:
    """The process-wide entity identity.

    Construction happens OUTSIDE ``_LOCK``. Building the identity may write the
    key file, and a write under a lock is the failure mode lockdep exists to
    catch — holding the module lock across construction put an fsync under it
    on the very first call. The cost is that two threads racing the first call
    can both build one; the first to publish wins and the loser is discarded,
    which is harmless because both read or create the same durable key.
    """

    global _SINGLETON
    if root is not None:
        return EntityIdentity(root)
    existing = _SINGLETON
    if existing is not None:
        return existing
    instance = EntityIdentity(None)
    with _LOCK:
        if _SINGLETON is None:
            _SINGLETON = instance
        return _SINGLETON


def reset_entity_identity_for_test() -> None:
    global _SINGLETON
    with _LOCK:
        _SINGLETON = None
