"""Who a non-parametric store belongs to, and where each entry came from.

The datastore holds hidden-state vectors keyed to next tokens. Two things
decide whether a stored vector means anything to a running model: the
model that produced it, and the tokenizer whose ids it names. CP126
``aba3eb39`` found that neither was recorded — the store's whole identity
was its hidden WIDTH. Two models of the same width shared a store, so one
model's vectors and the other's token ids were combined, and the code
reported successful reuse while doing it.

CP126 ``ff3a4505`` found the second half. The module's own docstring says
the store is fed only verifier-clean trusted knowledge, and ``add``
accepted any vector, any token id and any weight from any caller with no
receipt, no provenance and no source. Nothing could later prove why an
entry was admitted, and nothing could revoke every entry from a source
that turned out to be wrong.

And ``62309dad``: entries carried no principal, so one person's query
searched another person's memory.

So identity and provenance are types, and they travel with the data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "CENTERING_VERSION",
    "STORE_IDENTITY_SCHEMA",
    "ENTRY_PROVENANCE_SCHEMA",
    "EntryProvenance",
    "StoreIdentity",
    "TrustLevel",
    "identity_from_mapping",
]

STORE_IDENTITY_SCHEMA = "aura.nonparametric_memory.store_identity.v1"
ENTRY_PROVENANCE_SCHEMA = "aura.nonparametric_memory.entry_provenance.v1"

#: Bumped whenever the anisotropy correction changes shape. A persisted
#: running mean produced by a different centering scheme is not comparable
#: to one produced by this scheme, and mixing them silently changes every
#: similarity the gate reads.
CENTERING_VERSION = 1


class TrustLevel:
    """How far an entry may reach. Ordered, weakest first."""

    #: Someone called ``add``. Nothing checked the content.
    UNVERIFIED = "unverified"
    #: A verifier passed it and named itself.
    VERIFIED = "verified"
    #: Aura's own belief store, already through the belief gate.
    BELIEF = "belief"

    ORDER = (UNVERIFIED, VERIFIED, BELIEF)

    @classmethod
    def rank(cls, level: str) -> int:
        try:
            return cls.ORDER.index(str(level))
        except ValueError:
            return 0


@dataclass(frozen=True)
class StoreIdentity:
    """Everything that has to match before two vectors are comparable.

    Width alone was the identity. A checkpoint change, a quantization
    change, a different hidden-state tap or a different tokenizer all
    produce vectors and token ids that mean something else at the same
    width — and the store reported successful reuse.
    """

    dim: int
    checkpoint: str = "unknown"
    architecture: str = "unknown"
    tokenizer_vocab_size: int = 0
    quantization: str = "unknown"
    hidden_state_tap: str = "last_token"
    adapter: str = ""
    centering_version: int = CENTERING_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = STORE_IDENTITY_SCHEMA
        return payload

    def fingerprint(self) -> str:
        """Stable digest of every field. The store's real name."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

    def slug(self) -> str:
        """Filesystem-safe name, width first so a human can read it."""
        return f"{self.dim}_{self.fingerprint()}"

    def compatible_with(self, other: StoreIdentity) -> tuple[bool, str]:
        """Whether vectors under ``other`` may be read under this identity."""
        if self.dim != other.dim:
            return False, f"dim {other.dim} != {self.dim}"
        for field_name in (
            "checkpoint",
            "architecture",
            "quantization",
            "hidden_state_tap",
            "adapter",
        ):
            mine = getattr(self, field_name)
            theirs = getattr(other, field_name)
            # "unknown" on either side is not a match. A store written before
            # the identity existed cannot claim compatibility with a known
            # model; that assumption is the finding.
            if mine != theirs:
                return False, f"{field_name} {theirs!r} != {mine!r}"
        if self.tokenizer_vocab_size != other.tokenizer_vocab_size:
            return False, (
                f"tokenizer_vocab_size {other.tokenizer_vocab_size} != "
                f"{self.tokenizer_vocab_size}"
            )
        if self.centering_version != other.centering_version:
            return False, (
                f"centering_version {other.centering_version} != {self.centering_version}"
            )
        return True, "identity matches"


@dataclass(frozen=True)
class EntryProvenance:
    """Why one entry was admitted, and who can revoke it.

    ``source_id`` is the revocation handle: when a source turns out to be
    wrong, every entry that named it can be dropped in one call. Without
    it a discredited source's entries were indistinguishable from any
    other and stayed in the store forever.
    """

    source_id: str
    trust: str = TrustLevel.UNVERIFIED
    verifier: str = ""
    evidence_id: str = ""
    principal: str = "anonymous"
    content_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = ENTRY_PROVENANCE_SCHEMA
        return payload

    @property
    def trust_rank(self) -> int:
        return TrustLevel.rank(self.trust)


def identity_from_mapping(value: Any, *, dim: int) -> StoreIdentity | None:
    """Rebuild an identity from a persisted mapping, or None when unreadable."""
    if not isinstance(value, dict) or value.get("schema") != STORE_IDENTITY_SCHEMA:
        return None
    try:
        return StoreIdentity(
            dim=int(value.get("dim", dim)),
            checkpoint=str(value.get("checkpoint", "unknown")),
            architecture=str(value.get("architecture", "unknown")),
            tokenizer_vocab_size=int(value.get("tokenizer_vocab_size", 0) or 0),
            quantization=str(value.get("quantization", "unknown")),
            hidden_state_tap=str(value.get("hidden_state_tap", "last_token")),
            adapter=str(value.get("adapter", "")),
            centering_version=int(value.get("centering_version", 0) or 0),
        )
    except (TypeError, ValueError):
        return None
