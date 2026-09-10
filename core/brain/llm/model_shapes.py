"""The four shapes the cortex registry hands around.

What an active cortex IS, what an artifact resolution found, and what each
serving lane is allowed. Plain frozen records with no reads and no policy —
they decode the JSON they were handed and answer questions about themselves,
which is why they can sit apart from the registry that fills them.

Moved out because ``model_registry`` was past the 2,000-line ceiling above
which a new module is never grandfathered. Nothing here imports the registry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ActiveCortexSpec",
    "canonical_contract_json",
    "contract_digest",
    "CortexBoundArtifactResolution",
    "CortexServingLaneLimits",
    "CortexServingLimits",
]


@dataclass(frozen=True)
class ActiveCortexSpec:
    """Immutable, validated view of the one active cortex pointer.

    JSON dictionaries are retained as canonical strings so callers cannot
    mutate the registry's cached authority object.  Accessors return fresh
    values when a subsystem needs the complete contract.
    """

    manifest_path: Path
    pointer_sha256: str
    schema_version: int
    model_path: Path
    base_model: str
    tag: str
    size_class: str
    descriptor_sha256: str
    repository_id: str
    revision: str
    serving_profile_sha256: str
    migration_contract_sha256: str
    evaluation_sha256: str
    exact_identity: bool
    promotion_qualified: bool
    predecessor_pointer_sha256: str = ""
    identity_transition_sha256: str = ""
    identity_transition_verified: bool = False
    _artifact_descriptor_json: str = ""
    _serving_profile_json: str = ""
    _migration_contract_json: str = ""
    _evaluation_json: str = ""

    @staticmethod
    def _decode(value: str) -> dict[str, object] | None:
        if not value:
            return None
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else None

    def artifact_descriptor(self) -> dict[str, object] | None:
        return self._decode(self._artifact_descriptor_json)

    def serving_profile(self) -> dict[str, object] | None:
        return self._decode(self._serving_profile_json)

    def migration_contract(self) -> dict[str, object] | None:
        return self._decode(self._migration_contract_json)

    def evaluation(self) -> dict[str, object] | None:
        """Return a fresh copy of the validated promotion evaluation."""

        return self._decode(self._evaluation_json)


@dataclass(frozen=True)
class CortexBoundArtifactResolution:
    """Admission result for tissue that belongs only to the active cortex.

    A non-cortex lane is not an identity failure. Brainstem, reflex, draft,
    and specialist workers legitimately load other checkpoints, so callers
    need a quiet way to distinguish those workers from a corrupted active
    cortex identity.
    """

    status: str
    model_path: Path | None = None
    descriptor: dict[str, object] | None = None
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.status == "matched"


@dataclass(frozen=True)
class CortexServingLaneLimits:
    """One qualified input/output envelope from the active cortex profile."""

    name: str
    max_input_tokens: int
    max_output_tokens: int


@dataclass(frozen=True)
class CortexServingLimits:
    """Immutable serving limits bound to one exact active model artifact."""

    model_path: Path
    descriptor_sha256: str
    profile_sha256: str
    source: str
    qualified: bool
    served_context_tokens: int
    prefill_chunk_tokens: int
    lanes: tuple[CortexServingLaneLimits, ...]

    def lane(self, name: str) -> CortexServingLaneLimits | None:
        normalized = str(name or "").strip().lower()
        return next((lane for lane in self.lanes if lane.name == normalized), None)


def canonical_contract_json(value: dict[str, object] | None) -> str:
    if not isinstance(value, dict):
        return ""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def contract_digest(value: dict[str, object], *, digest_key: str) -> str:
    material = dict(value)
    material.pop(digest_key, None)
    return hashlib.sha256(canonical_contract_json(material).encode("ascii")).hexdigest()
