"""Bounded compatibility inventory for cortex-facing neural tissue.

The inventory does not load a model and does not infer compatibility from tensor
width. It validates one existing model artifact descriptor, reads bounded tissue
metadata, and separates reusable data or recipes from checkpoint-basis state.
"""

from __future__ import annotations

import hashlib
import json
import stat
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.brain.llm.model_artifact_profile import validate_model_artifact_descriptor

INVENTORY_SCHEMA: Final = "aura.model_tissue_migration_inventory.v1"
ENTRY_SCHEMA: Final = "aura.model_tissue_migration_entry.v1"

OUTCOMES: Final = frozenset({"qualified", "retrain", "retire", "refuse"})
BASIS_CLASSES: Final = frozenset(
    {
        "architecture_independent",
        "checkpoint_basis",
        "activation_basis",
        "mixed_tokenized_data",
        "ephemeral_checkpoint_basis",
    }
)
FAMILIES: Final = (
    "persona_crsm",
    "caa_steering",
    "recurrent_tissue",
    "fast_weight_adapters",
    "expert_adapters",
)

MAX_METADATA_BYTES: Final = 4 * 1024 * 1024
MAX_REUSABLE_INPUT_BYTES: Final = 64 * 1024 * 1024
MAX_PROBES: Final = 96
MAX_JSON_NODES: Final = 50_000
MAX_DATASET_ROWS: Final = 10_000
MAX_DISCOVERED_ADAPTERS: Final = 32

_IDENTITY_SCAN_SKIP_KEYS: Final = frozenset(
    {
        "answer_tokens",
        "data",
        "examples",
        "input_ids",
        "labels",
        "prompt_tokens",
        "records",
        "samples",
        "token_ids",
        "tokens",
    }
)


class TissueInventoryError(RuntimeError):
    """The inventory could not establish a bounded, trustworthy result."""


@dataclass(frozen=True)
class TissueProbe:
    """One declared artifact or source input to classify."""

    family: str
    artifact_id: str
    artifact_kind: str
    path: Path
    basis_class: str
    mismatch_outcome: str = "retrain"
    required: bool = False
    qualification_scope: str = "candidate_runtime"
    portable_fields: tuple[str, ...] = ()
    basis_bound_fields: tuple[str, ...] = ()
    related_patterns: tuple[str, ...] = ()
    declared_count_field: str = ""
    require_self_digest: bool = False
    linked_file_field: str = ""
    linked_file_name: str = ""
    linked_sha256_field: str = ""

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown_tissue_family:{self.family}")
        if self.basis_class not in BASIS_CLASSES:
            raise ValueError(f"unknown_tissue_basis:{self.basis_class}")
        if self.mismatch_outcome not in {"retrain", "retire", "refuse"}:
            raise ValueError(f"invalid_mismatch_outcome:{self.mismatch_outcome}")
        if self.linked_file_field and self.linked_file_name:
            raise ValueError("linked_file_source_is_ambiguous")
        if bool(self.linked_file_field or self.linked_file_name) != bool(
            self.linked_sha256_field
        ):
            raise ValueError("linked_file_digest_contract_is_incomplete")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _stable_file_bytes(path: Path, *, maximum: int) -> bytes:
    """Read one regular, non-symlink file under an explicit byte ceiling."""

    lexical = path.expanduser().absolute()
    try:
        before = lexical.lstat()
    except OSError as exc:
        raise TissueInventoryError(f"artifact_unreadable:{lexical}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TissueInventoryError(f"artifact_not_regular:{lexical}")
    if not 0 < before.st_size <= maximum:
        raise TissueInventoryError(f"artifact_size_out_of_bounds:{lexical}")
    try:
        with lexical.open("rb") as handle:
            payload = handle.read(maximum + 1)
        after = lexical.lstat()
    except OSError as exc:
        raise TissueInventoryError(f"artifact_unreadable:{lexical}") from exc
    identity = lambda item: (  # noqa: E731
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        len(payload) != before.st_size
        or len(payload) > maximum
        or identity(before) != identity(after)
    ):
        raise TissueInventoryError(f"artifact_changed_while_reading:{lexical}")
    return payload


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _stable_file_bytes(path, maximum=MAX_METADATA_BYTES)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, ValueError) as exc:
        raise TissueInventoryError(f"artifact_json_invalid:{path}") from exc
    if not isinstance(value, dict):
        raise TissueInventoryError(f"artifact_json_not_object:{path}")
    return value, payload


def load_candidate_descriptor(path: Path | str) -> dict[str, Any]:
    """Load one descriptor through bounded strict JSON and Aura's validator."""

    descriptor, _payload = _strict_json(Path(path).expanduser().absolute())
    return validate_model_artifact_descriptor(descriptor)


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    pending = [value]
    visited = 0
    while pending:
        item = pending.pop()
        visited += 1
        if visited > MAX_JSON_NODES:
            raise TissueInventoryError("artifact_json_node_limit_exceeded")
        if isinstance(item, Mapping):
            yield item
            pending.extend(
                child
                for key, child in item.items()
                if key not in _IDENTITY_SCAN_SKIP_KEYS
            )
        elif isinstance(item, list):
            pending.extend(item)


def _candidate_identity(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    weights = descriptor["weight_identity"]
    behavior = descriptor["behavior_identity"]
    profile = descriptor["artifact_profile"]
    config_sha256 = ""
    for record in behavior.get("files", []):
        if isinstance(record, Mapping) and record.get("path") == "config.json":
            config_sha256 = str(record.get("sha256") or "")
            break
    return {
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "canonical_path": descriptor["canonical_path"],
        "repository_id": descriptor.get("repository_id") or "",
        "revision": descriptor.get("revision") or "",
        "weight_fingerprint": weights.get("fingerprint") or "",
        "behavior_bundle_sha256": behavior.get("bundle_sha256") or "",
        "config_sha256": config_sha256,
        "model_type": profile.get("model_type") or "",
        "hidden_size": profile.get("hidden_size") or 0,
        "num_hidden_layers": profile.get("num_hidden_layers") or 0,
        "vocab_size": profile.get("vocab_size") or 0,
        "layer_types": list(profile.get("layer_types") or []),
    }


def _artifact_identity(
    value: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor_hashes: set[str] = set()
    weight_fingerprints: set[str] = set()
    behavior_hashes: set[str] = set()
    config_hashes: set[str] = set()
    model_references: set[str] = set()

    for mapping in _walk_mappings(value):
        descriptor = mapping.get("model_descriptor_sha256")
        if isinstance(descriptor, str) and descriptor:
            descriptor_hashes.add(descriptor)
        checkpoint = mapping.get("base_checkpoint")
        if isinstance(checkpoint, Mapping):
            fingerprint = checkpoint.get("fingerprint")
            if isinstance(fingerprint, str) and fingerprint:
                weight_fingerprints.add(fingerprint)
        checkpoint_fingerprint = mapping.get("checkpoint_fingerprint")
        if isinstance(checkpoint_fingerprint, str) and checkpoint_fingerprint:
            weight_fingerprints.add(checkpoint_fingerprint)
        weight_identity = mapping.get("weight_identity")
        if isinstance(weight_identity, Mapping):
            fingerprint = weight_identity.get("fingerprint")
            if isinstance(fingerprint, str) and fingerprint:
                weight_fingerprints.add(fingerprint)
        behavior = mapping.get("model_behavior_bundle")
        if isinstance(behavior, Mapping):
            bundle = behavior.get("bundle_sha256")
            if isinstance(bundle, str) and bundle:
                behavior_hashes.add(bundle)
        config_hash = mapping.get("model_config_sha256")
        if isinstance(config_hash, str) and config_hash:
            config_hashes.add(config_hash)
        for key in ("model", "model_path", "model_path_input", "base_model"):
            reference = mapping.get(key)
            if isinstance(reference, str) and reference.strip():
                model_references.add(reference.strip())

    expected_descriptor = str(candidate["descriptor_sha256"])
    expected_weight = str(candidate["weight_fingerprint"])
    expected_behavior = str(candidate["behavior_bundle_sha256"])
    expected_config = str(candidate["config_sha256"])
    expected_path = str(candidate["canonical_path"])

    if descriptor_hashes:
        if descriptor_hashes == {expected_descriptor}:
            status, strength = "match", "exact_descriptor"
        elif expected_descriptor in descriptor_hashes:
            status, strength = "ambiguous", "mixed_descriptors"
        else:
            status, strength = "mismatch", "exact_descriptor"
    elif weight_fingerprints and behavior_hashes:
        if (
            weight_fingerprints == {expected_weight}
            and behavior_hashes == {expected_behavior}
        ):
            status, strength = "match", "exact_weight_and_behavior_bundle"
        else:
            status, strength = "mismatch", "exact_weight_and_behavior_bundle"
    elif weight_fingerprints:
        status = "match_incomplete" if weight_fingerprints == {expected_weight} else "mismatch"
        strength = "weight_only"
    elif config_hashes:
        status = "match_incomplete" if expected_config and config_hashes == {expected_config} else "mismatch"
        strength = "config_only"
    elif model_references:
        normalized: set[str] = set()
        for reference in model_references:
            path = Path(reference).expanduser()
            if path.is_absolute():
                normalized.add(str(path.resolve(strict=False)))
            else:
                normalized.add(reference)
        candidates = {expected_path}
        if candidate.get("repository_id"):
            candidates.add(str(candidate["repository_id"]))
        status = "match_incomplete" if normalized <= candidates else "mismatch"
        strength = "path_or_repository_only"
    else:
        status, strength = "unbound", "none"

    return {
        "status": status,
        "strength": strength,
        "candidate_descriptor_sha256": expected_descriptor,
        "artifact_descriptor_sha256s": sorted(descriptor_hashes),
        "artifact_weight_fingerprints": sorted(weight_fingerprints),
        "artifact_behavior_bundle_sha256s": sorted(behavior_hashes),
        "artifact_config_sha256s": sorted(config_hashes),
        "artifact_model_references": sorted(model_references),
    }


def _verify_self_digest(value: Mapping[str, Any]) -> tuple[bool | None, str]:
    claimed = value.get("manifest_sha256")
    if claimed is None:
        return None, "manifest_has_no_self_digest"
    if not _is_sha256(claimed):
        return False, "manifest_digest_invalid"
    material = dict(value)
    material.pop("manifest_sha256", None)
    if _canonical_sha256(material) != claimed:
        return False, "manifest_digest_mismatch"
    return True, "manifest_digest_verified"


def _related_bundle(
    probe: TissueProbe,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not probe.related_patterns:
        return {"file_count": 0, "byte_count": 0, "declared_count": None, "count_matches": None}
    files: dict[str, Path] = {}
    for pattern in probe.related_patterns:
        for path in probe.path.parent.glob(pattern):
            if path.is_file() and not path.is_symlink():
                files[str(path.resolve(strict=False))] = path
    declared: Any = metadata.get(probe.declared_count_field) if probe.declared_count_field else None
    try:
        declared_count = int(declared) if declared is not None else None
    except (TypeError, ValueError):
        declared_count = None
    observed = len(files)
    return {
        "file_count": observed,
        "byte_count": sum(path.stat().st_size for path in files.values()),
        "declared_count": declared_count,
        "count_matches": declared_count == observed if declared_count is not None else None,
    }


def _linked_artifact(
    probe: TissueProbe,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not probe.linked_sha256_field:
        return {
            "status": "not_declared",
            "path": "",
            "sha256": "",
            "size_bytes": 0,
        }
    name = probe.linked_file_name or metadata.get(probe.linked_file_field)
    claimed = metadata.get(probe.linked_sha256_field)
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or not _is_sha256(claimed)
    ):
        return {
            "status": "declaration_invalid",
            "path": "",
            "sha256": str(claimed or ""),
            "size_bytes": 0,
        }
    path = probe.path.parent / name
    try:
        payload = _stable_file_bytes(path, maximum=MAX_REUSABLE_INPUT_BYTES)
    except TissueInventoryError as exc:
        return {
            "status": str(exc),
            "path": str(path.absolute()),
            "sha256": str(claimed),
            "size_bytes": 0,
        }
    observed = hashlib.sha256(payload).hexdigest()
    return {
        "status": "verified" if observed == claimed else "digest_mismatch",
        "path": str(path.absolute()),
        "sha256": observed,
        "size_bytes": len(payload),
    }


def _declared_field_inventory(
    probe: TissueProbe,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    declared = tuple(dict.fromkeys((*probe.portable_fields, *probe.basis_bound_fields)))
    if not declared:
        return {
            "row_count": 0,
            "portable": {},
            "basis_bound": {},
            "complete": None,
        }
    rows = metadata.get("examples")
    if not isinstance(rows, list) or len(rows) > MAX_DATASET_ROWS:
        return {
            "row_count": len(rows) if isinstance(rows, list) else 0,
            "portable": {field: 0 for field in probe.portable_fields},
            "basis_bound": {field: 0 for field in probe.basis_bound_fields},
            "complete": False,
        }
    counts = {
        field: sum(
            1 for row in rows if isinstance(row, Mapping) and field in row
        )
        for field in declared
    }
    return {
        "row_count": len(rows),
        "portable": {field: counts[field] for field in probe.portable_fields},
        "basis_bound": {field: counts[field] for field in probe.basis_bound_fields},
        "complete": bool(rows) and all(count == len(rows) for count in counts.values()),
    }


def classify_tissue_probe(
    probe: TissueProbe,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify one artifact without loading any tensor into a model."""

    validate_model_artifact_descriptor(dict(descriptor))
    candidate = _candidate_identity(descriptor)
    path = probe.path.expanduser().absolute()
    base: dict[str, Any] = {
        "schema": ENTRY_SCHEMA,
        "family": probe.family,
        "artifact_id": probe.artifact_id,
        "artifact_kind": probe.artifact_kind,
        "path": str(path),
        "basis_class": probe.basis_class,
        "qualification_scope": probe.qualification_scope,
        "portable_fields": list(probe.portable_fields),
        "basis_bound_fields": list(probe.basis_bound_fields),
        "required": probe.required,
        "exists": path.is_file(),
        "candidate_load_authorized": False,
        "linked_artifact": {
            "status": "not_declared",
            "path": "",
            "sha256": "",
            "size_bytes": 0,
        },
        "field_inventory": {
            "row_count": 0,
            "portable": {},
            "basis_bound": {},
            "complete": None,
        },
    }
    if not path.is_file():
        outcome = "refuse" if probe.required else "retire"
        return {
            **base,
            "outcome": outcome,
            "reason_codes": ["required_artifact_missing" if probe.required else "artifact_absent"],
            "integrity": {"status": "absent", "sha256": "", "size_bytes": 0},
            "identity": {
                "status": "unbound",
                "strength": "none",
                "candidate_descriptor_sha256": candidate["descriptor_sha256"],
            },
            "related_bundle": {
                "file_count": 0,
                "byte_count": 0,
                "declared_count": None,
                "count_matches": None,
            },
        }

    maximum = (
        MAX_REUSABLE_INPUT_BYTES
        if probe.basis_class == "architecture_independent"
        else MAX_METADATA_BYTES
    )
    try:
        payload = _stable_file_bytes(path, maximum=maximum)
    except TissueInventoryError as exc:
        return {
            **base,
            "outcome": "refuse",
            "reason_codes": [str(exc)],
            "integrity": {"status": "refused", "sha256": "", "size_bytes": path.stat().st_size},
            "identity": {
                "status": "unbound",
                "strength": "none",
                "candidate_descriptor_sha256": candidate["descriptor_sha256"],
            },
            "related_bundle": {
                "file_count": 0,
                "byte_count": 0,
                "declared_count": None,
                "count_matches": None,
            },
        }

    integrity = {
        "status": "content_hashed",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    if probe.basis_class == "architecture_independent":
        reasons = ["architecture_independent_input_verified"]
        linked = base["linked_artifact"]
        if probe.require_self_digest:
            try:
                metadata, _raw = _strict_json(path)
                self_digest_ok, self_digest_reason = _verify_self_digest(metadata)
                linked = _linked_artifact(probe, metadata)
            except TissueInventoryError as exc:
                return {
                    **base,
                    "outcome": "refuse",
                    "reason_codes": [str(exc)],
                    "integrity": {**integrity, "status": "metadata_refused"},
                    "identity": {
                        "status": "not_applicable",
                        "strength": "content_digest",
                        "candidate_descriptor_sha256": candidate["descriptor_sha256"],
                    },
                    "related_bundle": {
                        "file_count": 0,
                        "byte_count": 0,
                        "declared_count": None,
                        "count_matches": None,
                    },
                }
            reasons.append(self_digest_reason)
            reasons.append(f"linked_artifact_{linked['status']}")
            if self_digest_ok is not True or linked["status"] != "verified":
                return {
                    **base,
                    "outcome": "refuse",
                    "reason_codes": sorted(set(reasons)),
                    "integrity": {**integrity, "status": "portable_bundle_refused"},
                    "identity": {
                        "status": "not_applicable",
                        "strength": "content_digest",
                        "candidate_descriptor_sha256": candidate["descriptor_sha256"],
                    },
                    "related_bundle": {
                        "file_count": 0,
                        "byte_count": 0,
                        "declared_count": None,
                        "count_matches": None,
                    },
                    "linked_artifact": linked,
                }
        return {
            **base,
            "outcome": "qualified",
            "reason_codes": sorted(set(reasons)),
            "integrity": integrity,
            "identity": {
                "status": "not_applicable",
                "strength": "content_digest",
                "candidate_descriptor_sha256": candidate["descriptor_sha256"],
            },
            "related_bundle": {
                "file_count": 0,
                "byte_count": 0,
                "declared_count": None,
                "count_matches": None,
            },
            "linked_artifact": linked,
        }

    try:
        metadata, _raw = _strict_json(path)
        self_digest_ok, self_digest_reason = _verify_self_digest(metadata)
        identity = _artifact_identity(metadata, candidate)
        related = _related_bundle(probe, metadata)
        linked = _linked_artifact(probe, metadata)
        field_inventory = _declared_field_inventory(probe, metadata)
    except TissueInventoryError as exc:
        return {
            **base,
            "outcome": "refuse",
            "reason_codes": [str(exc)],
            "integrity": {**integrity, "status": "metadata_refused"},
            "identity": {
                "status": "unbound",
                "strength": "none",
                "candidate_descriptor_sha256": candidate["descriptor_sha256"],
            },
            "related_bundle": {
                "file_count": 0,
                "byte_count": 0,
                "declared_count": None,
                "count_matches": None,
            },
        }

    reasons = [self_digest_reason]
    if related["count_matches"] is False:
        reasons.append("related_bundle_count_mismatch")
    if linked["status"] != "not_declared":
        reasons.append(f"linked_artifact_{linked['status']}")
    if field_inventory["complete"] is False:
        reasons.append("declared_dataset_fields_incomplete")
    status = identity["status"]
    if status == "match" and self_digest_ok is not False and related["count_matches"] is not False:
        outcome = "qualified"
        candidate_load_authorized = True
        reasons.append("candidate_exact_identity_match")
    elif status == "mismatch":
        outcome = probe.mismatch_outcome
        candidate_load_authorized = False
        reasons.append("candidate_basis_mismatch")
    elif status == "ambiguous":
        outcome = "refuse"
        candidate_load_authorized = False
        reasons.append("mixed_model_identities")
    elif probe.basis_class == "mixed_tokenized_data" and probe.portable_fields:
        outcome = "retrain"
        candidate_load_authorized = False
        reasons.append("portable_source_requires_candidate_retokenization")
    else:
        outcome = "refuse"
        candidate_load_authorized = False
        reasons.append("exact_candidate_identity_absent")

    if (
        self_digest_ok is False
        or related["count_matches"] is False
        or linked["status"] not in {"not_declared", "verified"}
        or field_inventory["complete"] is False
    ):
        candidate_load_authorized = False
        if outcome == "qualified":
            outcome = "refuse"
    return {
        **base,
        "outcome": outcome,
        "reason_codes": sorted(set(reasons)),
        "integrity": integrity,
        "identity": identity,
        "related_bundle": related,
        "linked_artifact": linked,
        "field_inventory": field_inventory,
        "candidate_load_authorized": candidate_load_authorized,
    }


def _family_summary(family: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = [str(entry["outcome"]) for entry in entries]
    basis_entries = [
        entry
        for entry in entries
        if entry["basis_class"] != "architecture_independent"
    ]
    if any(outcome == "refuse" for outcome in outcomes):
        outcome = "refuse"
    elif any(outcome == "retrain" for outcome in outcomes):
        outcome = "retrain"
    elif basis_entries and all(entry["outcome"] == "retire" for entry in basis_entries):
        outcome = "retire"
    elif entries and all(outcome in {"qualified", "retire"} for outcome in outcomes):
        outcome = "qualified"
    else:
        outcome = "retire"
    runtime_entries = [entry for entry in basis_entries if entry["qualification_scope"] == "candidate_runtime"]
    return {
        "family": family,
        "outcome": outcome,
        "entry_count": len(entries),
        "outcome_counts": {name: outcomes.count(name) for name in sorted(OUTCOMES)},
        "candidate_runtime_loadable": bool(runtime_entries)
        and all(entry["candidate_load_authorized"] for entry in runtime_entries),
        "reusable_input_count": sum(
            1
            for entry in entries
            if entry["basis_class"] == "architecture_independent"
            and entry["outcome"] == "qualified"
        ),
    }


def build_tissue_migration_inventory(
    descriptor: dict[str, Any],
    probes: Sequence[TissueProbe],
    *,
    generated_at: float | None = None,
) -> dict[str, Any]:
    """Build a signed-content inventory from a bounded declared probe set."""

    validate_model_artifact_descriptor(descriptor)
    if not probes or len(probes) > MAX_PROBES:
        raise TissueInventoryError("tissue_probe_count_out_of_bounds")
    identities = [(probe.family, probe.artifact_id) for probe in probes]
    if len(set(identities)) != len(identities):
        raise TissueInventoryError("duplicate_tissue_probe_identity")
    entries = [classify_tissue_probe(probe, descriptor) for probe in probes]
    entries.sort(key=lambda entry: (entry["family"], entry["artifact_id"]))
    summaries = [
        _family_summary(family, [entry for entry in entries if entry["family"] == family])
        for family in FAMILIES
    ]
    material: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "generated_at": float(time.time() if generated_at is None else generated_at),
        "candidate": _candidate_identity(descriptor),
        "limits": {
            "max_probes": MAX_PROBES,
            "max_metadata_bytes": MAX_METADATA_BYTES,
            "max_reusable_input_bytes": MAX_REUSABLE_INPUT_BYTES,
            "model_loaded": False,
            "training_run": False,
        },
        "entries": entries,
        "families": summaries,
        "promotion_ready": all(
            summary["outcome"] in {"qualified", "retire"}
            for summary in summaries
            if summary["family"] != "expert_adapters"
        ),
    }
    material["inventory_sha256"] = _canonical_sha256(material)
    return material


def validate_tissue_migration_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate inventory structure and its canonical content digest."""

    if not isinstance(value, Mapping):
        raise TissueInventoryError("inventory_schema_invalid")
    required = {
        "schema",
        "generated_at",
        "candidate",
        "limits",
        "entries",
        "families",
        "promotion_ready",
        "inventory_sha256",
    }
    if set(value) != required or value.get("schema") != INVENTORY_SCHEMA:
        raise TissueInventoryError("inventory_schema_invalid")
    claimed = value.get("inventory_sha256")
    material = dict(value)
    material.pop("inventory_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(material):
        raise TissueInventoryError("inventory_digest_invalid")
    entries = value.get("entries")
    families = value.get("families")
    if not isinstance(entries, list) or not isinstance(families, list):
        raise TissueInventoryError("inventory_collections_invalid")
    if len(entries) > MAX_PROBES:
        raise TissueInventoryError("inventory_entry_count_out_of_bounds")
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or entry.get("schema") != ENTRY_SCHEMA
            or entry.get("outcome") not in OUTCOMES
            or entry.get("basis_class") not in BASIS_CLASSES
        ):
            raise TissueInventoryError("inventory_entry_invalid")
    if [item.get("family") for item in families if isinstance(item, Mapping)] != list(FAMILIES):
        raise TissueInventoryError("inventory_family_order_invalid")
    return dict(value)


def default_tissue_probes(
    *,
    repo_root: Path | str,
    state_root: Path | str,
) -> list[TissueProbe]:
    """Declare the bounded production-facing migration surface at CP912."""

    repo = Path(repo_root).expanduser().absolute()
    state = Path(state_root).expanduser().absolute()
    cp195 = repo / "artifacts/closeout/latent_cortex/resident_32b_v3_cp195/adapter"
    probes = [
        TissueProbe(
            "persona_crsm",
            "persona_training_corpus",
            "training_corpus",
            repo / "training/data/train.jsonl",
            "architecture_independent",
            qualification_scope="retraining_input",
        ),
        TissueProbe(
            "persona_crsm",
            "persona_validation_corpus",
            "validation_corpus",
            repo / "training/data/valid.jsonl",
            "architecture_independent",
            qualification_scope="retraining_input",
        ),
        TissueProbe(
            "persona_crsm",
            "persona_adapter_config",
            "lora_training_config",
            repo / "training/adapters/aura-personality/adapter_config.json",
            "checkpoint_basis",
            mismatch_outcome="retrain",
        ),
        TissueProbe(
            "caa_steering",
            "caa_extraction_recipe",
            "extraction_recipe",
            repo / "training/extract_steering_vectors.py",
            "architecture_independent",
            qualification_scope="rebuild_recipe",
        ),
        TissueProbe(
            "caa_steering",
            "caa_vector_bundle",
            "activation_vector_bundle",
            repo / "training/vectors/caa_steering_meta.json",
            "activation_basis",
            mismatch_outcome="retrain",
            related_patterns=("*.npz", "*.npy"),
            declared_count_field="total_vectors",
        ),
        TissueProbe(
            "recurrent_tissue",
            "recurrent_curriculum_recipe",
            "curriculum_recipe",
            repo / "core/learning/recurrence_curriculum.py",
            "architecture_independent",
            qualification_scope="rebuild_recipe",
        ),
        TissueProbe(
            "recurrent_tissue",
            "recurrent_training_recipe",
            "training_recipe",
            repo / "tools/recurrence_native_train_v2.py",
            "architecture_independent",
            qualification_scope="rebuild_recipe",
        ),
        TissueProbe(
            "recurrent_tissue",
            "recurrent_source_dataset",
            "prompt_text_and_tokenized_dataset",
            cp195 / "dataset_manifest.json",
            "mixed_tokenized_data",
            mismatch_outcome="retrain",
            qualification_scope="retraining_input",
            portable_fields=("prompt", "answer", "family", "depth", "seed"),
            basis_bound_fields=("prompt_tokens", "answer_tokens"),
        ),
        TissueProbe(
            "recurrent_tissue",
            "resident_recurrent_warm_start",
            "recurrent_adapter_authority",
            repo / "config/latent_cortex/resident_32b_recurrent_policy_warm_start.json",
            "checkpoint_basis",
            mismatch_outcome="retrain",
        ),
    ]
    for asset in (
        "neural_transition_tissue_v1",
        "mathematics_memory_tissue_v1",
        "systematic_neural_alu_v1",
    ):
        probes.append(
            TissueProbe(
                "recurrent_tissue",
                asset,
                "architecture_independent_auxiliary_tissue",
                repo / f"core/brain/llm/latent_cortex/assets/{asset}/manifest.json",
                "architecture_independent",
                qualification_scope="cross_model_auxiliary",
                require_self_digest=True,
                linked_file_field="weights_file",
                linked_sha256_field="weights_sha256",
            )
        )
    for artifact_id, relative in (
        ("fast_weight_runtime_recipe", "core/brain/llm/latent_cortex/fast_weights.py"),
        ("fast_weight_learning_recipe", "core/brain/llm/latent_cortex/fast_weight_learning.py"),
        ("durable_fast_weight_recipe", "core/learning/latent_adapter_distillation.py"),
    ):
        probes.append(
            TissueProbe(
                "fast_weight_adapters",
                artifact_id,
                "query_scoped_rebuild_recipe",
                repo / relative,
                "architecture_independent",
                qualification_scope="rebuild_recipe",
            )
        )
    durable_root = state / "data/latent_cortex/durable_adapters"
    durable_manifests: list[Path] = []
    if durable_root.exists():
        root_stat = durable_root.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise TissueInventoryError(f"durable_adapter_root_invalid:{durable_root}")
        durable_manifests = sorted(
            path
            for path in durable_root.glob("*/manifest.json")
            if path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
        )
        if len(durable_manifests) > MAX_DISCOVERED_ADAPTERS:
            raise TissueInventoryError("durable_adapter_count_out_of_bounds")
    if durable_manifests:
        for manifest in durable_manifests:
            probes.append(
                TissueProbe(
                    "fast_weight_adapters",
                    f"durable_fast_weight:{manifest.parent.name}",
                    "durable_fast_weight_adapter",
                    manifest,
                    "checkpoint_basis",
                    mismatch_outcome="retire",
                    linked_file_name="delta_weights.npz",
                    linked_sha256_field="delta_sha256",
                )
            )
    else:
        probes.append(
            TissueProbe(
                "fast_weight_adapters",
                "durable_fast_weight_artifacts",
                "durable_fast_weight_adapter_inventory",
                durable_root / "__no_adapter_manifest__.json",
                "checkpoint_basis",
                mismatch_outcome="retire",
            )
        )
    probes.append(
        TissueProbe(
            "expert_adapters",
            "expert_adapter_registry",
            "expert_adapter_registry",
            state / "data/adapters/library.json",
            "checkpoint_basis",
            mismatch_outcome="retire",
            qualification_scope="candidate_runtime",
        )
    )
    return probes


def inventory_as_json(value: Mapping[str, Any]) -> str:
    """Stable operator output for files or stdout."""

    validate_tissue_migration_inventory(value)
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


__all__ = [
    "BASIS_CLASSES",
    "ENTRY_SCHEMA",
    "FAMILIES",
    "INVENTORY_SCHEMA",
    "MAX_METADATA_BYTES",
    "MAX_DISCOVERED_ADAPTERS",
    "MAX_PROBES",
    "MAX_REUSABLE_INPUT_BYTES",
    "OUTCOMES",
    "TissueInventoryError",
    "TissueProbe",
    "build_tissue_migration_inventory",
    "classify_tissue_probe",
    "default_tissue_probes",
    "inventory_as_json",
    "load_candidate_descriptor",
    "validate_tissue_migration_inventory",
]
