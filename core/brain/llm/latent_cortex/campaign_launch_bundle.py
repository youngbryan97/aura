"""Immutable adapter snapshots and launch-bundle artifact contracts.

Campaign execution must not depend on a mutable training directory.  This
module copies only the artifacts transitively named by a supported scoped
adapter manifest, seals them behind a content-root certificate, and verifies
that certificate without loading model weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
    COMPLETION_SCHEMA_V1,
    IDENTITY_RECEIPT_SCHEMA_V2,
    MANIFEST_SCHEMA_V2,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    BINDING_ROLES as GRPO_BINDING_ROLES,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    COMPLETION_SCHEMA as GRPO_COMPLETION_SCHEMA,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    IDENTITY_RECEIPT_SCHEMA as GRPO_IDENTITY_RECEIPT_SCHEMA,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    MANIFEST_SCHEMA as GRPO_MANIFEST_SCHEMA,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    VERIFIED_IDENTITY_RECEIPT_SCHEMA as VERIFIED_GRPO_IDENTITY_RECEIPT_SCHEMA,
)
from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
    validate_verified_recurrent_grpo_adapter_identity_receipt,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
    IDENTITY_RECEIPT_SCHEMA as RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
    MANIFEST_SCHEMAS as RESIDENT_SFT_MANIFEST_SCHEMAS,
)
from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
    declared_bindings as resident_sft_declared_bindings,
)
from core.runtime.file_read_gateway import open_stable_readonly_binary, read_stable_bytes

ADAPTER_FREEZE_SCHEMA = "aura.latent_cortex.adapter_freeze.v1"
ADAPTER_FREEZE_FILE = "adapter_freeze.json"
PRELAUNCH_BUNDLE_SCHEMA = "aura.latent_cortex.prelaunch_bundle.v1"
PRELAUNCH_MANIFEST_FILE = "prelaunch_manifest.json"
LAUNCH_PACKET_SCHEMA = "aura.latent_cortex.launch_packet.v1"
LAUNCH_PACKET_FILE = "launch_packet.json"

_MANIFEST_FILE = "recurrence_adapter_manifest.json"
_COMPLETION_FILE = "training_completion.json"
_BINDING_ROLES = (
    "adapter",
    "adapter_alias",
    "loader_config",
    "training_receipt",
    "training_config",
    "dataset_manifest",
    "execution_spec",
)
_MAX_JSON_BYTES = 256 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 1 << 40
_MAX_ARTIFACTS = 128
_SUPPORTED_IDENTITY_RECEIPT_SCHEMAS = {
    IDENTITY_RECEIPT_SCHEMA_V2,
    GRPO_IDENTITY_RECEIPT_SCHEMA,
    VERIFIED_GRPO_IDENTITY_RECEIPT_SCHEMA,
    RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA,
}
RESIDENT_SFT_COMPLETION_SCHEMA = "aura.resident_recurrent_sft_adapter_package_completion.v1"


class CampaignLaunchBundleError(ValueError):
    """Stable fail-closed launch preparation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise CampaignLaunchBundleError(code)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any) -> str:
    try:
        return sha256_bytes(canonical_json_bytes(value))
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("launch_value_not_canonical")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json(raw: bytes, *, role: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{role}_duplicate_key")
            result[key] = value
        return result

    def parse_float(raw_value: str) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            _fail(f"{role}_number_invalid")
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_float=parse_float,
            parse_constant=lambda _value: _fail(f"{role}_number_invalid"),
        )
    except CampaignLaunchBundleError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, OverflowError, RecursionError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    return value


def _read_strict_json_payload(
    path: Path,
    *,
    role: str,
) -> tuple[dict[str, Any], bytes]:
    try:
        observed = path.lstat()
    except OSError:
        _fail(f"{role}_unavailable")
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
        or observed.st_mode & 0o022
    ):
        _fail(f"{role}_storage_invalid")
    try:
        raw = read_stable_bytes(path, max_bytes=_MAX_JSON_BYTES)
    except (OSError, ValueError):
        _fail(f"{role}_unavailable")
    return _strict_json(raw, role=role), raw


def read_strict_json(
    path: Path,
    *,
    role: str,
) -> dict[str, Any]:
    """Read one owner-protected JSON object without prescribing its layout."""
    value, _raw = _read_strict_json_payload(path, role=role)
    return value


def read_canonical_json(
    path: Path,
    *,
    role: str,
    trailing_newline: bool | None = True,
) -> dict[str, Any]:
    value, raw = _read_strict_json_payload(path, role=role)
    if trailing_newline is None:
        trailing_newline = value.get("schema") not in {
            *RESIDENT_SFT_MANIFEST_SCHEMAS,
            RESIDENT_SFT_COMPLETION_SCHEMA,
            "aura.resident_recurrent_sft_training_admission.v1",
        }
    expected = canonical_json_bytes(value) + (b"\n" if trailing_newline else b"")
    if raw != expected:
        _fail(f"{role}_noncanonical")
    return value


def _relative_path(value: Any, *, role: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{role}_path_invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _fail(f"{role}_path_invalid")
    return value


def _contained_file(root: Path, relative: str, *, role: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        try:
            observed = current.lstat()
        except OSError:
            _fail(f"{role}_unavailable")
        if stat.S_ISLNK(observed.st_mode):
            _fail(f"{role}_symlink_rejected")
        if current != candidate and (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_mode & 0o022
        ):
            _fail(f"{role}_directory_storage_invalid")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError:
        _fail(f"{role}_unavailable")
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        _fail(f"{role}_path_escape")
    return resolved


def _file_binding(path: Path, *, role: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        with open_stable_readonly_binary(path, max_bytes=_MAX_ARTIFACT_BYTES) as (
            handle,
            identity,
        ):
            observed = os.fstat(handle.fileno())
            if (
                observed.st_uid != os.geteuid()
                or observed.st_nlink != 1
                or observed.st_mode & 0o022
            ):
                _fail(f"{role}_storage_invalid")
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except CampaignLaunchBundleError:
        raise
    except (OSError, ValueError):
        _fail(f"{role}_unavailable")
    return {"sha256": digest.hexdigest(), "size_bytes": identity.size}


def _declared_paths(manifest: Mapping[str, Any]) -> list[str]:
    schema = manifest.get("schema")
    if schema == MANIFEST_SCHEMA_V2:
        binding_roles = _BINDING_ROLES
    elif schema == GRPO_MANIFEST_SCHEMA:
        binding_roles = GRPO_BINDING_ROLES
    elif schema in RESIDENT_SFT_MANIFEST_SCHEMAS:
        paths = [_MANIFEST_FILE, _COMPLETION_FILE]
        paths.extend(binding["path"] for _role, binding in resident_sft_declared_bindings(manifest))
        if len(paths) > _MAX_ARTIFACTS or len(set(paths)) != len(paths):
            _fail("adapter_artifact_set_invalid")
        return sorted(paths)
    else:
        _fail("adapter_manifest_schema_invalid")
    paths = [_MANIFEST_FILE, _COMPLETION_FILE]
    for role in binding_roles:
        binding = manifest.get(role)
        if not isinstance(binding, Mapping):
            _fail(f"adapter_{role}_binding_invalid")
        paths.append(_relative_path(binding.get("path"), role=f"adapter_{role}"))
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        _fail("adapter_sources_invalid")
    for source_role, source in sorted(sources.items()):
        if not isinstance(source_role, str) or not isinstance(source, Mapping):
            _fail("adapter_sources_invalid")
        paths.append(
            _relative_path(source.get("snapshot_path"), role=f"adapter_source_{source_role}")
        )
    if len(paths) > _MAX_ARTIFACTS or len(set(paths)) != len(paths):
        _fail("adapter_artifact_set_invalid")
    return sorted(paths)


def adapter_artifact_inventory(
    root: Path,
    *,
    reject_unplanned: bool,
) -> list[dict[str, Any]]:
    """Return the exact transitive scoped-adapter inventory under ``root``."""

    supplied_root = root.expanduser()
    if supplied_root.is_symlink():
        _fail("adapter_root_symlink_rejected")
    try:
        resolved_root = supplied_root.resolve(strict=True)
    except OSError:
        _fail("adapter_root_unavailable")
    root_metadata = resolved_root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or root_metadata.st_mode & 0o022
    ):
        _fail("adapter_root_invalid")
    manifest_path = _contained_file(resolved_root, _MANIFEST_FILE, role="adapter_manifest")
    manifest = read_canonical_json(
        manifest_path,
        role="adapter_manifest",
        trailing_newline=None,
    )
    schema = manifest.get("schema")
    if schema == MANIFEST_SCHEMA_V2:
        binding_roles = _BINDING_ROLES
    elif schema == GRPO_MANIFEST_SCHEMA:
        binding_roles = GRPO_BINDING_ROLES
    elif schema in RESIDENT_SFT_MANIFEST_SCHEMAS:
        binding_roles = ()
    else:
        _fail("adapter_manifest_schema_invalid")
    paths = _declared_paths(manifest)
    inventory: list[dict[str, Any]] = []
    for index, relative in enumerate(paths):
        path = _contained_file(resolved_root, relative, role=f"adapter_artifact_{index}")
        binding = _file_binding(path, role=f"adapter_artifact_{index}")
        inventory.append({"path": relative, **binding})

    by_path = {record["path"]: record for record in inventory}
    for role in binding_roles:
        declared = manifest[role]
        actual = by_path[declared["path"]]
        if (
            declared.get("sha256") != actual["sha256"]
            or declared.get("size_bytes") != actual["size_bytes"]
        ):
            _fail(f"adapter_{role}_binding_mismatch")
    if schema not in RESIDENT_SFT_MANIFEST_SCHEMAS:
        for source_role, source in manifest["sources"].items():
            actual = by_path[source["snapshot_path"]]
            if (
                source.get("sha256") != actual["sha256"]
                or source.get("size_bytes") != actual["size_bytes"]
            ):
                _fail(f"adapter_source_{source_role}_binding_mismatch")

    if schema in RESIDENT_SFT_MANIFEST_SCHEMAS:
        for role, declared in resident_sft_declared_bindings(manifest):
            actual = by_path[declared["path"]]
            if (
                declared.get("sha256") != actual["sha256"]
                or declared.get("size_bytes") != actual["size_bytes"]
            ):
                _fail(f"adapter_{role}_binding_mismatch")

    completion = read_canonical_json(
        _contained_file(resolved_root, _COMPLETION_FILE, role="training_completion"),
        role="training_completion",
        trailing_newline=None,
    )
    if schema == MANIFEST_SCHEMA_V2:
        valid_completion = (
            set(completion)
            == {
                "schema",
                "complete",
                "halt_reason",
                "step",
                "adapter_sha256",
                "receipt_sha256",
                "manifest_sha256",
            }
            and completion.get("schema") == COMPLETION_SCHEMA_V1
            and completion.get("complete") is True
            and completion.get("halt_reason") == "max_steps"
            and completion.get("manifest_sha256") == by_path[_MANIFEST_FILE]["sha256"]
            and completion.get("adapter_sha256") == manifest["adapter"]["sha256"]
            and completion.get("receipt_sha256") == manifest["training_receipt"]["sha256"]
        )
    elif schema == GRPO_MANIFEST_SCHEMA:
        valid_completion = (
            set(completion)
            == {
                "schema",
                "complete",
                "halt_reason",
                "step",
                "optimizer_updates",
                "adapter_sha256",
                "receipt_sha256",
                "protocol_sha256",
                "execution_spec_sha256",
                "manifest_sha256",
            }
            and completion.get("schema") == GRPO_COMPLETION_SCHEMA
            and completion.get("complete") is True
            and completion.get("halt_reason") == "max_steps"
            and type(completion.get("step")) is int
            and completion["step"] > 0
            and type(completion.get("optimizer_updates")) is int
            and 0 < completion["optimizer_updates"] <= completion["step"]
            and completion.get("manifest_sha256") == by_path[_MANIFEST_FILE]["sha256"]
            and completion.get("adapter_sha256") == manifest["adapter"]["sha256"]
            and completion.get("receipt_sha256") == manifest["training_receipt"]["sha256"]
            and completion.get("protocol_sha256") == manifest["protocol_sha256"]
            and completion.get("execution_spec_sha256") == manifest["execution_spec_sha256"]
        )
    else:
        adapter_binding = manifest.get("bindings", {}).get("adapter", {})
        checkpoint_binding = manifest.get("bindings", {}).get("checkpoint_complete", {})
        authority_binding = manifest.get("bindings", {}).get("authority", {})
        authority = read_canonical_json(
            _contained_file(
                resolved_root,
                authority_binding.get("path"),
                role="resident_sft_authority",
            ),
            role="resident_sft_authority",
            trailing_newline=False,
        )
        valid_completion = (
            set(completion)
            == {
                "schema",
                "complete",
                "halt_reason",
                "step",
                "adapter_sha256",
                "checkpoint_complete_sha256",
                "authority_sha256",
                "manifest_sha256",
            }
            and completion.get("schema") == RESIDENT_SFT_COMPLETION_SCHEMA
            and completion.get("complete") is True
            and completion.get("halt_reason") == "max_steps"
            and type(completion.get("step")) is int
            and completion["step"] > 0
            and completion.get("manifest_sha256") == by_path[_MANIFEST_FILE]["sha256"]
            and completion.get("adapter_sha256") == adapter_binding.get("sha256")
            and completion.get("checkpoint_complete_sha256") == checkpoint_binding.get("sha256")
            and completion.get("authority_sha256") == authority.get("authority_sha256")
        )
    if not valid_completion:
        _fail("training_completion_binding_invalid")

    if reject_unplanned:
        if root_metadata.st_mode & 0o222:
            _fail("frozen_adapter_root_writable")
        observed: set[str] = set()
        for path in resolved_root.rglob("*"):
            relative = path.relative_to(resolved_root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                _fail("frozen_adapter_symlink_rejected")
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_mode & 0o222:
                    _fail("frozen_adapter_file_writable")
                observed.add(relative)
            elif stat.S_ISDIR(metadata.st_mode):
                if metadata.st_mode & 0o222:
                    _fail("frozen_adapter_directory_writable")
            else:
                _fail("frozen_adapter_special_file_rejected")
        allowed = set(paths) | {ADAPTER_FREEZE_FILE}
        if observed != allowed:
            _fail("frozen_adapter_artifact_set_mismatch")
    return inventory


def inventory_root_sha256(inventory: list[dict[str, Any]]) -> str:
    return _sha256(
        {
            "schema": "aura.latent_cortex.adapter_artifact_inventory.v1",
            "artifacts": inventory,
        }
    )


def _validated_model_identity(model_identity: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "fingerprint",
        "files",
        "model_behavior_bundle_sha256",
        "runtime_bundle_sha256",
        "runtime_environment_identity_sha256",
        "personality_adapter_bundle_sha256",
        "effective_stack_sha256",
    }
    if set(model_identity) != required:
        _fail("adapter_freeze_model_identity_invalid")
    files = model_identity.get("files")
    personality = model_identity.get("personality_adapter_bundle_sha256")
    sha_roles = required - {"files", "personality_adapter_bundle_sha256"}
    if (
        type(files) is not int
        or files < 1
        or any(not _is_sha256(model_identity.get(role)) for role in sha_roles)
        or not (personality == "" or _is_sha256(personality))
    ):
        _fail("adapter_freeze_model_identity_invalid")
    return dict(model_identity)


def _valid_identity_receipt(
    identity_receipt: Any,
    *,
    adapter_id: Any,
) -> bool:
    if (
        not isinstance(identity_receipt, Mapping)
        or identity_receipt.get("schema") not in _SUPPORTED_IDENTITY_RECEIPT_SCHEMAS
        or identity_receipt.get("adapter_id") != adapter_id
        or identity_receipt.get("complete") is not True
    ):
        return False
    if identity_receipt.get("schema") != VERIFIED_GRPO_IDENTITY_RECEIPT_SCHEMA:
        return True
    try:
        validate_verified_recurrent_grpo_adapter_identity_receipt(identity_receipt)
    # not a failure: a receipt that will not validate has not validated, and False is
    # the refusing direction.
    except ValueError:
        return False
    return True


def _verify_frozen_grpo_identity(
    root: Path,
    receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> None:
    """Rebuild the embedded base identity from the immutable frozen artifacts."""

    from core.brain.llm.latent_cortex.recurrent_grpo_adapter_identity import (
        declared_bindings,
        strict_json_loads,
        tensor_metadata_from_safetensors,
        validate_recurrent_grpo_adapter_identity,
    )

    manifest_path = _contained_file(root, _MANIFEST_FILE, role="verified_manifest")
    manifest_bytes = read_stable_bytes(manifest_path, max_bytes=_MAX_JSON_BYTES)
    manifest = strict_json_loads(manifest_bytes, role="verified_frozen_manifest")
    artifacts: dict[str, bytes] = {}
    for role, binding in declared_bindings(manifest):
        relative = _relative_path(binding.get("path"), role=f"verified_{role}")
        artifacts[relative] = read_stable_bytes(
            _contained_file(root, relative, role=f"verified_{role}"),
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
    artifacts[_COMPLETION_FILE] = read_stable_bytes(
        _contained_file(root, _COMPLETION_FILE, role="verified_completion"),
        max_bytes=_MAX_JSON_BYTES,
    )
    adapter_binding = manifest.get("adapter")
    if not isinstance(adapter_binding, Mapping):
        _fail("adapter_freeze_verified_manifest_invalid")
    adapter_relative = _relative_path(adapter_binding.get("path"), role="verified_adapter")
    rebuilt = validate_recurrent_grpo_adapter_identity(
        manifest_bytes,
        adapter_id=receipt["adapter_id"],
        actual_base_checkpoint=manifest["base_checkpoint"],
        actual_model_behavior_bundle=manifest["model_behavior_bundle"],
        actual_personality_adapter=manifest["personality_adapter"],
        actual_runtime_environment=manifest["training_runtime"],
        artifacts=artifacts,
        tensor_metadata=tensor_metadata_from_safetensors(artifacts[adapter_relative]),
    )
    expected = (
        receipt.get("base_identity")
        if receipt.get("schema") == VERIFIED_GRPO_IDENTITY_RECEIPT_SCHEMA
        else receipt
    )
    if rebuilt != expected:
        _fail("adapter_freeze_verified_base_identity_mismatch")
    if (
        rebuilt.get("base_checkpoint_fingerprint") != model_identity.get("fingerprint")
        or rebuilt.get("model_behavior_bundle_sha256")
        != model_identity.get("model_behavior_bundle_sha256")
        or rebuilt.get("personality_adapter_bundle_sha256")
        != model_identity.get("personality_adapter_bundle_sha256")
        or rebuilt.get("training_runtime_identity_sha256")
        != model_identity.get("runtime_environment_identity_sha256")
    ):
        _fail("adapter_freeze_verified_model_identity_mismatch")


def _verify_frozen_resident_sft_identity(
    root: Path,
    receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> None:
    from core.brain.llm.latent_cortex.adapter_identity import (
        inspect_mlx_tensor_metadata,
    )
    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        runtime_environment_identity,
    )
    from core.brain.llm.latent_cortex.resident_recurrent_sft_adapter_identity import (
        validate_resident_recurrent_sft_adapter_identity,
    )

    manifest_path = _contained_file(root, _MANIFEST_FILE, role="verified_manifest")
    manifest_bytes = read_stable_bytes(manifest_path, max_bytes=_MAX_JSON_BYTES)
    manifest = _strict_json(manifest_bytes, role="verified_resident_sft_manifest")
    artifacts: dict[str, bytes] = {}
    for role, binding in resident_sft_declared_bindings(manifest):
        relative = _relative_path(binding["path"], role=f"verified_{role}")
        artifacts[relative] = read_stable_bytes(
            _contained_file(root, relative, role=f"verified_{role}"),
            max_bytes=_MAX_ARTIFACT_BYTES,
        )
    artifacts[_COMPLETION_FILE] = read_stable_bytes(
        _contained_file(root, _COMPLETION_FILE, role="verified_completion"),
        max_bytes=_MAX_JSON_BYTES,
    )
    adapter_binding = manifest["bindings"]["adapter"]
    adapter_path = _contained_file(root, adapter_binding["path"], role="verified_adapter")
    rebuilt = validate_resident_recurrent_sft_adapter_identity(
        manifest_bytes,
        adapter_id=receipt["adapter_id"],
        actual_base_checkpoint=manifest["base_checkpoint"],
        actual_model_behavior_bundle=manifest["model_behavior_bundle"],
        actual_personality_adapter=manifest["personality_adapter"],
        actual_runtime_environment=runtime_environment_identity(),
        artifacts=artifacts,
        tensor_metadata=inspect_mlx_tensor_metadata(adapter_path),
    )
    if rebuilt != receipt:
        _fail("adapter_freeze_verified_resident_sft_identity_mismatch")
    if (
        rebuilt.get("base_checkpoint_fingerprint") != model_identity.get("fingerprint")
        or rebuilt.get("model_behavior_bundle_sha256")
        != model_identity.get("model_behavior_bundle_sha256")
        or rebuilt.get("personality_adapter_bundle_sha256")
        != model_identity.get("personality_adapter_bundle_sha256")
        or rebuilt.get("evaluation_runtime_identity_sha256")
        != model_identity.get("runtime_environment_identity_sha256")
    ):
        _fail("adapter_freeze_verified_model_identity_mismatch")


def build_adapter_freeze_certificate(
    *,
    adapter_id: str,
    inventory: list[dict[str, Any]],
    identity_receipt: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    validator_identity: Mapping[str, str],
) -> dict[str, Any]:
    """Build a certificate after full scoped-adapter identity validation."""

    if (
        not isinstance(adapter_id, str)
        or not adapter_id
        or not _valid_identity_receipt(identity_receipt, adapter_id=adapter_id)
    ):
        _fail("adapter_identity_receipt_invalid")
    validated_model_identity = _validated_model_identity(model_identity)
    if not validator_identity or any(
        not isinstance(key, str) or not _is_sha256(value)
        for key, value in validator_identity.items()
    ):
        _fail("adapter_freeze_validator_identity_invalid")
    material = {
        "schema": ADAPTER_FREEZE_SCHEMA,
        "adapter_id": adapter_id,
        "content_root_sha256": inventory_root_sha256(inventory),
        "artifacts": inventory,
        "identity_receipt": dict(identity_receipt),
        "model_identity": validated_model_identity,
        "validator_identity": dict(sorted(validator_identity.items())),
    }
    return {**material, "certificate_sha256": _sha256(material)}


def verify_adapter_freeze(root: Path) -> dict[str, Any]:
    """Verify an immutable adapter snapshot and return its certificate."""

    supplied_root = root.expanduser()
    if supplied_root.is_symlink():
        _fail("adapter_root_symlink_rejected")
    resolved = supplied_root.resolve(strict=True)
    certificate = read_canonical_json(
        _contained_file(resolved, ADAPTER_FREEZE_FILE, role="adapter_freeze"),
        role="adapter_freeze",
    )
    required = {
        "schema",
        "adapter_id",
        "content_root_sha256",
        "artifacts",
        "identity_receipt",
        "model_identity",
        "validator_identity",
        "certificate_sha256",
    }
    material = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    if (
        set(certificate) != required
        or certificate.get("schema") != ADAPTER_FREEZE_SCHEMA
        or not _is_sha256(certificate.get("certificate_sha256"))
        or certificate.get("certificate_sha256") != _sha256(material)
    ):
        _fail("adapter_freeze_certificate_invalid")
    inventory = adapter_artifact_inventory(resolved, reject_unplanned=True)
    if certificate.get("artifacts") != inventory or certificate.get(
        "content_root_sha256"
    ) != inventory_root_sha256(inventory):
        _fail("adapter_freeze_content_mismatch")
    model_identity = certificate.get("model_identity")
    if not isinstance(model_identity, Mapping):
        _fail("adapter_freeze_model_identity_invalid")
    _validated_model_identity(model_identity)
    receipt = certificate.get("identity_receipt")
    if not _valid_identity_receipt(receipt, adapter_id=certificate.get("adapter_id")):
        _fail("adapter_freeze_identity_invalid")
    if receipt.get("schema") in {
        GRPO_IDENTITY_RECEIPT_SCHEMA,
        VERIFIED_GRPO_IDENTITY_RECEIPT_SCHEMA,
    }:
        _verify_frozen_grpo_identity(resolved, receipt, model_identity)
    elif receipt.get("schema") == RESIDENT_SFT_IDENTITY_RECEIPT_SCHEMA:
        _verify_frozen_resident_sft_identity(resolved, receipt, model_identity)
    return certificate


__all__ = [
    "ADAPTER_FREEZE_FILE",
    "ADAPTER_FREEZE_SCHEMA",
    "LAUNCH_PACKET_FILE",
    "LAUNCH_PACKET_SCHEMA",
    "PRELAUNCH_BUNDLE_SCHEMA",
    "PRELAUNCH_MANIFEST_FILE",
    "CampaignLaunchBundleError",
    "adapter_artifact_inventory",
    "build_adapter_freeze_certificate",
    "inventory_root_sha256",
    "read_canonical_json",
    "sha256_bytes",
    "verify_adapter_freeze",
]
