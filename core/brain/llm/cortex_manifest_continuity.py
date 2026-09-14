"""Compare retained model manifests against a consumer's declared dependencies."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.runtime.file_read_gateway import read_stable_bytes
from core.verify import invariant

MANIFEST_HISTORY_DIRECTORY = "manifest-history"
MAX_MANIFEST_BYTES = 512 * 1024


def manifest_snapshot_path(manifest: Path, sha256: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("manifest_snapshot_digest_invalid")
    return manifest.parent / MANIFEST_HISTORY_DIRECTORY / f"{sha256}.json"


def read_manifest_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not 1 < metadata.st_size <= MAX_MANIFEST_BYTES
    ):
        raise ValueError("manifest_snapshot_file_invalid")
    raw = read_stable_bytes(path, max_bytes=MAX_MANIFEST_BYTES)
    if len(raw) > MAX_MANIFEST_BYTES or not isinstance(json.loads(raw), dict):
        raise ValueError("manifest_snapshot_object_invalid")
    return raw


@invariant("cortex.manifest_dependency_continuity", scope="inference", owner="model_registry")
def verify_manifest_component_continuity(
    *,
    manifest: Path,
    predecessor_sha256: str,
    current_sha256: str,
    model_path: Path,
    independent_components: frozenset[str] = frozenset(),
    authority_key_path: Path | None = None,
) -> dict[str, Any]:
    """Prove that only explicitly independent, signed component bindings moved.

    This establishes runtime dependency continuity, not a new measurement of
    the model or a transfer of the predecessor's comparative results.
    """
    from core.brain.llm.model_registry import read_active_cortex_spec

    snapshot = manifest_snapshot_path(manifest, predecessor_sha256)
    if snapshot.parent.is_symlink():
        raise ValueError("manifest_snapshot_directory_invalid")
    previous_raw = read_manifest_bytes(snapshot)
    current_raw = read_manifest_bytes(manifest)
    if (
        hashlib.sha256(previous_raw).hexdigest() != predecessor_sha256
        or hashlib.sha256(current_raw).hexdigest() != current_sha256
    ):
        raise ValueError("manifest_continuity_digest_mismatch")

    previous = json.loads(previous_raw)
    current = json.loads(current_raw)
    spec = read_active_cortex_spec(str(manifest), authority_key_path=authority_key_path)
    if (
        spec is None
        or not spec.exact_identity
        or not spec.promotion_qualified
        or spec.pointer_sha256 != current_sha256
        or spec.model_path != model_path.resolve(strict=True)
    ):
        raise ValueError("manifest_continuity_current_authority_invalid")

    old_contract = previous.get("migration_contract")
    new_contract = current.get("migration_contract")
    if not isinstance(old_contract, dict) or not isinstance(new_contract, dict):
        raise ValueError("manifest_continuity_contract_missing")
    old_components = old_contract.get("components")
    new_components = new_contract.get("components")
    if (
        not isinstance(old_components, dict)
        or not isinstance(new_components, dict)
        or set(old_components) != set(new_components)
        or not independent_components <= set(new_components)
    ):
        raise ValueError("manifest_continuity_components_invalid")
    changed = sorted(
        name for name in old_components if old_components[name] != new_components[name]
    )
    if not changed or not set(changed) <= independent_components:
        raise ValueError("manifest_continuity_dependent_component_changed")

    # Ignore only the changed bindings and the containing contract's derived
    # digest/timestamp. Unknown fields, model bytes, profiles and evaluation
    # receipts must still compare equal, including their presence or absence.
    projected = deepcopy(current)
    projected_contract = projected["migration_contract"]
    for name in changed:
        projected_contract["components"][name] = old_components[name]
    for name in ("built_at", "migration_contract_sha256"):
        projected_contract[name] = old_contract.get(name)
    if projected != previous:
        raise ValueError("manifest_continuity_not_narrow")

    from core.brain.llm.model_artifact_profile import validate_model_artifact_files

    validate_model_artifact_files(current["artifact_descriptor"], model_path=model_path)

    return {
        "schema": "aura.cortex_manifest_component_continuity.v1",
        "predecessor_pointer_sha256": predecessor_sha256,
        "current_pointer_sha256": current_sha256,
        "model_descriptor_sha256": spec.descriptor_sha256,
        "changed_independent_components": changed,
        "current_authorities": {
            name: new_components[name]["authority_sha256"] for name in changed
        },
        "comparative_remeasurement": False,
    }
