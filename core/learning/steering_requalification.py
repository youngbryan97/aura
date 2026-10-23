"""A new steering generation reaches the active cortex without posing as a swap.

Her steering directions are one signed component of the active cortex's
migration contract. Activation installs a contract only with a model swap,
which needs a capability comparison of two models; a new generation of
directions for the same checkpoint has no second model to compare. On 23
October directions fitted to her own replies were qualified for her 27B and
the generation in the contract was one whose directions cost her voice.

So this replaces the steering component of the active pointer's contract, and
nothing else, by the rules the serving requalification keeps
(core/learning/serving_requalification.py):

* the contract is rebuilt by the activation's own builder, which revalidates
  every component's signature and evidence, the new one included;
* an operator at the change or a standing grant authorizes it;
* the pointer it replaces becomes the rollback pointer;
* the new pointer must load and its generation must open, or the old bytes go
  back;
* a qualified serving surface that would switch off puts the old bytes back.

Effective at next boot.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

__all__ = ["apply_steering_requalification"]


def apply_steering_requalification(
    *,
    authority: dict[str, Any],
    authorized_by: str,
    fused_model_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Replace the active cortex's steering authority with one issued for the same artifact."""
    from core.brain.llm.model_bound_steering import materialize_qualified_generation
    from core.brain.llm.model_registry import read_active_cortex_spec
    from core.learning.cortex_generation_upgrade import (
        ROLLBACK_POINTER_NAME,
        _governed_write,
        _read_pointer,
        _who_authorized_this,
        build_migration_contract,
    )
    from core.learning.serving_requalification import _valid_serving_surfaces

    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    fused_model_dir = Path(fused_model_dir)
    pointer_path = fused_model_dir / "active.json"
    current_bytes = pointer_path.read_bytes()
    current = _read_pointer(pointer_path)
    descriptor = current.get("artifact_descriptor")
    contract = current.get("migration_contract")
    components = contract.get("components") if isinstance(contract, dict) else None
    if current.get("schema_version") != 3 or not isinstance(descriptor, dict) or not isinstance(components, dict):
        raise ValueError("the active pointer names no migration contract to replace a component of")
    previous = components.get("steering") if isinstance(components.get("steering"), dict) else {}
    if previous.get("authority_sha256") == authority.get("authority_sha256"):
        return {"changed": False, "steering_authority_sha256": authority.get("authority_sha256")}
    rebuilt = build_migration_contract(descriptor, components={**components, "steering": authority})

    authorization = _who_authorized_this(
        fused_model_dir,
        authorized_by=authorized_by,
        model_path=str(current["active_model_path"]),
        descriptor_digest=str(descriptor["descriptor_sha256"]),
    )
    surfaces_before = _valid_serving_surfaces()
    replaced = {**current, "migration_contract": rebuilt}
    replaced_bytes = (json.dumps(replaced, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _governed_write(fused_model_dir / ROLLBACK_POINTER_NAME, current_bytes, source="cortex_upgrade.requalify_steering")
    _governed_write(pointer_path, replaced_bytes, source="cortex_upgrade.requalify_steering")
    try:
        read_active_cortex_spec(pointer_path)
        with tempfile.TemporaryDirectory() as opened:
            vectors = sorted(
                path.name
                for path in materialize_qualified_generation(
                    rebuilt["components"]["steering"],
                    descriptor_sha256=str(descriptor["descriptor_sha256"]),
                    model_cache_root=Path(opened),
                ).glob("*_layer*.np*")
            )
    except (OSError, ValueError, RuntimeError) as exc:
        _governed_write(pointer_path, current_bytes, source="cortex_upgrade.requalify_steering_undo")
        raise ValueError(f"the requalified pointer or its generation did not open, so the old one was put back: {exc}") from exc
    switched_off = sorted(surfaces_before - _valid_serving_surfaces())
    if switched_off:
        _governed_write(pointer_path, current_bytes, source="cortex_upgrade.requalify_steering_undo")
        raise ValueError(
            "the new steering generation would switch off a qualified serving surface, so the old "
            f"pointer was put back: {switched_off}"
        )
    return {
        "schema": "aura.cortex_upgrade.steering_requalification.v1",
        "changed": True,
        "active_model": current["active_model_path"],
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "previous_steering_authority_sha256": previous.get("authority_sha256"),
        "steering_authority_sha256": authority.get("authority_sha256"),
        "generation_sha256": (authority.get("claims") or {}).get("generation_sha256"),
        "vectors": len(vectors),
        "migration_contract_sha256": rebuilt["migration_contract_sha256"],
        "active_sha256": hashlib.sha256(replaced_bytes).hexdigest(),
        "rollback_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "authorization": authorization,
        "authorized_by": str(authorized_by or "").strip(),
        "effective": "next_boot",
        "requalified_at": time.time(),
    }
