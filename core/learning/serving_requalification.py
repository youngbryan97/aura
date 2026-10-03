"""A requalified serving window reaches the active cortex without posing as a swap.

The serving profile says how long a prompt the resident cortex may read and how
long an answer it may write, and it carries the measurement that qualified
those numbers for one exact artifact. Activation (core/learning/
cortex_generation_upgrade.py) installs a profile only as part of a model swap,
and a swap needs a fresh capability comparison of the two models. On
2026-10-03 the active cortex was requalified at 65536 tokens (131072 failed
recall) and activation refused, because the active pointer's evaluation
predates the current evaluation schema. There was no second model to compare:
the artifact was the same, and only its measured window had changed.

So this replaces the serving profile of the active pointer, and nothing else:

* the profile is validated against the active pointer's own descriptor, so a
  profile measured on any other artifact is refused;
* an operator at the change or a standing grant authorizes it, by the same
  rule activation uses;
* the pointer it replaces becomes the rollback pointer, and
  ``rollback_upgrade`` restores it byte for byte;
* the new pointer is read back through the runtime's own validator before the
  change is reported; one that does not load is replaced by the old bytes.

Effective at next boot, like activation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

__all__ = ["apply_serving_requalification"]


def apply_serving_requalification(
    *,
    serving_profile: dict[str, Any],
    authorized_by: str,
    fused_model_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Replace the active cortex's serving profile with one requalified for the same artifact."""
    from core.brain.llm.model_artifact_profile import validate_model_serving_profile
    from core.brain.llm.model_registry import read_active_cortex_spec
    from core.learning.cortex_generation_upgrade import (
        ROLLBACK_POINTER_NAME,
        _governed_write,
        _read_pointer,
        _who_authorized_this,
    )

    if fused_model_dir is None:
        from core.brain.llm.model_registry import get_fused_model_root

        fused_model_dir = get_fused_model_root()
    fused_model_dir = Path(fused_model_dir)
    pointer_path = fused_model_dir / "active.json"
    current_bytes = pointer_path.read_bytes()
    current = _read_pointer(pointer_path)
    descriptor = current.get("artifact_descriptor")
    if current.get("schema_version") != 3 or not isinstance(descriptor, dict):
        raise ValueError("the active pointer names no artifact descriptor to bind a profile to")
    validate_model_serving_profile(serving_profile, descriptor)
    previous = current.get("serving_profile") if isinstance(current.get("serving_profile"), dict) else {}
    if previous.get("profile_sha256") == serving_profile.get("profile_sha256"):
        return {"changed": False, "serving_profile_sha256": serving_profile.get("profile_sha256")}

    authorization = _who_authorized_this(
        fused_model_dir,
        authorized_by=authorized_by,
        model_path=str(current["active_model_path"]),
        descriptor_digest=str(descriptor["descriptor_sha256"]),
    )
    replaced = {**current, "serving_profile": serving_profile}
    replaced_bytes = (json.dumps(replaced, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _governed_write(fused_model_dir / ROLLBACK_POINTER_NAME, current_bytes, source="cortex_upgrade.requalify")
    _governed_write(pointer_path, replaced_bytes, source="cortex_upgrade.requalify")
    try:
        read_active_cortex_spec(pointer_path)
    except (OSError, ValueError, RuntimeError) as exc:
        _governed_write(pointer_path, current_bytes, source="cortex_upgrade.requalify_undo")
        raise ValueError(f"the requalified pointer did not load, so the old one was put back: {exc}") from exc
    return {
        "schema": "aura.cortex_upgrade.serving_requalification.v1",
        "changed": True,
        "active_model": current["active_model_path"],
        "model_descriptor_sha256": descriptor["descriptor_sha256"],
        "previous_serving_profile_sha256": previous.get("profile_sha256"),
        "serving_profile_sha256": serving_profile["profile_sha256"],
        "served_context_tokens": serving_profile.get("served_context_tokens"),
        "active_sha256": hashlib.sha256(replaced_bytes).hexdigest(),
        "rollback_sha256": hashlib.sha256(current_bytes).hexdigest(),
        "authorization": authorization,
        "authorized_by": str(authorized_by or "").strip(),
        "effective": "next_boot",
        "requalified_at": time.time(),
    }
