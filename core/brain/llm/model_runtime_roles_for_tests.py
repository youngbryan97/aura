"""Issue a serving-role assignment for a model path, for tests and harnesses.

CP941 replaced the path-token heuristic — "32b" means cortex, "72b" means
solver — with a role the registry assigns against artifact identity. That was
the right change: a serving role is an assignment, not a deduction from a
directory name. It also means a test or harness pointing at a synthetic path
gets ``auxiliary``/``best_effort``, which is correct and is almost never the
scenario being exercised.

Saying the role out loud is what CP941 asked callers to do. This is the one
place that spells the incantation, so a harness needs one line rather than
seven, and so a future change to the assignment contract has one place to
land instead of a dozen copies.

Not for production: a real client resolves its assignment from the registry,
which is bound to a measured artifact. This issues one from a locator hash and
says so in its authority_source.
"""

from __future__ import annotations

import hashlib

from core.runtime.model_runtime_assignment import ModelRuntimeAssignment

#: Stamped into every assignment issued here so an authority audit can tell
#: these apart from a registry decision at a glance.
AUTHORITY_SOURCE = "declared_role_for_tests"


def assignment_for(model_path: str, *, role: str, purpose: str = "serve"):
    """A runtime assignment binding ``model_path`` to a declared serving role."""
    return ModelRuntimeAssignment.issue(
        model_path=model_path,
        artifact_identity=hashlib.sha256(
            str(model_path).encode("utf-8")
        ).hexdigest(),
        artifact_identity_kind="canonical_locator_sha256",
        artifact_identity_exact=False,
        role=role,
        purpose=purpose,
        authority_source=AUTHORITY_SOURCE,
    )


__all__ = ["AUTHORITY_SOURCE", "assignment_for"]
