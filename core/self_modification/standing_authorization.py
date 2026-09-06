"""A standing authorization to activate a cortex generation, and its limits.

``activate_upgrade`` requires an operator authorization string. That is one
human decision per swap, and the pipeline around it — evaluate, plan, stage,
rollback — is already automatic and receipted. This module is the second
option on that spectrum: the operator authorizes a FAMILY of candidates once,
in advance, with an expiry and a ceiling, and activations inside that scope
proceed without a fresh prompt.

What it deliberately does not do is remove the human decision. It moves it
earlier and makes it narrower, which is a different thing from removing it:

  * A grant names what it covers. A candidate outside the scope is refused
    exactly as if no grant existed.
  * A grant expires. One without an end is a permanent transfer of the
    decision, so ``expires_at`` is required and a grant is refused once past
    it.
  * A grant is spent. ``most_activations`` bounds how many times it can
    authorize, and each use is recorded against it, so a grant of one cannot
    authorize twice.
  * A grant is not evidence. The PASS verdict, the parity check inside it and
    every critical gate still have to hold; the grant replaces the operator's
    presence, not the measurement.

**What protects the grant is what protects the rest of this package.** The
mutation constitution seals ``self_modification`` from plastic modification,
and nothing on any autonomous path calls ``write_standing_grant`` — a test
holds that. There is no filesystem-level protection here and this docstring
does not claim one: an operator who runs Aura's own code as themselves can
write a grant, which is the same thing as saying they authorized it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.StandingAuthorization")

__all__ = [
    "GRANT_SCHEMA",
    "AStandingGrant",
    "read_standing_grant",
    "where_a_grant_is_kept",
    "why_a_grant_does_not_cover",
    "write_standing_grant",
]

GRANT_SCHEMA = "aura.cortex_upgrade.standing_grant.v1"

#: The longest a grant may run. A standing authorization is an operator
#: deciding in advance, and deciding in advance about a year from now is not
#: deciding, it is delegating. Ninety days is long enough to cover a planned
#: upgrade window and short enough that a forgotten grant lapses.
LONGEST_GRANT_S = 90 * 24 * 3600


@dataclass(frozen=True)
class AStandingGrant:
    """One operator decision, made early, about a named family of candidates."""

    granted_by: str
    #: Model paths this covers, matched by prefix. An empty prefix would
    #: cover everything, which is not a scope, so it is refused at write.
    model_path_prefixes: tuple[str, ...] = ()
    #: Exact descriptor digests this covers. Narrower than a prefix and
    #: usable when the operator already knows the candidate.
    descriptor_digests: tuple[str, ...] = ()
    expires_at: float = 0.0
    most_activations: int = 1
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRANT_SCHEMA,
            "granted_by": self.granted_by,
            "model_path_prefixes": list(self.model_path_prefixes),
            "descriptor_digests": list(self.descriptor_digests),
            "expires_at": self.expires_at,
            "most_activations": self.most_activations,
            "reason": self.reason,
            "created_at": self.created_at,
            "used": self.used,
        }

    def digest(self) -> str:
        body = dict(self.to_dict())
        body.pop("used", None)
        return hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest()


def where_a_grant_is_kept(fused_model_dir: Path | str) -> Path:
    return Path(fused_model_dir) / "standing_activation_grant.json"


def why_a_grant_does_not_cover(
    grant: AStandingGrant | None,
    *,
    model_path: str,
    descriptor_digest: str,
    now: float | None = None,
) -> str:
    """Empty when this grant authorizes this candidate right now.

    Every refusal names its own condition. "Not authorized" tells an operator
    to look at everything; "expired 3 days ago" tells them what to do.
    """
    if grant is None:
        return "there is no standing grant"
    now = time.time() if now is None else float(now)
    if not str(grant.granted_by or "").strip():
        return "the grant names nobody as having granted it"
    if grant.expires_at <= 0:
        return "the grant has no expiry, so it is not a standing grant"
    if now >= grant.expires_at:
        return (
            f"the grant expired {(now - grant.expires_at) / 86400:.1f} days ago"
        )
    if grant.used >= grant.most_activations:
        return (
            f"the grant is spent: {grant.used} of {grant.most_activations} "
            "activations already used"
        )
    if not grant.model_path_prefixes and not grant.descriptor_digests:
        return "the grant names no candidates, so it covers none"
    by_digest = descriptor_digest and descriptor_digest in grant.descriptor_digests
    by_path = any(
        prefix and str(model_path).startswith(prefix)
        for prefix in grant.model_path_prefixes
    )
    if not (by_digest or by_path):
        return (
            f"the grant does not cover {model_path!r}: it covers "
            f"{list(grant.model_path_prefixes)} and "
            f"{len(grant.descriptor_digests)} digest(s)"
        )
    return ""


def write_standing_grant(
    fused_model_dir: Path | str,
    *,
    granted_by: str,
    model_path_prefixes: tuple[str, ...] = (),
    descriptor_digests: tuple[str, ...] = (),
    valid_for_s: float,
    most_activations: int = 1,
    reason: str = "",
) -> AStandingGrant:
    """Record an operator's decision made in advance. Never called autonomously.

    Refuses the shapes that would make a grant something other than a scoped,
    bounded, expiring decision — an unscoped grant, an unbounded one, or one
    that outlives the window it was written for.
    """
    who = str(granted_by or "").strip()
    if len(who) < 3:
        raise PermissionError("a standing grant requires a real operator identity")
    prefixes = tuple(one for one in model_path_prefixes if str(one).strip())
    digests = tuple(one for one in descriptor_digests if str(one).strip())
    if not prefixes and not digests:
        raise ValueError(
            "a standing grant must name what it covers; one covering everything "
            "is not a scope"
        )
    if valid_for_s <= 0 or valid_for_s > LONGEST_GRANT_S:
        raise ValueError(
            f"a standing grant runs for more than 0 and at most "
            f"{LONGEST_GRANT_S / 86400:.0f} days"
        )
    if most_activations < 1:
        raise ValueError("a standing grant authorizes at least one activation")

    grant = AStandingGrant(
        granted_by=who,
        model_path_prefixes=prefixes,
        descriptor_digests=digests,
        expires_at=time.time() + float(valid_for_s),
        most_activations=int(most_activations),
        reason=str(reason)[:400],
    )
    _keep(fused_model_dir, grant)
    logger.info(
        "standing activation grant written by %s for %s, %d activation(s), "
        "expiring in %.1f days",
        who, list(prefixes) or f"{len(digests)} digest(s)",
        grant.most_activations, valid_for_s / 86400,
    )
    return grant


def read_standing_grant(fused_model_dir: Path | str) -> AStandingGrant | None:
    """The grant on disk, or None. A malformed grant is None, not a crash."""
    path = where_a_grant_is_kept(fused_model_dir)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("schema") != GRANT_SCHEMA:
        return None
    try:
        return AStandingGrant(
            granted_by=str(body.get("granted_by") or ""),
            model_path_prefixes=tuple(body.get("model_path_prefixes") or ()),
            descriptor_digests=tuple(body.get("descriptor_digests") or ()),
            expires_at=float(body.get("expires_at") or 0.0),
            most_activations=int(body.get("most_activations") or 0),
            reason=str(body.get("reason") or ""),
            created_at=float(body.get("created_at") or 0.0),
            used=int(body.get("used") or 0),
        )
    except (TypeError, ValueError):
        return None


def spend_one_activation(fused_model_dir: Path | str, grant: AStandingGrant) -> AStandingGrant:
    """Record that this grant authorized an activation, and persist it.

    A ceiling nobody decrements is not a ceiling.
    """
    from dataclasses import replace

    spent = replace(grant, used=grant.used + 1)
    _keep(fused_model_dir, spent)
    return spent


def _keep(fused_model_dir: Path | str, grant: AStandingGrant) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    path = where_a_grant_is_kept(fused_model_dir)
    gateway = get_file_write_gateway()
    with local_internal_governed_scope(
        "cortex_upgrade.standing_grant", domain="state_mutation"
    ):
        gateway.ensure_directory(path.parent, source="standing_authorization")
        gateway.write_text(
            path,
            json.dumps(grant.to_dict(), indent=2, sort_keys=True) + "\n",
            source="standing_authorization",
        )
