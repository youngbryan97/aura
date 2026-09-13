"""One admission policy for recurrent work on a live user surface.

The cognitive caller and the MLX worker used to resolve this contract
independently. The caller could promise two loops while the worker's incident
ceiling admitted one, making a successful control receipt impossible before a
token was generated. This module owns both the requested default and the
authorized ceiling so every process computes the same admitted value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.runtime.flags import (
    user_surface_recurrent_loops_override,
    user_surface_recurrent_max_loops_override,
)

_IDENTITY_DEPTH = 1

#: Where the depth gate leaves what it authorized, one file per checkpoint. A
#: depth that helped one checkpoint says nothing about the next one, so the
#: receipt is named by the descriptor it was measured on and a checkpoint with
#: no receipt runs at the identity depth.
_RECEIPTS = Path(__file__).resolve().parents[3] / "artifacts/recurrent_depth/authorized"


def authorized_depth_for(descriptor_sha256: str) -> int:
    """The depth a gate authorized for this exact checkpoint, or one.

    The ceiling used to be reachable only through
    AURA_USER_SURFACE_RECURRENT_MAX_LOOPS, which is a person authorising an
    experiment rather than evidence authorising a default — and the comment
    that pinned it at one said it would stay there "until an accuracy gate says
    otherwise". This is where that gate's answer is read.

    Everything here fails closed. No descriptor, no receipt, a receipt that
    names a different checkpoint, a receipt that will not parse, a receipt
    carrying refusals: all of them are the identity depth.
    """
    descriptor = str(descriptor_sha256 or "").strip()
    if not descriptor:
        return _IDENTITY_DEPTH
    receipt = _RECEIPTS / f"{descriptor}.json"
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _IDENTITY_DEPTH
    if not isinstance(payload, dict):
        return _IDENTITY_DEPTH
    if str(payload.get("model_descriptor_sha256") or "") != descriptor:
        return _IDENTITY_DEPTH
    if payload.get("refusals"):
        return _IDENTITY_DEPTH
    return _integer(payload.get("authorized_depth"), _IDENTITY_DEPTH)


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def user_surface_recurrent_ceiling(descriptor_sha256: str = "") -> int:
    """Return the operationally authorized live-surface depth.

    Two ways up from the identity depth, and they are different things. The
    flag is a person asking for an experiment. The receipt is a gate reporting
    a measured improvement on the checkpoint that is actually loaded, which is
    the one that can be a default.
    """

    return max(
        _IDENTITY_DEPTH,
        _integer(user_surface_recurrent_max_loops_override(), _IDENTITY_DEPTH),
        authorized_depth_for(descriptor_sha256),
    )


def admit_user_surface_recurrent_loops(
    requested: Any = None, *, descriptor_sha256: str = ""
) -> int:
    """Clamp a caller's request to the same bounded policy the worker enforces.

    The descriptor is what lets a gate's receipt reach this decision. Without
    one the ceiling is whatever the flag says, which is the identity depth
    unless somebody set it by hand.
    """

    configured = (
        user_surface_recurrent_loops_override() if requested is None else requested
    )
    return max(
        _IDENTITY_DEPTH,
        min(
            _integer(configured, _IDENTITY_DEPTH),
            user_surface_recurrent_ceiling(descriptor_sha256),
        ),
    )


__all__ = [
    "admit_user_surface_recurrent_loops",
    "authorized_depth_for",
    "user_surface_recurrent_ceiling",
]
