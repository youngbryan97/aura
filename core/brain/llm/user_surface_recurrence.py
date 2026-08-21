"""One admission policy for recurrent work on a live user surface.

The cognitive caller and the MLX worker used to resolve this contract
independently. The caller could promise two loops while the worker's incident
ceiling admitted one, making a successful control receipt impossible before a
token was generated. This module owns both the requested default and the
authorized ceiling so every process computes the same admitted value.
"""

from __future__ import annotations

from typing import Any

from core.runtime.flags import (
    user_surface_recurrent_loops_override,
    user_surface_recurrent_max_loops_override,
)

_IDENTITY_DEPTH = 1


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def user_surface_recurrent_ceiling() -> int:
    """Return the operationally authorized live-surface depth."""

    return max(
        _IDENTITY_DEPTH,
        _integer(user_surface_recurrent_max_loops_override(), _IDENTITY_DEPTH),
    )


def admit_user_surface_recurrent_loops(requested: Any = None) -> int:
    """Clamp a caller's request to the same bounded policy the worker enforces."""

    configured = (
        user_surface_recurrent_loops_override() if requested is None else requested
    )
    return max(
        _IDENTITY_DEPTH,
        min(
            _integer(configured, _IDENTITY_DEPTH),
            user_surface_recurrent_ceiling(),
        ),
    )


__all__ = [
    "admit_user_surface_recurrent_loops",
    "user_surface_recurrent_ceiling",
]
