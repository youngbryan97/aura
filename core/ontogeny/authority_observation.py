"""Durable observation support for deciding ontogeny heads."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.Ontogeny")

# Persisted outcomes, rather than process-local counters, decide whether a
# learned head still has enough observation support to retain authority.
AUTHORITY_OBSERVATION_WINDOW = 500
AUTHORITY_OBSERVATION_MIN_CLOSED = 50
AUTHORITY_OBSERVATION_FLOOR = 0.05


class AuthorityObservationMixin:
    """Observation reporting and revocation for an ontogeny core."""

    def authority_observation_report(
        self,
        *,
        recent_limit: int = AUTHORITY_OBSERVATION_WINDOW,
        min_closed: int = AUTHORITY_OBSERVATION_MIN_CLOSED,
    ) -> dict[str, Any]:
        """Return durable observation support for every deciding head."""

        minimum_support = max(1, int(min_closed))
        control_points: dict[str, dict[str, Any]] = {}
        for control_point in self.control_points():
            if not self._authority.has_authority(control_point):
                continue
            with self._lock:
                schema_id = self._control_points[control_point].schema.schema_id
            stats = self._spine.observation_stats(
                control_point,
                recent_limit=recent_limit,
                feature_schema=schema_id,
            )
            eligible = bool(
                stats.get("available")
                and int(stats.get("closed") or 0) >= minimum_support
                and stats.get("observation_rate") is not None
            )
            control_points[control_point] = {**stats, "eligible": eligible}

        eligible_rates = [
            float(stats["observation_rate"])
            for stats in control_points.values()
            if stats.get("eligible")
        ]
        return {
            "schema": "aura.ontogeny.authority_observation.v1",
            "window_limit": max(1, min(5_000, int(recent_limit))),
            "minimum_closed": minimum_support,
            "control_points": control_points,
            "authoritative_control_points": len(control_points),
            "eligible_control_points": len(eligible_rates),
            "minimum_rate": min(eligible_rates) if eligible_rates else None,
        }

    def _enforce_authority_observation(self) -> tuple[str, ...]:
        """Revoke a deciding head whose durable outcome observer collapsed."""

        report = self.authority_observation_report()
        revoked: list[str] = []
        for control_point, stats in report["control_points"].items():
            if not stats.get("eligible"):
                continue
            rate = float(stats["observation_rate"])
            if rate > AUTHORITY_OBSERVATION_FLOOR:
                continue
            self._authority.revoke(
                control_point,
                (
                    f"durable observation rate {rate:.3f} at or below "
                    f"{AUTHORITY_OBSERVATION_FLOOR:.3f} across "
                    f"{int(stats['closed'])} recent closed episodes"
                ),
            )
            revoked.append(control_point)
            logger.warning(
                "ontogeny: revoked %s after durable observation collapsed to %.1f%%",
                control_point,
                rate * 100.0,
            )
        return tuple(revoked)


__all__ = [
    "AUTHORITY_OBSERVATION_FLOOR",
    "AUTHORITY_OBSERVATION_MIN_CLOSED",
    "AUTHORITY_OBSERVATION_WINDOW",
    "AuthorityObservationMixin",
]
