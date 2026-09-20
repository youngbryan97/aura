"""The vault snapshot and the fields read out of it.

Lifted whole out of `mycelium`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import math
import time
from typing import Any


class _ReadsTheVault:
    """Lifted whole out of MycelialNetwork; see mycelium.py."""

    def _vault_snapshot_locked(self) -> dict[str, Any]:
        from .mycelium import (
            _VAULT_CLOCK_SKEW_TOLERANCE_S,
            NeuralRoot,
        )

        now_monotonic = time.monotonic()
        captured_at_unix = time.time()
        pathways: dict[str, dict[str, Any]] = {}
        for key, pathway in self.pathways.items():
            data = pathway.to_dict()
            data.pop("id", None)
            created_at = self._vault_number(
                pathway.created_at,
                f"live pathway creation timestamp: {key}",
                minimum=0.0,
                maximum=captured_at_unix + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            last_matched = self._vault_number(
                pathway.last_matched,
                f"live pathway last-matched timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            created_age = max(0.0, captured_at_unix - created_at)
            last_matched_age = max(0.0, now_monotonic - last_matched)
            if last_matched_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(
                    f"live pathway last match predates creation: {key}"
                )
            data["created_at"] = created_at
            data["last_matched_age_s"] = last_matched_age
            data.pop("last_matched", None)
            pathways[key] = data

        hyphae: dict[str, dict[str, Any]] = {}
        for key, hypha in self.hyphae.items():
            data = hypha.model_dump()
            if isinstance(hypha, NeuralRoot):
                data.update(
                    {
                        "active": False,
                        "state": "unbound",
                        "owner_generation": "",
                        "last_activity_at": 0.0,
                        "last_probe_at": 0.0,
                        "last_probe_success_at": 0.0,
                        "last_error": (
                            "persisted_historical_topology_requires_owner_attestation"
                        ),
                    }
                )
            created_at = self._vault_number(
                hypha.created_at,
                f"live hypha creation timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            last_pulse = self._vault_number(
                hypha.last_pulse,
                f"live hypha pulse timestamp: {key}",
                minimum=0.0,
                maximum=now_monotonic + _VAULT_CLOCK_SKEW_TOLERANCE_S,
            )
            created_age = max(0.0, now_monotonic - created_at)
            last_pulse_age = max(0.0, now_monotonic - last_pulse)
            if last_pulse_age > created_age + _VAULT_CLOCK_SKEW_TOLERANCE_S:
                raise ValueError(f"live hypha pulse predates creation: {key}")
            data["created_age_s"] = created_age
            data["last_pulse_age_s"] = last_pulse_age
            data.pop("created_at", None)
            data.pop("last_pulse", None)
            hyphae[key] = data

        return {
            "schema_version": 3,
            "captured_at_unix": captured_at_unix,
            "pathways": pathways,
            "hyphae": hyphae,
            # Serialized by the sync worker and never edited: the shared copy.
            "mapped_files": self._mapped_files_canonical_locked(),
            "centrality": dict(self._centrality),
            "critical_modules": list(self._critical_modules),
            "cross_links": {
                key: list(value) for key, value in self._cross_links.items()
            },
            "infrastructure_mapped": self.infrastructure_mapped,
            "mapping_generation": self._mapping_generation,
            "topology_revision": self._topology_revision,
        }

    @staticmethod
    def _vault_age(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite non-negative number")
        age = float(value)
        if not math.isfinite(age) or age < 0.0:
            raise ValueError(f"{label} must be a finite non-negative number")
        return age

    @staticmethod
    def _vault_number(
        value: Any,
        label: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number")
        if minimum is not None and number < minimum:
            raise ValueError(f"{label} is below its minimum")
        if maximum is not None and number > maximum:
            raise ValueError(f"{label} exceeds its maximum")
        return number

    @staticmethod
    def _vault_count(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
        return value

    @staticmethod
    def _vault_optional_string(value: Any, label: str) -> str | None:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{label} must be a string or null")
        return value

