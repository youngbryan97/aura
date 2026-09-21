"""Durable prefixes of an ordered experiment, without a completion claim."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.verify.invariants import invariant


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class CampaignProgress:
    """Persist each sample and the continuation state after it, atomically.

    The caller owns random seeds and continuation-state semantics. Progress is
    neither a verdict nor an authority artifact. A different experiment must
    use a different path; it cannot reuse samples from this one.
    """

    def __init__(
        self,
        path: Path,
        *,
        identity: dict,
        conditions: tuple[str, ...],
        samples_per_condition: int,
        write: Callable[[dict], None],
        resume: bool=False,
    ) -> None:
        if not conditions or len(set(conditions)) != len(conditions) or samples_per_condition < 1:
            raise ValueError("campaign_progress_shape_invalid")
        self.path = path
        self._write = write
        self.identity = json.loads(_canonical(identity))
        self.conditions = conditions
        self.samples_per_condition = samples_per_condition
        self.outputs: dict[str, list[str]] = {name: [] for name in conditions}
        self.sample_metadata: dict[str, list[dict | None]] = {name: [] for name in conditions}
        self.last_condition = ""
        self.continuation: Any = None
        self.decode_seconds = 0.0
        if path.exists():
            if not resume:
                raise ValueError("campaign_progress_exists_use_resume")
            payload = json.loads(path.read_text(encoding="utf-8"))
            checksum = payload.pop("payload_sha256", None)
            if checksum != _digest(payload):
                raise ValueError("campaign_progress_corrupt")
            if (payload.get("schema") != "aura.campaign.progress.v1"
                    or payload.get("identity") != self.identity
                    or payload.get("conditions") != list(conditions)
                    or payload.get("samples_per_condition") != samples_per_condition):
                raise ValueError("campaign_progress_identity_mismatch")
            self.outputs = payload["outputs"]
            self.sample_metadata = payload.get("sample_metadata", {
                name: [None] * len(rows) for name, rows in self.outputs.items()
            })
            self.last_condition = payload["last_condition"]
            self.continuation = payload["continuation"]
            self.decode_seconds = payload["decode_seconds"]
            self._validate()
        elif resume:
            raise ValueError("campaign_progress_missing")

    def _validate(self) -> None:
        if set(self.outputs) != set(self.conditions) or set(self.sample_metadata) != set(self.conditions):
            raise ValueError("campaign_progress_conditions_invalid")
        unfinished = False
        last = ""
        for name in self.conditions:
            rows = self.outputs[name]
            if (not isinstance(rows, list) or any(not isinstance(row, str) for row in rows)
                    or len(rows) > self.samples_per_condition or (unfinished and rows)):
                raise ValueError("campaign_progress_not_a_prefix")
            if rows:
                last = name
            metadata = self.sample_metadata[name]
            if (not isinstance(metadata, list) or len(metadata) != len(rows)
                    or any(row is not None and not isinstance(row, dict) for row in metadata)):
                raise ValueError("campaign_progress_metadata_mismatch")
            unfinished |= len(rows) < self.samples_per_condition
        if last != self.last_condition:
            raise ValueError("campaign_progress_continuation_mismatch")
        if not math.isfinite(self.decode_seconds) or self.decode_seconds < 0:
            raise ValueError("campaign_progress_time_invalid")

    @property
    def complete(self) -> bool:
        return all(len(self.outputs[name]) == self.samples_per_condition for name in self.conditions)

    def state_for(self, condition: str) -> Any:
        return json.loads(_canonical(self.continuation)) if condition == self.last_condition else None

    def record(self, condition: str, output: str, continuation: Any, *, seconds: float,
               metadata: dict | None = None) -> None:
        pending = next((name for name in self.conditions
                        if len(self.outputs[name]) < self.samples_per_condition), None)
        if condition != pending or not isinstance(output, str):
            raise ValueError("campaign_progress_out_of_order")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("campaign_progress_time_invalid")
        state = json.loads(_canonical(continuation))
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("campaign_progress_metadata_invalid")
        sample_metadata = {name: list(rows) for name, rows in self.sample_metadata.items()}
        sample_metadata[condition].append(json.loads(_canonical(metadata)))
        outputs = {name: list(rows) for name, rows in self.outputs.items()}
        outputs[condition].append(output)
        payload = {
            "schema": "aura.campaign.progress.v1", "identity": self.identity,
            "conditions": list(self.conditions), "samples_per_condition": self.samples_per_condition,
            "outputs": outputs, "last_condition": condition, "continuation": state,
            "sample_metadata": sample_metadata,
            "decode_seconds": self.decode_seconds + seconds,
        }
        payload["payload_sha256"] = _digest(payload)
        self._write(payload)
        self.outputs = outputs
        self.sample_metadata = sample_metadata
        self.last_condition = condition
        self.continuation = state
        self.decode_seconds += seconds


@invariant("evaluation.progress_identity", scope="evaluation",
           owner="core/evaluation/campaign_progress.py", observational=False)
def _progress_identity() -> tuple:
    assert _digest({"seed": 1, "model": "a"}) != _digest({"seed": 2, "model": "a"})
    assert _digest({"seed": 1, "model": "a"}) != _digest({"seed": 1, "model": "b"})
    return ()
