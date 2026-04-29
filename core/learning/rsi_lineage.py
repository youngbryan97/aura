"""Tamper-evident RSI generation lineage.

The lineage ledger records successor attempts as evidence, not vibes. It does
not declare hard RSI by itself; it gives auditors enough structure to verify
generation-to-generation capability and improver-score movement.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GENESIS_HASH = "sha256:" + "0" * 64
SCHEMA_VERSION = 1

VERDICT_NO_RSI = "NO_RSI"
VERDICT_BOUNDED = "BOUNDED_SELF_OPTIMIZATION"
VERDICT_WEAK = "WEAK_RSI"
VERDICT_STRONG = "STRONG_RSI"
VERDICT_UNDENIABLE = "UNDENIABLE_RSI"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _hash(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj)).hexdigest()


@dataclass(frozen=True)
class RSIGenerationRecord:
    generation_id: str
    parent_generation_id: Optional[str]
    hypothesis: str
    intervention_type: str
    artifact_hashes: Dict[str, str]
    baseline_score: float
    after_score: float
    hidden_eval_score: float
    regressions: List[str] = field(default_factory=list)
    promoted: bool = False
    rollback_performed: bool = False
    ablation_result: str = "not_run"
    time_to_valid_improvement_s: float = 0.0
    improver_score: float = 0.0
    tamper_flags: List[str] = field(default_factory=list)
    safety_flags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def score_delta(self) -> float:
        return float(self.after_score) - float(self.baseline_score)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["score_delta"] = self.score_delta
        return payload


@dataclass(frozen=True)
class RSILineageVerdict:
    verdict: str
    reasons: List[str]
    generations: int
    capability_curve: List[float]
    improver_curve: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RSILineageLedger:
    """Append-only hash chain for RSI generation records."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: RSIGenerationRecord) -> Dict[str, Any]:
        prev_hash, seq = self._head()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "seq": seq,
            "prev_hash": prev_hash,
            "record": record.to_dict(),
        }
        payload["record_hash"] = _hash(payload["record"])
        payload["entry_hash"] = _hash({
            "schema_version": payload["schema_version"],
            "seq": payload["seq"],
            "prev_hash": payload["prev_hash"],
            "record_hash": payload["record_hash"],
        })
        line = json.dumps(payload, sort_keys=True, default=str) + "\n"
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self.path), flags, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            try:
                os.fsync(fd)
            except OSError:
                pass
        finally:
            os.close(fd)
        return payload

    def load_records(self) -> List[RSIGenerationRecord]:
        records: List[RSIGenerationRecord] = []
        for entry in self._entries():
            data = dict(entry["record"])
            data.pop("score_delta", None)
            records.append(RSIGenerationRecord(**data))
        return records

    def verify(self) -> Tuple[bool, List[str]]:
        problems: List[str] = []
        expected_prev = GENESIS_HASH
        expected_seq = 0
        for entry in self._entries():
            seq = int(entry.get("seq", -1))
            if seq != expected_seq:
                problems.append(f"seq_gap:{expected_seq}->{seq}")
            if entry.get("prev_hash") != expected_prev:
                problems.append(f"prev_hash_mismatch:seq{seq}")
            record_hash = _hash(entry.get("record", {}))
            if entry.get("record_hash") != record_hash:
                problems.append(f"record_hash_mismatch:seq{seq}")
            entry_hash = _hash({
                "schema_version": entry.get("schema_version"),
                "seq": entry.get("seq"),
                "prev_hash": entry.get("prev_hash"),
                "record_hash": entry.get("record_hash"),
            })
            if entry.get("entry_hash") != entry_hash:
                problems.append(f"entry_hash_mismatch:seq{seq}")
            expected_prev = str(entry.get("entry_hash"))
            expected_seq = seq + 1
        return not problems, problems

    def _head(self) -> Tuple[str, int]:
        last_hash = GENESIS_HASH
        next_seq = 0
        for entry in self._entries():
            last_hash = str(entry.get("entry_hash", GENESIS_HASH))
            next_seq = int(entry.get("seq", -1)) + 1
        return last_hash, next_seq

    def _entries(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def evaluate_lineage(records: List[RSIGenerationRecord], *, independently_reproduced: bool = False) -> RSILineageVerdict:
    if not records:
        return RSILineageVerdict(VERDICT_NO_RSI, ["no generation records"], 0, [], [])

    capability_curve = [float(record.after_score) for record in records]
    improver_curve = [float(record.improver_score) for record in records]
    reasons: List[str] = []

    if any(record.tamper_flags for record in records):
        reasons.append("tamper flags present")
    if any(record.regressions for record in records):
        reasons.append("regressions present")
    if not all(record.promoted for record in records):
        reasons.append("not every generation promoted")
    if len(records) < 2:
        reasons.append("fewer than two generations")

    capability_monotone = all(b > a for a, b in zip(capability_curve, capability_curve[1:]))
    improver_monotone = all(b > a for a, b in zip(improver_curve, improver_curve[1:]))
    if not capability_monotone:
        reasons.append("capability curve is not strictly increasing")
    if not improver_monotone:
        reasons.append("improver curve is not strictly increasing")

    if reasons:
        return RSILineageVerdict(VERDICT_BOUNDED, reasons, len(records), capability_curve, improver_curve)
    if len(records) >= 4 and independently_reproduced:
        return RSILineageVerdict(
            VERDICT_UNDENIABLE,
            ["independent reproduction plus monotone capability and improver curves"],
            len(records),
            capability_curve,
            improver_curve,
        )
    if len(records) >= 4:
        return RSILineageVerdict(
            VERDICT_STRONG,
            ["monotone capability and improver curves across at least four generations"],
            len(records),
            capability_curve,
            improver_curve,
        )
    return RSILineageVerdict(
        VERDICT_WEAK,
        ["monotone capability and improver curves, but too few generations for strong RSI"],
        len(records),
        capability_curve,
        improver_curve,
    )


__all__ = [
    "RSIGenerationRecord",
    "RSILineageLedger",
    "RSILineageVerdict",
    "VERDICT_BOUNDED",
    "VERDICT_NO_RSI",
    "VERDICT_STRONG",
    "VERDICT_UNDENIABLE",
    "VERDICT_WEAK",
    "evaluate_lineage",
]
