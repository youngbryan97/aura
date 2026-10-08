"""Bind benchmark results to source cases, including repeated requests.

Content identifies a request; its occurrence in the complete source identifies
a case. Unique legacy IDs remain usable. Repeated IDs receive their original
source ordinal before any ordering or subset selection takes place.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def reference_identity(value: Any) -> str:
    """A type-preserving identity for a JSON benchmark reference."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceCase:
    id: str
    legacy_id: str
    source_ordinal: int
    reference_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {"source_ordinal": self.source_ordinal, "reference_sha256": self.reference_sha256}


class CaseCatalog:
    """The identities and reference bindings of one complete source."""

    def __init__(self, cases: Sequence[SourceCase]) -> None:
        self.cases = tuple(cases)
        self.by_id = {case.id: case for case in self.cases}
        if len(self.by_id) != len(self.cases):
            raise ValueError("source case IDs are not unique")
        counts = Counter(case.legacy_id for case in self.cases)
        self.ambiguous_legacy_ids = frozenset(key for key, count in counts.items() if count > 1)

    def validate_row(self, row: Mapping[str, Any], *, reference_field: str = "target") -> SourceCase:
        """Accept a saved result only when its case and reference agree."""
        key = row.get("id")
        if not isinstance(key, str):
            raise ValueError("saved row has no string case ID")
        if key in self.ambiguous_legacy_ids:
            raise ValueError(f"ambiguous legacy case ID {key!r}; the saved row needs explicit recovery")
        if key not in self.by_id:
            raise ValueError(f"saved row has an unexpected case ID {key!r}")
        case = self.by_id[key]
        if reference_field not in row or reference_identity(row[reference_field]) != case.reference_sha256:
            raise ValueError(f"saved row reference does not match source case {key!r}")
        if "source_ordinal" in row:
            ordinal = row["source_ordinal"]
            if type(ordinal) is not int or ordinal != case.source_ordinal:
                raise ValueError(f"saved row source ordinal does not match case {key!r}")
        elif case.id != case.legacy_id:
            raise ValueError(f"disambiguated case {key!r} has no saved source ordinal")
        if "reference_sha256" in row and row["reference_sha256"] != case.reference_sha256:
            raise ValueError(f"saved row reference identity does not match case {key!r}")
        return case

    def saved_rows(self, directory: Path) -> dict[str, dict[str, Any]]:
        """Read flat result rows without repairing or replacing their evidence."""
        saved: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError(f"saved row is not an object: {path}")
            case = self.validate_row(row)
            if path.stem != case.id:
                raise ValueError(f"saved row filename does not match case {case.id!r}: {path}")
            if case.id in saved:
                raise ValueError(f"duplicate saved case ID {case.id!r}")
            saved[case.id] = row
        return saved


def source_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    legacy_id: Callable[[Mapping[str, Any]], str],
    reference: Callable[[Mapping[str, Any]], Any],
) -> CaseCatalog:
    """Index the complete source once; select or reorder its cases afterward."""
    keys = [legacy_id(record) for record in records]
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("legacy case IDs must be nonempty strings")
    counts = Counter(keys)
    cases = [SourceCase(key if counts[key] == 1 else f"{key}--case-{ordinal}", key, ordinal,
                        reference_identity(reference(record)))
             for ordinal, (key, record) in enumerate(zip(keys, records, strict=True))]
    return CaseCatalog(cases)
