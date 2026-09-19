#!/usr/bin/env python3
"""Strict evidence ledger for the Aura requirement-to-proof control plane.

The requirement registry is generated from tracker scope and must remain free
of hand-maintained proof claims. This module owns the separate, reviewable
overlay that binds a requirement and evidence class to exact artifact bytes
and the source commit those bytes evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reqproof.schema import (  # noqa: E402
    REQUIREMENT_ID_RE,
    SHA256_RE,
    EvidenceRef,
    Registry,
    RegistrySchemaError,
    load_registry,
)

LEDGER_SCHEMA_VERSION = 3
COMMAND_RECEIPT_SCHEMA_V1 = "aura.reqproof.command_receipt.v1"
COMMAND_RECEIPT_SCHEMA_V2 = "aura.reqproof.command_receipt.v2"
COMMAND_RECEIPT_SCHEMAS = frozenset(
    {COMMAND_RECEIPT_SCHEMA_V1, COMMAND_RECEIPT_SCHEMA_V2}
)
DEFAULT_REGISTRY_PATH = ROOT / "config" / "requirement_registry.json"
DEFAULT_EVIDENCE_LEDGER_PATH = ROOT / "config" / "requirement_evidence_ledger.json"


class EvidenceLedgerError(ValueError):
    """The evidence ledger or one of its references violated its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceLedgerError(message)


def _check_string(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
    return cast(str, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_evidence_target(root: Path, ref: str) -> Path:
    """Resolve one canonical repo-relative regular file without symlink escape."""
    posix = PurePosixPath(ref)
    _require("\\" not in ref, f"evidence ref must use POSIX separators: {ref!r}")
    _require(not posix.is_absolute(), f"evidence ref must be repo-relative: {ref!r}")
    _require(
        bool(posix.parts) and all(part not in ("", ".", "..") for part in posix.parts),
        f"evidence ref contains an unsafe path component: {ref!r}",
    )
    _require(posix.as_posix() == ref, f"evidence ref is not canonical: {ref!r}")

    root_resolved = root.resolve()
    target = root.joinpath(*posix.parts)
    current = root
    for part in posix.parts:
        current = current / part
        _require(not current.is_symlink(), f"evidence ref traverses a symlink: {ref!r}")
    _require(target.is_file(), f"evidence ref does not name a regular file: {ref!r}")
    resolved = target.resolve()
    _require(
        resolved.is_relative_to(root_resolved),
        f"evidence ref escapes repository root: {ref!r}",
    )
    return resolved


def validate_source_selectors(data: Any) -> dict[str, tuple[str, ...]]:
    _require(isinstance(data, dict), "source_selectors must be an object")
    _require(
        set(data) == {"paths", "globs"},
        "source_selectors fields must be exactly paths and globs",
    )
    validated: dict[str, tuple[str, ...]] = {}
    for key in ("paths", "globs"):
        values = data.get(key)
        _require(isinstance(values, list), f"source_selectors.{key} must be a list")
        items: list[str] = []
        for index, value in enumerate(values):
            text = _check_string(value, f"source_selectors.{key}[{index}]")
            posix = PurePosixPath(text)
            _require(
                "\\" not in text and not posix.is_absolute(),
                f"unsafe source selector {text!r}",
            )
            _require(
                bool(posix.parts)
                and all(part not in ("", ".", "..") for part in posix.parts)
                and posix.as_posix() == text,
                f"non-canonical source selector {text!r}",
            )
            if key == "paths":
                _require(
                    not any(char in text for char in "*?["),
                    f"exact source path contains glob syntax: {text!r}",
                )
            else:
                _require(
                    any(char in text for char in "*?["),
                    f"source glob has no pattern syntax: {text!r}",
                )
            items.append(text)
        _require(items == sorted(set(items)), f"source_selectors.{key} must be sorted and unique")
        validated[key] = tuple(items)
    _require(
        bool(validated["paths"] or validated["globs"]),
        "source_selectors must declare at least one path or glob",
    )
    return validated


def expand_source_selectors(
    root: Path,
    selectors: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Expand exact paths and repo-relative globs without symlink traversal."""
    root_resolved = root.resolve()
    refs = set(selectors["paths"])
    for pattern in selectors["globs"]:
        for candidate in root.glob(pattern):
            if candidate.is_symlink():
                raise EvidenceLedgerError(
                    f"source selector matched a symlink: {candidate.relative_to(root)}"
                )
            if candidate.is_file():
                resolved = candidate.resolve()
                _require(
                    resolved.is_relative_to(root_resolved),
                    f"source selector escaped repository root: {pattern!r}",
                )
                refs.add(resolved.relative_to(root_resolved).as_posix())
    _require(bool(refs), "source selectors matched no files")
    for ref in sorted(refs):
        resolve_evidence_target(root, ref)
    return tuple(sorted(refs))


def load_evidence_receipt(
    target: Path,
    *,
    requirement_id: str,
    evidence_class: str,
    acceptance_ids: tuple[str, ...],
    commit: str,
) -> dict[str, Any]:
    """Validate the provenance envelope required for external ledger evidence."""
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceLedgerError(f"evidence receipt is not valid UTF-8 JSON: {exc}") from exc
    _require(isinstance(data, dict), "evidence receipt must be an object")
    schema = data.get("schema")
    _require(
        schema in COMMAND_RECEIPT_SCHEMAS,
        "evidence receipt schema must be a supported command receipt schema",
    )
    _require(data.get("verdict") == "pass", "evidence receipt verdict must be pass")
    _require(
        data.get("source_commit") == commit,
        "evidence receipt source_commit does not match ledger commit",
    )
    targets = data.get("evidence_targets")
    _require(isinstance(targets, list) and bool(targets), "evidence_targets must be non-empty")
    matching_acceptance: set[str] = set()
    for index, item in enumerate(targets):
        _require(isinstance(item, dict), f"evidence_targets[{index}] must be an object")
        _require(
            set(item) == {"requirement_id", "evidence_class", "acceptance_ids"},
            f"evidence_targets[{index}] fields are invalid",
        )
        values = item.get("acceptance_ids")
        _require(
            isinstance(values, list)
            and bool(values)
            and all(isinstance(value, str) for value in values),
            f"evidence_targets[{index}].acceptance_ids is invalid",
        )
        if (
            item.get("requirement_id") == requirement_id
            and item.get("evidence_class") == evidence_class
        ):
            matching_acceptance.update(values)
    _require(
        set(acceptance_ids) <= matching_acceptance,
        "evidence receipt does not declare every ledger acceptance unit",
    )

    manifest = data.get("source_manifest")
    _require(isinstance(manifest, list) and bool(manifest), "source_manifest must be non-empty")
    paths: list[str] = []
    for index, item in enumerate(manifest):
        _require(isinstance(item, dict), f"source_manifest[{index}] must be an object")
        _require(
            set(item) == {"path", "sha256", "size_bytes"},
            f"source_manifest[{index}] fields are invalid",
        )
        ref = _check_string(item.get("path"), f"source_manifest[{index}].path")
        posix = PurePosixPath(ref)
        _require("\\" not in ref and not posix.is_absolute(), f"unsafe source path {ref!r}")
        _require(
            bool(posix.parts)
            and all(part not in ("", ".", "..") for part in posix.parts)
            and posix.as_posix() == ref,
            f"non-canonical source path {ref!r}",
        )
        digest = item.get("sha256")
        _require(
            isinstance(digest, str) and bool(SHA256_RE.match(digest)),
            f"source_manifest[{index}].sha256 is invalid",
        )
        size = item.get("size_bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"source_manifest[{index}].size_bytes is invalid",
        )
        paths.append(ref)
    _require(paths == sorted(set(paths)), "source_manifest paths must be sorted and unique")
    if schema == COMMAND_RECEIPT_SCHEMA_V2:
        validate_source_selectors(data.get("source_selectors"))
    return cast(dict[str, Any], data)


@dataclass(frozen=True)
class EvidenceLedgerEntry:
    requirement_id: str
    acceptance_ids: tuple[str, ...]
    evidence: EvidenceRef

    ALLOWED_KEYS = frozenset(
        {
            "requirement_id",
            "acceptance_ids",
            "evidence_class",
            "ref",
            "sha256",
            "commit",
            "recorded_at",
        }
    )
    ACCEPTANCE_ID_RE = re.compile(r"^A[1-9][0-9]*$")

    @classmethod
    def from_dict(cls, data: Any, name: str) -> EvidenceLedgerEntry:
        _require(isinstance(data, dict), f"{name} must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"{name} has unknown fields: {sorted(unknown)}")
        requirement_id = _check_string(data.get("requirement_id"), f"{name}.requirement_id")
        _require(
            bool(REQUIREMENT_ID_RE.match(requirement_id)),
            f"{name}.requirement_id does not match the requirement ID pattern",
        )
        acceptance_raw = data.get("acceptance_ids")
        _require(isinstance(acceptance_raw, list), f"{name}.acceptance_ids must be a list")
        acceptance_ids = tuple(
            _check_string(value, f"{name}.acceptance_ids[{index}]")
            for index, value in enumerate(acceptance_raw)
        )
        _require(bool(acceptance_ids), f"{name}.acceptance_ids must be non-empty")
        _require(
            all(cls.ACCEPTANCE_ID_RE.match(value) for value in acceptance_ids),
            f"{name}.acceptance_ids entries must match A1, A2, ...",
        )
        _require(
            acceptance_ids == tuple(sorted(set(acceptance_ids), key=lambda value: int(value[1:]))),
            f"{name}.acceptance_ids must be numerically sorted and unique",
        )
        try:
            evidence = EvidenceRef.from_dict(
                {key: data.get(key) for key in EvidenceRef.ALLOWED_KEYS}, name
            )
        except RegistrySchemaError as exc:
            raise EvidenceLedgerError(str(exc)) from exc
        return cls(
            requirement_id=requirement_id,
            acceptance_ids=acceptance_ids,
            evidence=evidence,
        )

    @property
    def sort_key(self) -> tuple[str, ...]:
        evidence = self.evidence
        return (
            self.requirement_id,
            evidence.evidence_class,
            ",".join(self.acceptance_ids),
            evidence.ref,
            evidence.sha256,
            evidence.commit,
            evidence.recorded_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "acceptance_ids": list(self.acceptance_ids),
            **self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceLedger:
    schema_version: int
    registry_declaration_sha256: str
    entries: tuple[EvidenceLedgerEntry, ...]
    content_sha256: str = field(default="", compare=False)

    ALLOWED_KEYS = frozenset(
        {"schema_version", "registry_declaration_sha256", "entries", "content_sha256"}
    )

    @classmethod
    def empty_for(cls, registry: Registry) -> EvidenceLedger:
        return cls(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=registry.declaration_sha256(),
            entries=(),
        )

    @classmethod
    def from_dict(cls, data: Any, *, verify_hash: bool = True) -> EvidenceLedger:
        _require(isinstance(data, dict), "evidence ledger must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"evidence ledger has unknown fields: {sorted(unknown)}")
        missing = cls.ALLOWED_KEYS - set(data)
        _require(not missing, f"evidence ledger is missing fields: {sorted(missing)}")
        _require(
            data.get("schema_version") == LEDGER_SCHEMA_VERSION,
            f"evidence ledger schema_version must be {LEDGER_SCHEMA_VERSION}",
        )
        registry_hash = _check_string(
            data.get("registry_declaration_sha256"), "registry_declaration_sha256"
        )
        _require(bool(SHA256_RE.match(registry_hash)), "registry_declaration_sha256 is not sha256")
        raw_entries = data.get("entries")
        _require(isinstance(raw_entries, list), "evidence ledger entries must be a list")
        entries = tuple(
            EvidenceLedgerEntry.from_dict(item, f"entries[{index}]")
            for index, item in enumerate(raw_entries)
        )
        keys = [entry.sort_key for entry in entries]
        _require(keys == sorted(keys), "evidence ledger entries must be canonically sorted")
        _require(len(keys) == len(set(keys)), "evidence ledger contains duplicate entries")
        ledger = cls(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=registry_hash,
            entries=entries,
            content_sha256=str(data.get("content_sha256", "")),
        )
        if verify_hash:
            recorded = data.get("content_sha256")
            _require(
                isinstance(recorded, str) and bool(SHA256_RE.match(recorded)),
                "evidence ledger content_sha256 missing or malformed",
            )
            actual = ledger.compute_content_sha256()
            _require(
                recorded == actual,
                "evidence ledger content_sha256 does not match content: "
                f"recorded {recorded[:12]}..., actual {actual[:12]}...",
            )
        return ledger

    def body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_declaration_sha256": self.registry_declaration_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def compute_content_sha256(self) -> str:
        canonical = json.dumps(self.body_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        body = self.body_dict()
        body["content_sha256"] = self.compute_content_sha256()
        return body

    def to_canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def by_requirement(self) -> dict[str, tuple[EvidenceRef, ...]]:
        grouped: dict[str, list[EvidenceRef]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.requirement_id, []).append(entry.evidence)
        return {key: tuple(value) for key, value in grouped.items()}

    def entries_by_requirement(self) -> dict[str, tuple[EvidenceLedgerEntry, ...]]:
        grouped: dict[str, list[EvidenceLedgerEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.requirement_id, []).append(entry)
        return {key: tuple(value) for key, value in grouped.items()}


def load_evidence_ledger(path: Path) -> EvidenceLedger:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceLedgerError(f"evidence ledger not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceLedgerError(f"evidence ledger is not valid JSON: {path}: {exc}") from exc
    return EvidenceLedger.from_dict(data)


def write_evidence_ledger_atomic(ledger: EvidenceLedger, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ledger.to_canonical_json()
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return ledger.compute_content_sha256()


def verify_ledger_binding(ledger: EvidenceLedger, registry: Registry) -> None:
    expected = registry.declaration_sha256()
    _require(
        ledger.registry_declaration_sha256 == expected,
        "evidence ledger is bound to a different set of requirements: "
        f"recorded {ledger.registry_declaration_sha256[:12]}..., expected {expected[:12]}... "
        "(a requirement, an acceptance cell or an evidence class changed under "
        "the evidence; re-record the evidence rather than the hash)",
    )
    known = registry.by_id()
    for entry in ledger.entries:
        _require(
            entry.requirement_id in known,
            f"evidence ledger references unknown requirement {entry.requirement_id}",
        )
        requirement = known[entry.requirement_id]
        _require(
            entry.evidence.evidence_class in requirement.evidence_required,
            f"{entry.requirement_id} does not require evidence class "
            f"{entry.evidence.evidence_class}",
        )
        valid_acceptance_ids = {
            f"A{index}" for index in range(1, len(requirement.acceptance) + 1)
        }
        unknown_acceptance_ids = sorted(
            set(entry.acceptance_ids) - valid_acceptance_ids
        )
        _require(
            not unknown_acceptance_ids,
            f"{entry.requirement_id} has unknown acceptance IDs "
            f"{unknown_acceptance_ids}",
        )
        invalid_modalities = [
            acceptance_id
            for acceptance_id in entry.acceptance_ids
            if entry.evidence.evidence_class
            not in requirement.required_evidence_for(acceptance_id)
        ]
        _require(
            not invalid_modalities,
            f"{entry.requirement_id} does not require evidence class "
            f"{entry.evidence.evidence_class} for acceptance IDs "
            f"{invalid_modalities}",
        )


def add_entry(
    ledger: EvidenceLedger,
    registry: Registry,
    *,
    requirement_id: str,
    evidence_class: str,
    acceptance_ids: tuple[str, ...],
    ref: str,
    commit: str,
    recorded_at: str,
    root: Path,
) -> EvidenceLedger:
    verify_ledger_binding(ledger, registry)
    known = registry.by_id()
    _require(requirement_id in known, f"unknown requirement {requirement_id}")
    _require(
        evidence_class in known[requirement_id].evidence_required,
        f"{requirement_id} does not require evidence class {evidence_class}",
    )
    requirement = known[requirement_id]
    valid_acceptance_ids = set(requirement.acceptance_ids())
    unknown_acceptance_ids = sorted(set(acceptance_ids) - valid_acceptance_ids)
    _require(
        not unknown_acceptance_ids,
        f"{requirement_id} has unknown acceptance IDs {unknown_acceptance_ids}",
    )
    invalid_modalities = [
        acceptance_id
        for acceptance_id in acceptance_ids
        if evidence_class not in requirement.required_evidence_for(acceptance_id)
    ]
    _require(
        not invalid_modalities,
        f"{requirement_id} does not require evidence class {evidence_class} "
        f"for acceptance IDs {invalid_modalities}",
    )
    target = resolve_evidence_target(root, ref)
    load_evidence_receipt(
        target,
        requirement_id=requirement_id,
        evidence_class=evidence_class,
        acceptance_ids=acceptance_ids,
        commit=commit,
    )
    try:
        evidence = EvidenceRef.from_dict(
            {
                "evidence_class": evidence_class,
                "ref": ref,
                "sha256": sha256_file(target),
                "commit": commit,
                "recorded_at": recorded_at,
            },
            "evidence",
        )
    except RegistrySchemaError as exc:
        raise EvidenceLedgerError(str(exc)) from exc
    entry = EvidenceLedgerEntry.from_dict(
        {
            "requirement_id": requirement_id,
            "acceptance_ids": list(acceptance_ids),
            **evidence.to_dict(),
        },
        "evidence entry",
    )
    _require(
        entry.sort_key not in {item.sort_key for item in ledger.entries},
        "evidence entry already exists",
    )
    verify_ledger_binding(
        EvidenceLedger(
            schema_version=LEDGER_SCHEMA_VERSION,
            registry_declaration_sha256=registry.declaration_sha256(),
            entries=(entry,),
        ),
        registry,
    )
    retained: list[EvidenceLedgerEntry] = []
    replacement_cells = set(entry.acceptance_ids)
    for current in ledger.entries:
        if (
            current.requirement_id != entry.requirement_id
            or current.evidence.evidence_class != entry.evidence.evidence_class
        ):
            retained.append(current)
            continue
        remaining = tuple(
            acceptance_id
            for acceptance_id in current.acceptance_ids
            if acceptance_id not in replacement_cells
        )
        if remaining:
            retained.append(
                EvidenceLedgerEntry(
                    requirement_id=current.requirement_id,
                    acceptance_ids=remaining,
                    evidence=current.evidence,
                )
            )
    entries = tuple(sorted((*retained, entry), key=lambda item: item.sort_key))
    return EvidenceLedger(
        schema_version=LEDGER_SCHEMA_VERSION,
        registry_declaration_sha256=registry.declaration_sha256(),
        entries=entries,
    )


def _resolve_commit(root: Path, revision: str) -> str:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    result = get_subprocess_gateway().run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=root,
        timeout=30,
        read_only=True,
        source="reqproof_evidence_resolve_commit",
        accelerator_capability="none",
    )
    _require(result.returncode == 0, f"unknown git commit {revision!r}")
    commit = result.stdout.strip()
    _require(bool(commit) and len(commit) == 40, f"git returned invalid commit for {revision!r}")
    return cast(str, commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--ledger", default=str(DEFAULT_EVIDENCE_LEDGER_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="create an empty ledger for the current registry")
    subparsers.add_parser("rebind", help="bind existing valid entries to the current registry")
    record = subparsers.add_parser("record", help="record one exact evidence artifact")
    record.add_argument("--requirement", required=True)
    record.add_argument("--class", dest="evidence_class", required=True)
    record.add_argument(
        "--acceptance",
        dest="acceptance_ids",
        action="append",
        required=True,
        help="acceptance unit ID (A1, A2, ...); repeat for multiple units",
    )
    record.add_argument("--ref", required=True)
    record.add_argument("--commit", default="HEAD")
    record.add_argument("--recorded-at", default=date.today().isoformat())
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    ledger_path = Path(args.ledger)
    if args.command == "init":
        _require(not ledger_path.exists(), f"refusing to overwrite existing ledger {ledger_path}")
        ledger = EvidenceLedger.empty_for(registry)
    else:
        ledger = load_evidence_ledger(ledger_path)
        if args.command == "rebind":
            known = registry.by_id()
            for entry in ledger.entries:
                _require(
                    entry.requirement_id in known,
                    f"cannot rebind unknown requirement {entry.requirement_id}",
                )
                _require(
                    entry.evidence.evidence_class
                    in known[entry.requirement_id].evidence_required,
                    f"cannot rebind obsolete evidence class for {entry.requirement_id}",
                )
                requirement = known[entry.requirement_id]
                invalid_modalities = [
                    acceptance_id
                    for acceptance_id in entry.acceptance_ids
                    if entry.evidence.evidence_class
                    not in requirement.required_evidence_for(acceptance_id)
                ]
                _require(
                    not invalid_modalities,
                    "cannot rebind evidence to acceptance units that do not "
                    f"require its class: {entry.requirement_id} "
                    f"{invalid_modalities}",
                )
            ledger = EvidenceLedger(
                schema_version=LEDGER_SCHEMA_VERSION,
                registry_declaration_sha256=registry.declaration_sha256(),
                entries=ledger.entries,
            )
            verify_ledger_binding(ledger, registry)
        else:
            ledger = add_entry(
                ledger,
                registry,
                requirement_id=args.requirement,
                evidence_class=args.evidence_class,
                acceptance_ids=tuple(args.acceptance_ids),
                ref=args.ref,
                commit=_resolve_commit(ROOT, args.commit),
                recorded_at=args.recorded_at,
                root=ROOT,
            )
    digest = write_evidence_ledger_atomic(ledger, ledger_path)
    print(
        json.dumps(
            {"ledger": str(ledger_path), "entries": len(ledger.entries), "sha256": digest},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
