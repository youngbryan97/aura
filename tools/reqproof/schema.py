#!/usr/bin/env python3
"""Requirement-to-proof registry schema (SCOPE-001 / PROGRESS-CONTROL-001).

This module defines the canonical machine-readable requirement registry: the
single structure from which scope, closure, coverage, and (later) progress
percentages are derived. Design constraints, in priority order:

1. **Strict**: unknown fields, bad enums, malformed IDs, and type mismatches
   are hard errors. A registry that parses is a registry whose every field is
   meaningful.
2. **Deterministic**: serialization is canonical (sorted keys, sorted
   requirement order, trailing newline) so the registry content hash is
   reproducible byte-for-byte from the same logical content.
3. **No prose authority**: nothing in this schema lets free text close a
   requirement. Closure is a function of ``state`` plus the validator's
   evidence and child checks (tools/reqproof/validate.py) — editing ``notes``
   or ``status_detail`` can never change closure truth.

The registry file lives at ``config/requirement_registry.json`` and is a pure
deterministic function of the tracker extraction (tools/reqproof/migrate.py)
until later sessions layer evidence ledgers on top. Hand-maintained knowledge
belongs in separate overlay files (coverage map, evidence ledger), never in
edits to the generated registry.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = 2

STATES: tuple[str, ...] = (
    "open",
    "in_progress",
    "blocked",
    "complete",
    "deferred",
    "withdrawn",
)

# States that count as "no longer an open obligation" for closure math.
# ``withdrawn`` is out of scope by decision; ``complete`` requires evidence.
CLOSED_STATES: tuple[str, ...] = ("complete", "withdrawn")

KINDS: tuple[str, ...] = ("parent", "atomic")

EVIDENCE_CLASSES: tuple[str, ...] = (
    "implementation",
    "test",
    "live",
    "gui",
    "security",
    "portability",
    "release",
    "soak",
)

WEIGHT_PROVENANCES: tuple[str, ...] = ("default", "assigned")

REQUIREMENT_ID_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RegistrySchemaError(ValueError):
    """A registry document violated the strict schema."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistrySchemaError(message)


def _check_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str), f"{name} must be a string, got {type(value).__name__}")
    if not allow_empty:
        _require(bool(value), f"{name} must be non-empty")
    return cast(str, value)


def _check_str_list(value: Any, name: str) -> list[str]:
    _require(isinstance(value, list), f"{name} must be a list")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(_check_str(item, f"{name}[{index}]"))
    return out


@dataclass(frozen=True)
class SourceRef:
    """Provenance of an obligation: where in which corpus it came from."""

    corpus: str
    locator: str
    sha256: str = ""

    ALLOWED_KEYS = frozenset({"corpus", "locator", "sha256"})

    @classmethod
    def from_dict(cls, data: Any, name: str) -> SourceRef:
        _require(isinstance(data, dict), f"{name} must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"{name} has unknown fields: {sorted(unknown)}")
        corpus = _check_str(data.get("corpus"), f"{name}.corpus")
        locator = _check_str(data.get("locator"), f"{name}.locator")
        sha = data.get("sha256", "")
        _require(isinstance(sha, str), f"{name}.sha256 must be a string")
        if sha:
            _require(bool(SHA256_RE.match(sha)), f"{name}.sha256 is not a sha256 hex digest")
        return cls(corpus=corpus, locator=locator, sha256=sha)

    def to_dict(self) -> dict[str, str]:
        return {"corpus": self.corpus, "locator": self.locator, "sha256": self.sha256}


@dataclass(frozen=True)
class EvidenceRef:
    """One recorded, verifiable piece of closure evidence.

    ``ref`` is a repo-relative path or artifact path; ``sha256`` pins its
    content; ``commit`` pins the source commit the evidence was produced
    from. The validator rejects refs whose target is missing, whose hash no
    longer matches, or whose commit is unknown to the repository — an old
    artifact cannot silently satisfy a new obligation.
    """

    evidence_class: str
    ref: str
    sha256: str
    commit: str
    recorded_at: str

    ALLOWED_KEYS = frozenset({"evidence_class", "ref", "sha256", "commit", "recorded_at"})

    @classmethod
    def from_dict(cls, data: Any, name: str) -> EvidenceRef:
        _require(isinstance(data, dict), f"{name} must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"{name} has unknown fields: {sorted(unknown)}")
        evidence_class = _check_str(data.get("evidence_class"), f"{name}.evidence_class")
        _require(
            evidence_class in EVIDENCE_CLASSES,
            f"{name}.evidence_class {evidence_class!r} not in {EVIDENCE_CLASSES}",
        )
        ref = _check_str(data.get("ref"), f"{name}.ref")
        sha = _check_str(data.get("sha256"), f"{name}.sha256")
        _require(bool(SHA256_RE.match(sha)), f"{name}.sha256 is not a sha256 hex digest")
        commit = _check_str(data.get("commit"), f"{name}.commit")
        _require(
            bool(re.match(r"^[0-9a-f]{7,40}$", commit)),
            f"{name}.commit is not a git object id",
        )
        recorded_at = _check_str(data.get("recorded_at"), f"{name}.recorded_at")
        _require(
            bool(ISO_DATE_RE.match(recorded_at)),
            f"{name}.recorded_at must be an ISO date (YYYY-MM-DD)",
        )
        return cls(
            evidence_class=evidence_class,
            ref=ref,
            sha256=sha,
            commit=commit,
            recorded_at=recorded_at,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "evidence_class": self.evidence_class,
            "ref": self.ref,
            "sha256": self.sha256,
            "commit": self.commit,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class Requirement:
    """One requirement: either a parent workstream or an atomic closure unit.

    Closure semantics (enforced by validate.py, restated here as contract):

    * ``state == "complete"`` is a claim, not a fact. It is only accepted when
      every acceptance/class pair in ``acceptance_evidence_required`` has
      verifiable evidence AND every requirement in ``closure_requires`` is
      itself closed. Otherwise the validator reports FALSE-CLOSURE defects.
    * A ``parent`` cannot be more closed than its children: any non-closed
      member of ``closure_requires`` blocks the parent.
    * ``mandatory`` requirements in a non-closed state block release.
    """

    id: str
    title: str
    kind: str
    state: str
    status_detail: str
    status_date: str
    mandatory: bool
    owner: str
    risk_weight: float
    proof_weight: float
    weight_provenance: str
    sources: tuple[SourceRef, ...]
    depends_on: tuple[str, ...]
    closure_requires: tuple[str, ...]
    parent: str | None
    acceptance: tuple[str, ...]
    acceptance_evidence_required: tuple[tuple[str, ...], ...]
    evidence_required: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    non_claims: tuple[str, ...]
    notes: str = ""

    ALLOWED_KEYS = frozenset(
        {
            "id",
            "title",
            "kind",
            "state",
            "status_detail",
            "status_date",
            "mandatory",
            "owner",
            "risk_weight",
            "proof_weight",
            "weight_provenance",
            "sources",
            "depends_on",
            "closure_requires",
            "parent",
            "acceptance",
            "acceptance_evidence_required",
            "evidence_required",
            "evidence",
            "non_claims",
            "notes",
        }
    )

    @classmethod
    def from_dict(cls, data: Any) -> Requirement:
        _require(isinstance(data, dict), "requirement must be an object")
        raw_id = data.get("id", "<missing id>")
        name = f"requirement {raw_id!r}"
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"{name} has unknown fields: {sorted(unknown)}")
        missing = cls.ALLOWED_KEYS - set(data) - {"notes"}
        _require(not missing, f"{name} is missing fields: {sorted(missing)}")

        req_id = _check_str(data.get("id"), f"{name}.id")
        _require(
            bool(REQUIREMENT_ID_RE.match(req_id)),
            f"{name}.id does not match required pattern {REQUIREMENT_ID_RE.pattern}",
        )
        title = _check_str(data.get("title"), f"{name}.title")
        kind = _check_str(data.get("kind"), f"{name}.kind")
        _require(kind in KINDS, f"{name}.kind {kind!r} not in {KINDS}")
        state = _check_str(data.get("state"), f"{name}.state")
        _require(state in STATES, f"{name}.state {state!r} not in {STATES}")
        status_detail = _check_str(data.get("status_detail"), f"{name}.status_detail", allow_empty=True)
        status_date = _check_str(data.get("status_date"), f"{name}.status_date", allow_empty=True)
        if status_date:
            _require(
                bool(ISO_DATE_RE.match(status_date)),
                f"{name}.status_date must be an ISO date (YYYY-MM-DD)",
            )
        mandatory = data.get("mandatory")
        _require(isinstance(mandatory, bool), f"{name}.mandatory must be a boolean")
        owner = _check_str(data.get("owner"), f"{name}.owner")

        risk_weight = data.get("risk_weight")
        proof_weight = data.get("proof_weight")
        for weight_name, weight in (("risk_weight", risk_weight), ("proof_weight", proof_weight)):
            _require(
                isinstance(weight, (int, float)) and not isinstance(weight, bool),
                f"{name}.{weight_name} must be a number",
            )
            _require(float(weight) > 0.0, f"{name}.{weight_name} must be positive")
        weight_provenance = _check_str(data.get("weight_provenance"), f"{name}.weight_provenance")
        _require(
            weight_provenance in WEIGHT_PROVENANCES,
            f"{name}.weight_provenance {weight_provenance!r} not in {WEIGHT_PROVENANCES}",
        )

        sources_raw = data.get("sources")
        _require(isinstance(sources_raw, list), f"{name}.sources must be a list")
        sources = tuple(
            SourceRef.from_dict(item, f"{name}.sources[{index}]")
            for index, item in enumerate(sources_raw)
        )
        _require(bool(sources), f"{name}.sources must record at least one provenance entry")

        depends_on = tuple(_check_str_list(data.get("depends_on"), f"{name}.depends_on"))
        closure_requires = tuple(
            _check_str_list(data.get("closure_requires"), f"{name}.closure_requires")
        )
        for list_name, ids in (("depends_on", depends_on), ("closure_requires", closure_requires)):
            _require(
                len(set(ids)) == len(ids),
                f"{name}.{list_name} contains duplicate entries",
            )
            _require(
                req_id not in ids,
                f"{name}.{list_name} references the requirement itself",
            )

        parent = data.get("parent")
        if parent is not None:
            parent = _check_str(parent, f"{name}.parent")
            _require(parent != req_id, f"{name}.parent references the requirement itself")

        acceptance = tuple(_check_str_list(data.get("acceptance"), f"{name}.acceptance"))
        if kind == "atomic":
            _require(
                bool(acceptance),
                f"{name} is atomic and must declare at least one acceptance criterion",
            )
        acceptance_evidence_raw = data.get("acceptance_evidence_required")
        _require(
            isinstance(acceptance_evidence_raw, list),
            f"{name}.acceptance_evidence_required must be a list",
        )
        _require(
            len(acceptance_evidence_raw) == len(acceptance),
            f"{name}.acceptance_evidence_required must align one-to-one with acceptance",
        )
        acceptance_evidence_required: list[tuple[str, ...]] = []
        for index, raw_classes in enumerate(acceptance_evidence_raw):
            classes = tuple(
                _check_str_list(
                    raw_classes,
                    f"{name}.acceptance_evidence_required[{index}]",
                )
            )
            _require(
                bool(classes),
                f"{name}.acceptance_evidence_required[{index}] must be non-empty",
            )
            _require(
                len(set(classes)) == len(classes),
                f"{name}.acceptance_evidence_required[{index}] contains duplicates",
            )
            for cls_name in classes:
                _require(
                    cls_name in EVIDENCE_CLASSES,
                    f"{name}.acceptance_evidence_required[{index}] entry "
                    f"{cls_name!r} not in {EVIDENCE_CLASSES}",
                )
            expected_order = tuple(
                cls_name for cls_name in EVIDENCE_CLASSES if cls_name in classes
            )
            _require(
                classes == expected_order,
                f"{name}.acceptance_evidence_required[{index}] is not in canonical order",
            )
            acceptance_evidence_required.append(classes)
        evidence_required = tuple(
            _check_str_list(data.get("evidence_required"), f"{name}.evidence_required")
        )
        _require(bool(evidence_required), f"{name}.evidence_required must be non-empty")
        for cls_name in evidence_required:
            _require(
                cls_name in EVIDENCE_CLASSES,
                f"{name}.evidence_required entry {cls_name!r} not in {EVIDENCE_CLASSES}",
            )
        _require(
            len(set(evidence_required)) == len(evidence_required),
            f"{name}.evidence_required contains duplicates",
        )
        expected_union = tuple(
            cls_name
            for cls_name in EVIDENCE_CLASSES
            if any(cls_name in classes for classes in acceptance_evidence_required)
        )
        _require(
            evidence_required == expected_union,
            f"{name}.evidence_required must equal the canonical union of "
            "acceptance_evidence_required",
        )

        evidence_raw = data.get("evidence")
        _require(isinstance(evidence_raw, list), f"{name}.evidence must be a list")
        evidence = tuple(
            EvidenceRef.from_dict(item, f"{name}.evidence[{index}]")
            for index, item in enumerate(evidence_raw)
        )

        non_claims = tuple(_check_str_list(data.get("non_claims"), f"{name}.non_claims"))
        notes = data.get("notes", "")
        _require(isinstance(notes, str), f"{name}.notes must be a string")

        return cls(
            id=req_id,
            title=title,
            kind=kind,
            state=state,
            status_detail=status_detail,
            status_date=status_date,
            mandatory=mandatory,
            owner=owner,
            risk_weight=float(risk_weight),
            proof_weight=float(proof_weight),
            weight_provenance=weight_provenance,
            sources=sources,
            depends_on=depends_on,
            closure_requires=closure_requires,
            parent=parent,
            acceptance=acceptance,
            acceptance_evidence_required=tuple(acceptance_evidence_required),
            evidence_required=evidence_required,
            evidence=evidence,
            non_claims=non_claims,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "state": self.state,
            "status_detail": self.status_detail,
            "status_date": self.status_date,
            "mandatory": self.mandatory,
            "owner": self.owner,
            "risk_weight": self.risk_weight,
            "proof_weight": self.proof_weight,
            "weight_provenance": self.weight_provenance,
            "sources": [ref.to_dict() for ref in self.sources],
            "depends_on": list(self.depends_on),
            "closure_requires": list(self.closure_requires),
            "parent": self.parent,
            "acceptance": list(self.acceptance),
            "acceptance_evidence_required": [
                list(classes) for classes in self.acceptance_evidence_required
            ],
            "evidence_required": list(self.evidence_required),
            "evidence": [ref.to_dict() for ref in self.evidence],
            "non_claims": list(self.non_claims),
            "notes": self.notes,
        }

    def acceptance_ids(self) -> tuple[str, ...]:
        return tuple(f"A{index}" for index in range(1, len(self.acceptance) + 1))

    def required_evidence_for(self, acceptance_id: str) -> tuple[str, ...]:
        if not re.fullmatch(r"A[1-9][0-9]*", acceptance_id):
            raise RegistrySchemaError(f"invalid acceptance ID {acceptance_id!r}")
        index = int(acceptance_id[1:]) - 1
        if index >= len(self.acceptance_evidence_required):
            raise RegistrySchemaError(
                f"{self.id} has no acceptance unit {acceptance_id}"
            )
        return self.acceptance_evidence_required[index]

    def required_evidence_cells(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (acceptance_id, class_name)
            for acceptance_id, classes in zip(
                self.acceptance_ids(),
                self.acceptance_evidence_required,
                strict=True,
            )
            for class_name in classes
        )


@dataclass(frozen=True)
class GeneratedFrom:
    """Pins the exact inputs the registry was generated from.

    ``tracker_extraction_sha256`` is the hash of the *normative extraction*
    (parsed tables and numbered obligations), not the raw file, so narrative
    checkpoint prose can change without invalidating the registry — but any
    change to an ID, status, burden, or reference forces re-migration.
    """

    tracker_path: str
    tracker_extraction_sha256: str
    migration_rules_version: int

    ALLOWED_KEYS = frozenset(
        {"tracker_path", "tracker_extraction_sha256", "migration_rules_version"}
    )

    @classmethod
    def from_dict(cls, data: Any) -> GeneratedFrom:
        name = "generated_from"
        _require(isinstance(data, dict), f"{name} must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"{name} has unknown fields: {sorted(unknown)}")
        tracker_path = _check_str(data.get("tracker_path"), f"{name}.tracker_path")
        extraction = _check_str(
            data.get("tracker_extraction_sha256"), f"{name}.tracker_extraction_sha256"
        )
        _require(
            bool(SHA256_RE.match(extraction)),
            f"{name}.tracker_extraction_sha256 is not a sha256 hex digest",
        )
        rules_version = data.get("migration_rules_version")
        _require(
            isinstance(rules_version, int) and not isinstance(rules_version, bool),
            f"{name}.migration_rules_version must be an integer",
        )
        _require(rules_version >= 1, f"{name}.migration_rules_version must be >= 1")
        return cls(
            tracker_path=tracker_path,
            tracker_extraction_sha256=extraction,
            migration_rules_version=rules_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_path": self.tracker_path,
            "tracker_extraction_sha256": self.tracker_extraction_sha256,
            "migration_rules_version": self.migration_rules_version,
        }


@dataclass(frozen=True)
class Registry:
    """The full registry document."""

    schema_version: int
    registry_revision: int
    generated_from: GeneratedFrom
    requirements: tuple[Requirement, ...]
    content_sha256: str = field(default="", compare=False)

    ALLOWED_KEYS = frozenset(
        {
            "schema_version",
            "registry_revision",
            "generated_from",
            "requirements",
            "content_sha256",
        }
    )

    @classmethod
    def from_dict(cls, data: Any, *, verify_hash: bool = True) -> Registry:
        _require(isinstance(data, dict), "registry must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"registry has unknown fields: {sorted(unknown)}")
        schema_version = data.get("schema_version")
        _require(
            isinstance(schema_version, int) and not isinstance(schema_version, bool),
            "registry.schema_version must be an integer",
        )
        _require(
            schema_version == SCHEMA_VERSION,
            f"registry.schema_version {schema_version} != supported {SCHEMA_VERSION}",
        )
        registry_revision = data.get("registry_revision")
        _require(
            isinstance(registry_revision, int) and not isinstance(registry_revision, bool),
            "registry.registry_revision must be an integer",
        )
        _require(registry_revision >= 1, "registry.registry_revision must be >= 1")
        generated_from = GeneratedFrom.from_dict(data.get("generated_from"))
        requirements_raw = data.get("requirements")
        _require(isinstance(requirements_raw, list), "registry.requirements must be a list")
        requirements = tuple(Requirement.from_dict(item) for item in requirements_raw)
        ids = [req.id for req in requirements]
        _require(
            ids == sorted(ids),
            "registry.requirements must be sorted by id (canonical order)",
        )
        registry = cls(
            schema_version=schema_version,
            registry_revision=registry_revision,
            generated_from=generated_from,
            requirements=requirements,
            content_sha256=str(data.get("content_sha256", "")),
        )
        if verify_hash:
            recorded = data.get("content_sha256")
            _require(
                isinstance(recorded, str) and bool(SHA256_RE.match(recorded or "")),
                "registry.content_sha256 missing or malformed",
            )
            actual = registry.compute_content_sha256()
            _require(
                recorded == actual,
                "registry.content_sha256 does not match content: "
                f"recorded {recorded[:12]}…, actual {actual[:12]}… "
                "(the registry was edited without regeneration)",
            )
        return registry

    def body_dict(self) -> dict[str, Any]:
        """The canonical body: everything except the content hash itself."""
        return {
            "schema_version": self.schema_version,
            "registry_revision": self.registry_revision,
            "generated_from": self.generated_from.to_dict(),
            "requirements": [req.to_dict() for req in self.requirements],
        }

    def compute_content_sha256(self) -> str:
        canonical = json.dumps(self.body_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def declaration_sha256(self) -> str:
        """A digest over what the registry DECLARES, not over the file.

        ``content_sha256`` also covers ``registry_revision`` and the tracker
        extraction hash, which move whenever the tracker's prose moves. The
        evidence ledger binds to the registry so that a requirement cannot
        change under its evidence, and binding it to the file meant a reworded
        sentence in the tracker broke the ledger — repairable only by copying
        the new hash across, which checks nothing. This covers the
        requirements and the schema they are written in, so a reword leaves it
        alone and a changed obligation, acceptance cell or evidence class does
        not.
        """
        canonical = json.dumps(
            {
                "schema_version": self.schema_version,
                "requirements": [req.to_dict() for req in self.requirements],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        body = self.body_dict()
        body["content_sha256"] = self.compute_content_sha256()
        return body

    def to_canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    def by_id(self) -> dict[str, Requirement]:
        return {req.id: req for req in self.requirements}


def load_registry(path: Path, *, verify_hash: bool = True) -> Registry:
    """Load and strictly validate a registry file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RegistrySchemaError(f"registry file not found: {path}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegistrySchemaError(f"registry file is not valid JSON: {path}: {exc}") from exc
    return Registry.from_dict(data, verify_hash=verify_hash)


def write_registry_atomic(registry: Registry, path: Path) -> str:
    """Write the registry canonically and atomically; returns content hash.

    Write-to-temp-then-rename so a crash mid-write can never leave a torn
    registry: readers observe either the old complete file or the new one.
    """
    payload = registry.to_canonical_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        import os

        os.fsync(handle.fileno())
    tmp_path.replace(path)
    return registry.compute_content_sha256()
