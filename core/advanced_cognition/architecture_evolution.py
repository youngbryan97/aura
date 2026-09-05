"""Safe architecture evolution under proof obligations.

A promotion gate is only a gate if the thing it checks is hard to forge and
the thing it guards cannot move without it. Previously neither held: an
obligation was satisfied by the truthiness of ``evidence["passed"]``, tiering
was decided by substring matches on unresolved path strings, the plan
identifier did not bind the code it referred to, every record stayed mutable
after the decision, and "rollback" was a string.

CP126 65ecb1f1 / aedd615c / 33a4241b / a8481160 / 8ca18b5b / d8af3bdc.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Sequence

from .schemas import canonical_json, stable_hash

logger = logging.getLogger("Aura.ArchitectureEvolution")

#: Repository root this governor is allowed to reason about. Targets are
#: resolved and confined to it (CP126 aedd615c).
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

#: A 64-hex content digest, the one artifact form that needs no filesystem.
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

#: Marker recorded for a target that does not exist yet (a new module).
ABSENT_DIGEST = "absent"


class MutationTier(IntEnum):
    CONFIG = 0
    ADAPTER = 1
    FEATURE_MODULE = 2
    SHARED_RUNTIME = 3
    GOVERNANCE_OR_IDENTITY = 4
    SEALED_CORE = 5


# --------------------------------------------------------------------------
# Immutability (CP126 8ca18b5b)
# --------------------------------------------------------------------------


class FrozenMap(Mapping):
    """A read-only mapping whose values are themselves frozen.

    ``promotable`` was previously a property over mutable dataclasses holding
    mutable nested dictionaries, so evidence could be edited after the
    decision was taken and any serialized receipt disagreed with the object.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {str(k): freeze(v) for k, v in dict(data or {}).items()}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FrozenMap({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(canonical_json(thaw(self)))


def freeze(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze(item) for item in value), key=repr))
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value


# --------------------------------------------------------------------------
# Proof obligations (CP126 65ecb1f1)
# --------------------------------------------------------------------------

#: Every obligation must produce these. CP126 65ecb1f1: ``passed`` alone —
#: and by truthiness, so the string "no" satisfied it — let a caller-built
#: dictionary promote a change with no test artifact, no verifier identity and
#: nothing tying the result to the code under test.
BASE_EVIDENCE_KEYS = ("passed", "artifact", "verifier", "subject_digest")

#: Governance-tier and above additionally have to name the trust root that
#: vouches for the verifier.
TRUST_ROOT_KEY = "trust_root"


@dataclass(frozen=True)
class ProofObligation:
    name: str
    description: str
    required: bool = True
    evidence: FrozenMap = field(default_factory=FrozenMap)
    #: Which evidence keys this obligation demands, and what the evidence has
    #: to be *about*.
    required_evidence: tuple[str, ...] = BASE_EVIDENCE_KEYS
    subject_digest: str = ""
    repo_root: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", freeze(self.evidence))

    @property
    def defects(self) -> tuple[str, ...]:
        """Every reason this obligation is not met. Empty means satisfied."""
        if not self.required:
            return ()
        evidence = self.evidence
        problems: list[str] = []

        for key in self.required_evidence:
            if key not in evidence:
                problems.append(f"missing evidence key: {key}")

        passed = evidence.get("passed")
        if "passed" in self.required_evidence and passed is not True:
            # Strictly True. A truthy string or a non-empty dict is a caller
            # asserting success, not a verifier reporting it.
            problems.append(f"passed is {passed!r}, not the boolean True")

        if "artifact" in self.required_evidence:
            problems.extend(self._artifact_defects(evidence.get("artifact")))

        if "verifier" in self.required_evidence:
            verifier = evidence.get("verifier")
            if not isinstance(verifier, str) or not verifier.strip():
                problems.append("verifier identity is missing or not a string")

        if "subject_digest" in self.required_evidence:
            claimed = evidence.get("subject_digest")
            if not isinstance(claimed, str) or not claimed.strip():
                problems.append("subject_digest is missing")
            elif self.subject_digest and claimed != self.subject_digest:
                problems.append(
                    "subject_digest does not match the planned targets "
                    f"({claimed[:12]}… != {self.subject_digest[:12]}…)"
                )

        if TRUST_ROOT_KEY in self.required_evidence:
            root = evidence.get(TRUST_ROOT_KEY)
            if not isinstance(root, str) or not root.strip():
                problems.append("trust_root is missing for a governance-tier obligation")

        return tuple(dict.fromkeys(problems))

    def _artifact_defects(self, artifact: Any) -> list[str]:
        """An artifact must actually resolve to something inspectable."""
        if isinstance(artifact, Mapping):
            digest = artifact.get("digest")
            if isinstance(digest, str) and _DIGEST_RE.match(digest.strip().lower()):
                return []
            return ["artifact mapping carries no sha256 digest"]
        if not isinstance(artifact, str) or not artifact.strip():
            return ["artifact is missing"]
        text = artifact.strip()
        if _DIGEST_RE.match(text.lower()):
            return []
        root = Path(self.repo_root) if self.repo_root else DEFAULT_REPO_ROOT
        candidate = Path(text)
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            return [f"artifact path escapes the repository: {text}"]
        if not resolved.is_file():
            return [f"artifact does not resolve to a file: {text}"]
        return []

    @property
    def satisfied(self) -> bool:
        return not self.defects

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "evidence": thaw(self.evidence),
            "required_evidence": list(self.required_evidence),
            "subject_digest": self.subject_digest,
            "satisfied": self.satisfied,
            "defects": list(self.defects),
        }


@dataclass(frozen=True)
class ArchitectureMutationPlan:
    plan_id: str
    target_paths: tuple[str, ...]
    tier: MutationTier
    summary: str
    obligations: tuple[ProofObligation, ...]
    rollback_strategy: str
    sealed: bool = False
    #: Canonical repo-relative targets and the digest of each at plan time.
    #: CP126 a8481160: without these the plan named path *strings* and could
    #: be satisfied against code it had never seen.
    source_digests: FrozenMap = field(default_factory=FrozenMap)
    subject_digest: str = ""
    #: Targets that could not be canonicalized/confined, with the reason.
    rejected_targets: tuple[FrozenMap, ...] = ()
    certificate_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligations", tuple(self.obligations))
        object.__setattr__(self, "source_digests", freeze(self.source_digests))
        object.__setattr__(self, "rejected_targets", tuple(freeze(self.rejected_targets)))
        if not self.certificate_id:
            object.__setattr__(self, "certificate_id", self._compute_certificate_id())

    def _compute_certificate_id(self) -> str:
        """Identity of *this decision*, evidence included.

        CP126 a8481160: ``plan_id`` deliberately stays the identity of the
        proposal, so it is stable across evaluation; the certificate id moves
        whenever the evidence or the verdict moves.
        """
        return stable_hash(
            {
                "plan": self.plan_id,
                "obligations": [
                    {
                        "name": o.name,
                        "evidence": thaw(o.evidence),
                        "satisfied": o.satisfied,
                    }
                    for o in self.obligations
                ],
                "promotable": self.promotable,
                "sealed": self.sealed,
            },
            prefix="archcert_",
        )

    @property
    def promotable(self) -> bool:
        if self.sealed or self.rejected_targets or not self.target_paths:
            return False
        return all(o.satisfied for o in self.obligations)

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.sealed:
            reasons.append("target is in the sealed core; human escalation required")
        if not self.target_paths:
            reasons.append("no usable target paths")
        for rejected in self.rejected_targets:
            reasons.append(f"rejected target {rejected.get('path')!r}: {rejected.get('reason')}")
        for obligation in self.obligations:
            for defect in obligation.defects:
                reasons.append(f"{obligation.name}: {defect}")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "certificate_id": self.certificate_id,
            "target_paths": list(self.target_paths),
            "tier": int(self.tier),
            "tier_name": self.tier.name.lower(),
            "summary": self.summary,
            "obligations": [o.to_dict() for o in self.obligations],
            "rollback_strategy": self.rollback_strategy,
            "sealed": self.sealed,
            "source_digests": thaw(self.source_digests),
            "subject_digest": self.subject_digest,
            "rejected_targets": [thaw(item) for item in self.rejected_targets],
            "promotable": self.promotable,
            "blocking_reasons": list(self.blocking_reasons),
        }


class ArchitectureEvolutionGovernor:
    """Assigns mutation tiers and promotion gates for self-modification."""

    #: Paths whose mutation is never autonomous. Matched on the canonical
    #: repo-relative path — exactly, or as a directory prefix. CP126 aedd615c:
    #: substring matching on raw strings both over-classified (any path merely
    #: *containing* "core/security") and under-classified (traversal, "./",
    #: backslashes, symlinks, case differences on a case-insensitive volume).
    SEALED_TARGETS = (
        "core/will.py",
        "core/executive/authority_gateway.py",
        "core/governance",
        "core/governance_context.py",
        "core/constitution",
        "core/security",
        "core/runtime/atomic_writer.py",
        "core/runtime/file_write_gateway.py",
    )

    #: Identity- and authority-shaped code: governed, but not sealed.
    GOVERNANCE_TARGETS = (
        "core/sovereignty",
        "core/identity",
        "core/persona",
        "core/heartstone",
        "core/executive",
        "core/consciousness",
        "core/brain/persona_adapter.py",
        "core/brain/personality_bridge.py",
    )

    SHARED_RUNTIME_TARGETS = (
        "core/runtime",
        "core/memory",
        "core/environment",
        "core/reasoning",
        "core/bus",
        "core/brain/llm",
    )

    CONFIG_SUFFIXES = (".toml", ".json", ".yaml", ".yml", ".ini", ".cfg", ".env")

    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or DEFAULT_REPO_ROOT).resolve()

    # -- planning ---------------------------------------------------------

    def plan_mutation(
        self,
        *,
        target_paths: Sequence[str],
        summary: str,
        evidence: Mapping[str, Any] | None = None,
    ) -> ArchitectureMutationPlan:
        canonical: list[str] = []
        rejected: list[dict[str, Any]] = []
        for raw in target_paths or ():
            relative, fault = self._canonical_target(raw)
            if fault:
                rejected.append({"path": str(raw), "reason": fault})
            else:
                canonical.append(relative)
        canonical = list(dict.fromkeys(canonical))

        tier = self._tier(canonical)
        # A target we could not canonicalize is treated at the strongest tier
        # we can justify: we do not know what it points at.
        if rejected and tier < MutationTier.GOVERNANCE_OR_IDENTITY:
            tier = MutationTier.GOVERNANCE_OR_IDENTITY

        digests = {path: self._digest_of(path) for path in canonical}
        subject_digest = self._subject_digest(digests)
        sealed = tier >= MutationTier.SEALED_CORE
        obligations = self._obligations(tier, evidence or {}, subject_digest, self.repo_root)

        return ArchitectureMutationPlan(
            plan_id=stable_hash(
                {
                    "paths": tuple(canonical),
                    "digests": digests,
                    "summary": summary,
                    "tier": int(tier),
                    "obligations": tuple(o.name for o in obligations),
                },
                prefix="arch_",
            ),
            target_paths=tuple(canonical),
            tier=tier,
            summary=summary,
            obligations=obligations,
            rollback_strategy="ghost_boot_then_atomic_promote_with_tombstone",
            sealed=sealed,
            source_digests=FrozenMap(digests),
            subject_digest=subject_digest,
            rejected_targets=tuple(FrozenMap(item) for item in rejected),
        )

    def evaluate_promotion(
        self,
        plan: ArchitectureMutationPlan,
        evidence: Mapping[str, Any],
    ) -> ArchitectureMutationPlan:
        refreshed = tuple(
            ProofObligation(
                obligation.name,
                obligation.description,
                obligation.required,
                freeze(evidence.get(obligation.name, thaw(obligation.evidence))),
                obligation.required_evidence,
                obligation.subject_digest,
                obligation.repo_root,
            )
            for obligation in plan.obligations
        )
        return ArchitectureMutationPlan(
            plan_id=plan.plan_id,
            target_paths=plan.target_paths,
            tier=plan.tier,
            summary=plan.summary,
            obligations=refreshed,
            rollback_strategy=plan.rollback_strategy,
            sealed=plan.sealed,
            source_digests=plan.source_digests,
            subject_digest=plan.subject_digest,
            rejected_targets=plan.rejected_targets,
        )

    # -- execution gateway (CP126 d8af3bdc) --------------------------------

    def promote(
        self,
        plan: ArchitectureMutationPlan,
        evidence: Mapping[str, Any] | None = None,
        *,
        apply: Callable[[dict[str, str]], Any] | None = None,
        verify: Callable[[dict[str, str]], bool] | None = None,
    ) -> dict[str, Any]:
        """Apply a mutation only through a certified, reversible transaction.

        The governor previously produced plans and never touched the act it
        governed: there was no gateway, no check that a caller held a
        promotable plan, no compare-and-swap against the code the plan was
        certified over, and rollback was the literal string
        ``"ghost_boot_then_atomic_promote_with_tombstone"``.
        """
        certified = self.evaluate_promotion(plan, evidence) if evidence is not None else plan
        receipt: dict[str, Any] = {
            "plan_id": certified.plan_id,
            "certificate_id": certified.certificate_id,
            "tier": certified.tier.name.lower(),
            "promoted": False,
            "rolled_back": False,
            "reasons": list(certified.blocking_reasons),
        }
        if not certified.promotable:
            receipt["status"] = "refused"
            logger.info(
                "Architecture promotion refused for %s: %s",
                certified.plan_id,
                "; ".join(receipt["reasons"][:4]) or "not promotable",
            )
            return receipt

        # Compare-and-swap: the evidence certified a specific tree state.
        drifted = [
            path
            for path in certified.target_paths
            if self._digest_of(path) != certified.source_digests.get(path)
        ]
        if drifted:
            receipt["status"] = "refused"
            receipt["reasons"].append(
                "targets changed since certification: " + ", ".join(sorted(drifted))
            )
            return receipt

        if apply is None:
            receipt["status"] = "certified"
            receipt["reasons"] = []
            return receipt

        originals = self._capture(certified.target_paths)
        try:
            apply({path: str(self.repo_root / path) for path in certified.target_paths})
        except Exception as exc:  # noqa: BLE001 - restored, re-reported below
            self._restore(originals)
            receipt["status"] = "rolled_back"
            receipt["rolled_back"] = True
            receipt["reasons"].append(f"apply raised {type(exc).__name__}: {exc}")
            return receipt

        after = {path: self._digest_of(path) for path in certified.target_paths}
        if after == thaw(certified.source_digests):
            receipt["status"] = "no_op"
            receipt["reasons"].append("apply changed nothing on disk")
            return receipt

        if verify is not None:
            try:
                verified = bool(verify({path: str(self.repo_root / path) for path in certified.target_paths}))
            except Exception as exc:  # noqa: BLE001 - a failed check is a failure
                verified = False
                receipt["reasons"].append(f"verify raised {type(exc).__name__}: {exc}")
            if not verified:
                self._restore(originals)
                receipt["status"] = "rolled_back"
                receipt["rolled_back"] = True
                receipt["reasons"].append("post-promotion verification failed")
                return receipt

        receipt["status"] = "promoted"
        receipt["promoted"] = True
        receipt["reasons"] = []
        receipt["post_digests"] = after
        return receipt

    def rollback(self, originals: Mapping[str, bytes | None]) -> dict[str, Any]:
        """Public rollback: restore captured contents and report the result."""
        restored, failures = self._restore(originals)
        return {"restored": restored, "failures": failures, "ok": not failures}

    def _capture(self, paths: Sequence[str]) -> dict[str, bytes | None]:
        captured: dict[str, bytes | None] = {}
        for path in paths:
            target = self.repo_root / path
            captured[path] = target.read_bytes() if target.is_file() else None
        return captured

    def _restore(self, originals: Mapping[str, bytes | None]) -> tuple[list[str], list[str]]:
        from core.runtime.atomic_writer import atomic_write_bytes

        restored: list[str] = []
        failures: list[str] = []
        for path, payload in originals.items():
            target = self.repo_root / path
            try:
                if payload is None:
                    # The file did not exist before; a created file is undone
                    # by removing it, not by writing b"".
                    if target.is_file():
                        target.unlink()
                else:
                    atomic_write_bytes(target, payload)
                restored.append(path)
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append(f"{path}: {exc}")
                logger.error("Architecture rollback failed for %s: %s", path, exc)
        return restored, failures

    # -- classification (CP126 aedd615c / 33a4241b) ------------------------

    def _canonical_target(self, raw: Any) -> tuple[str, str]:
        """Canonical repo-relative POSIX path, or ('', reason)."""
        text = str(raw or "").strip()
        if not text:
            return "", "empty path"
        if "\x00" in text:
            return "", "path contains a NUL byte"
        # Windows-style separators are not a different namespace here.
        text = text.replace("\\", "/")
        candidate = Path(text)
        absolute = candidate if candidate.is_absolute() else self.repo_root / candidate
        try:
            # Resolve symlinks BEFORE the containment check, or a link inside
            # the tree can point anywhere.
            resolved = absolute.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return "", f"path could not be resolved: {exc}"
        try:
            relative = resolved.relative_to(self.repo_root)
        except ValueError:
            return "", "path escapes the repository root"
        if not str(relative) or str(relative) == ".":
            return "", "path resolves to the repository root itself"
        return relative.as_posix(), ""

    @staticmethod
    def _matches(relative: str, targets: Sequence[str]) -> bool:
        """Exact or directory-prefix match, case-folded.

        macOS volumes are case-insensitive, so ``Core/Will.py`` names the same
        file as ``core/will.py`` and must classify the same way.
        """
        needle = relative.casefold()
        for target in targets:
            folded = target.casefold().rstrip("/")
            if needle == folded or needle.startswith(folded + "/"):
                return True
        return False

    def _tier(self, paths: Sequence[str]) -> MutationTier:
        """The strongest tier any target demands."""
        tier = MutationTier.ADAPTER
        for path in paths:
            tier = max(tier, self._tier_for_path(path))
        return tier

    def _tier_for_path(self, path: str) -> MutationTier:
        if self._matches(path, self.SEALED_TARGETS):
            # CP126 33a4241b: sealed paths were classified GOVERNANCE_OR_IDENTITY
            # and SEALED_CORE was defined but never returned by anything.
            return MutationTier.SEALED_CORE
        if self._matches(path, self.GOVERNANCE_TARGETS):
            return MutationTier.GOVERNANCE_OR_IDENTITY
        if self._matches(path, self.SHARED_RUNTIME_TARGETS):
            return MutationTier.SHARED_RUNTIME
        if self._matches(path, ("core",)):
            # Anything under core/ is at least a feature module, including a
            # config file that lives there.
            return MutationTier.FEATURE_MODULE
        if path.casefold().endswith(self.CONFIG_SUFFIXES):
            return MutationTier.CONFIG
        return MutationTier.ADAPTER

    # -- digests ----------------------------------------------------------

    def _digest_of(self, relative: str) -> str:
        target = self.repo_root / relative
        try:
            if not target.is_file():
                return ABSENT_DIGEST
            return hashlib.sha256(target.read_bytes()).hexdigest()
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not digest %s: %s", relative, exc)
            return f"unreadable:{type(exc).__name__}"

    @staticmethod
    def _subject_digest(digests: Mapping[str, str]) -> str:
        return hashlib.sha256(
            canonical_json({k: digests[k] for k in sorted(digests)}).encode("utf-8")
        ).hexdigest()

    # -- obligations ------------------------------------------------------

    @staticmethod
    def _obligations(
        tier: MutationTier,
        evidence: Mapping[str, Any],
        subject_digest: str,
        repo_root: Path,
    ) -> tuple[ProofObligation, ...]:
        names = [
            ("unit_tests", "Focused unit tests pass"),
            ("hidden_tests", "Hidden or held-out tests pass"),
            ("proof_substrate", "Artifact/tool/env graph is reproducible"),
            ("rollback", "Rollback path is available and tested"),
        ]
        if tier >= MutationTier.SHARED_RUNTIME:
            names.extend(
                [
                    ("integration_tests", "Runtime integration tests pass"),
                    ("soak_or_replay", "Replay/soak evidence shows no regression"),
                    ("stability_canaries", "Identity/governance canaries hold"),
                ]
            )
        if tier >= MutationTier.GOVERNANCE_OR_IDENTITY:
            names.extend(
                [
                    ("formal_or_static_proof", "Static proof rejects bypasses"),
                    ("human_review", "Human escalation is required for sealed governance mutation"),
                ]
            )
        required_keys = BASE_EVIDENCE_KEYS + (
            (TRUST_ROOT_KEY,) if tier >= MutationTier.GOVERNANCE_OR_IDENTITY else ()
        )
        return tuple(
            ProofObligation(
                name,
                description,
                True,
                freeze(evidence.get(name, {})),
                required_keys,
                subject_digest,
                str(repo_root),
            )
            for name, description in names
        )
