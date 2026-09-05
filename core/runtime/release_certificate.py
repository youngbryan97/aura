"""What a release is known to survive — and what it is not.

Aura has a large test corpus. Raw test count is not the remaining maturity
problem; the problem is that "the current checkout worked during this run"
is not the same claim as "this release survives its declared operating
envelope", and only the second one is worth shipping on.

The declared envelope is not the suite. It is the 64GB deployment profile,
long conversations, background cognition under foreground load, model
death and reload, memory-corruption recovery, tool failures, real desktop
task sequences, install/upgrade/rollback, and multi-day continuity. A
green suite says nothing about most of those.

So this builds ONE certificate from evidence that was actually produced,
and its defining behaviour is that it **refuses**. A requirement with no
evidence is ``MISSING`` and blocks certification — it never quietly counts
as satisfied, and it is never inferred from a neighbouring requirement
passing. That is the whole point: this codebase's recurring defect is the
absence of a check reported as a passed check, and a release certificate
is exactly the artifact where that mistake would do the most damage.

Evidence goes stale. A shard result from three weeks and forty commits ago
is not evidence about this build, so every piece of evidence names the
commit it was produced at, and evidence from a different commit is
``STALE`` rather than valid.
"""
from __future__ import annotations

import enum
import json
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import runtime_identity, state_root
from core.runtime.subprocess_gateway import get_subprocess_gateway

__all__ = [
    "Requirement",
    "RequirementStatus",
    "Evidence",
    "ReleaseCertificate",
    "CertificateBuilder",
    "REQUIREMENTS",
    "current_commit",
]


class RequirementStatus(str, enum.Enum):
    SATISFIED = "satisfied"
    FAILED = "failed"
    #: No evidence was submitted. Blocks certification. Distinct from
    #: FAILED because "we did not look" and "we looked and it broke" call
    #: for different responses, and merging them hides which one happened.
    MISSING = "missing"
    #: Evidence exists but was produced at a different commit.
    STALE = "stale"
    #: Declared not applicable, with a reason, by a named person.
    WAIVED = "waived"


@dataclass(frozen=True)
class Requirement:
    """One thing a release must be shown to survive."""

    key: str
    description: str
    #: False for requirements that are genuinely advisory. Kept small on
    #: purpose — a certificate where most requirements are optional
    #: certifies nothing.
    blocking: bool = True


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement("hermetic_shards", "All test shards pass hermetically"),
    Requirement(
        "no_order_dependence",
        "No test passes alone and fails in-chunk (an order-dependence defect)",
    ),
    Requirement("conversation_soak", "A 200-turn conversation soak completes"),
    Requirement(
        "contention_soak",
        "Background cognition under sustained foreground load, multi-hour",
    ),
    Requirement("chaos_injection", "Injected faults are survived, not merely logged"),
    Requirement("memory_ceiling", "Stays inside the 64GB deployment profile"),
    Requirement(
        "worker_cancellation",
        "A generation can be cancelled and the worker recovers with weights warm",
    ),
    Requirement("model_death_reload", "Model death and reload is survived"),
    Requirement("user_journeys", "Ten canonical real user journeys complete"),
    Requirement("install_upgrade_rollback", "Install, upgrade, rollback and uninstall"),
    Requirement("latency_profile", "p50, p95 and worst-case latency recorded"),
    Requirement(
        "no_blank_responses",
        "Zero unexplained user-facing blank responses",
    ),
    Requirement(
        "learning_evidence",
        "Zero durable learning updates admitted without sufficient evidence",
    ),
)

_BY_KEY = {requirement.key: requirement for requirement in REQUIREMENTS}


@dataclass(frozen=True)
class Evidence:
    """A claim about one requirement, tied to the commit that produced it."""

    key: str
    passed: bool
    commit: str
    at: float
    #: Free-form measurements: counts, latencies, peak RSS. Recorded so a
    #: reader can disagree with the verdict rather than only accept it.
    detail: Mapping[str, Any] = field(default_factory=dict)
    produced_by: str = "unknown"


@dataclass(frozen=True)
class RequirementResult:
    requirement: Requirement
    status: RequirementStatus
    note: str
    evidence: Evidence | None = None

    @property
    def blocks_release(self) -> bool:
        return self.requirement.blocking and self.status in (
            RequirementStatus.FAILED,
            RequirementStatus.MISSING,
            RequirementStatus.STALE,
        )


@dataclass(frozen=True)
class ReleaseCertificate:
    """The artifact. Either it certifies, or it says exactly why it cannot."""

    commit: str
    certified: bool
    built_at: float
    results: tuple[RequirementResult, ...]
    runtime: Mapping[str, Any]

    @property
    def blocking_failures(self) -> tuple[RequirementResult, ...]:
        return tuple(result for result in self.results if result.blocks_release)

    def summary(self) -> str:
        if self.certified:
            return f"CERTIFIED at {self.commit[:12]}"
        reasons = ", ".join(
            f"{result.requirement.key}={result.status.value}"
            for result in self.blocking_failures
        )
        return f"NOT CERTIFIED at {self.commit[:12]}: {reasons}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "certified": self.certified,
            "built_at": self.built_at,
            "summary": self.summary(),
            "runtime": dict(self.runtime),
            "requirements": [
                {
                    "key": result.requirement.key,
                    "description": result.requirement.description,
                    "blocking": result.requirement.blocking,
                    "status": result.status.value,
                    "note": result.note,
                    "produced_by": result.evidence.produced_by if result.evidence else None,
                    "evidence_commit": result.evidence.commit if result.evidence else None,
                    "detail": dict(result.evidence.detail) if result.evidence else {},
                }
                for result in self.results
            ],
        }


def current_commit() -> str:
    """The commit being certified.

    Returns ``"unknown"`` rather than raising — and an unknown commit makes
    every piece of evidence STALE, so a certificate can never be issued
    for a build nobody can identify.
    """
    try:
        result = get_subprocess_gateway().run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            read_only=True,
            text=True,
            timeout=10,
            check=False,
            source="runtime.release_certificate.current_commit",
            accelerator_capability="none",
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except (OSError, subprocess.SubprocessError) as exc:
        record_degradation(
            "release_certificate",
            exc,
            severity="warning",
            action="could not identify the commit; all evidence will read as stale",
        )
        return "unknown"


class CertificateBuilder:
    """Collects evidence, then issues a verdict that refuses by default."""

    def __init__(self, *, commit: str | None = None) -> None:
        self.commit = str(commit or current_commit())
        self._evidence: dict[str, Evidence] = {}
        self._waivers: dict[str, tuple[str, str]] = {}

    def submit(
        self,
        key: str,
        *,
        passed: bool,
        detail: Mapping[str, Any] | None = None,
        commit: str | None = None,
        produced_by: str = "unknown",
    ) -> None:
        """Record evidence for one requirement.

        A key that is not a declared requirement is rejected outright: a
        certificate that accepted arbitrary keys could be padded with
        evidence for things nobody asked about while a real requirement
        stayed missing.
        """
        if key not in _BY_KEY:
            raise KeyError(
                f"{key!r} is not a release requirement; declared: {sorted(_BY_KEY)}"
            )
        if not isinstance(passed, bool):
            raise TypeError("passed must be a bool; a truthy string is not a result")
        self._evidence[key] = Evidence(
            key=key,
            passed=passed,
            commit=str(commit or self.commit),
            at=time.time(),
            detail=dict(detail or {}),
            produced_by=str(produced_by),
        )

    def waive(self, key: str, *, reason: str, waived_by: str) -> None:
        """Declare a requirement not applicable to this release.

        Requires a named person and a reason. An anonymous waiver is
        indistinguishable from a requirement quietly deleted, so both are
        refused.
        """
        if key not in _BY_KEY:
            raise KeyError(f"{key!r} is not a release requirement")
        if not str(reason or "").strip():
            raise ValueError("a waiver requires a reason")
        if not str(waived_by or "").strip():
            raise ValueError("a waiver requires a named person")
        self._waivers[key] = (str(reason), str(waived_by))

    def build(self) -> ReleaseCertificate:
        """Issue the certificate. Certifies only when nothing blocks."""
        results: list[RequirementResult] = []
        for requirement in REQUIREMENTS:
            waiver = self._waivers.get(requirement.key)
            if waiver is not None:
                reason, waived_by = waiver
                results.append(
                    RequirementResult(
                        requirement=requirement,
                        status=RequirementStatus.WAIVED,
                        note=f"waived by {waived_by}: {reason}",
                    )
                )
                continue

            evidence = self._evidence.get(requirement.key)
            if evidence is None:
                results.append(
                    RequirementResult(
                        requirement=requirement,
                        status=RequirementStatus.MISSING,
                        note="no evidence submitted; this is not a pass",
                    )
                )
                continue
            if evidence.commit != self.commit or self.commit == "unknown":
                results.append(
                    RequirementResult(
                        requirement=requirement,
                        status=RequirementStatus.STALE,
                        note=(
                            f"evidence was produced at {evidence.commit[:12]}, "
                            f"this build is {self.commit[:12]}"
                        ),
                        evidence=evidence,
                    )
                )
                continue
            results.append(
                RequirementResult(
                    requirement=requirement,
                    status=(
                        RequirementStatus.SATISFIED
                        if evidence.passed
                        else RequirementStatus.FAILED
                    ),
                    note=f"produced by {evidence.produced_by}",
                    evidence=evidence,
                )
            )

        frozen = tuple(results)
        certified = not any(result.blocks_release for result in frozen)
        return ReleaseCertificate(
            commit=self.commit,
            certified=certified,
            built_at=time.time(),
            results=frozen,
            runtime=runtime_identity(),
        )


async def write_certificate_async(
    certificate: ReleaseCertificate, *, path: Path | None = None
) -> str | None:
    """Persist a certificate through the governed write gateway."""
    target = path or (state_root() / "release" / f"certificate-{certificate.commit[:12]}.json")
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("release_certificate", domain="state_mutation"):
            return await get_file_write_gateway().write_json_async(
                target, certificate.to_dict(), source="release_certificate"
            )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "release_certificate",
            exc,
            severity="degraded",
            action="built the certificate but could not persist it",
        )
        return None


def load_certificate(path: Path | str) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except (OSError, ValueError):
        return None


def summarize_certificates(certificates: Iterable[ReleaseCertificate]) -> dict[str, Any]:
    entries = list(certificates)
    return {
        "certificates": len(entries),
        "certified": sum(1 for entry in entries if entry.certified),
        "latest": entries[-1].summary() if entries else None,
    }
