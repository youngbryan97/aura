"""SkillContract + SkillVerifier framework.

Every Aura skill, per the audit, must declare:

    name, version, inputs, outputs, preconditions, postconditions,
    required_tools, required_permissions, timeout_seconds, retry_policy,
    rollback_supported, verifier, benchmark, memory_policy,
    autonomy_level_required

Skill execution must yield a typed ``SkillExecutionResult`` whose status
distinguishes success_verified / success_unverified / partial_success /
failed_recoverable / failed_fatal / blocked_by_policy / needs_human_approval.

A skill without a registered verifier is recorded as ``unverified`` and is
flagged by the conformance suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence


class SkillStatus(str, Enum):
    SUCCESS_VERIFIED = "success_verified"
    SUCCESS_UNVERIFIED = "success_unverified"
    PARTIAL_SUCCESS = "partial_success"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_FATAL = "failed_fatal"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    NEEDS_HUMAN_APPROVAL = "needs_human_approval"


@dataclass
class SkillContract:
    name: str
    version: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    retry_policy: str = "none"
    rollback_supported: bool = False
    verifier: Optional[str] = None
    benchmark: Optional[str] = None
    memory_policy: str = "session"
    autonomy_level_required: int = 0


@dataclass
class SkillExecutionResult:
    skill: str
    status: SkillStatus
    output: Any = None
    receipt_id: Optional[str] = None
    verification_evidence: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == SkillStatus.SUCCESS_VERIFIED


class VerifierMissing(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillRegistry:
    def __init__(self):
        self._contracts: Dict[str, SkillContract] = {}
        self._verifiers: Dict[str, Callable[[SkillExecutionResult], SkillExecutionResult]] = {}

    def register(self, contract: SkillContract) -> None:
        self._contracts[contract.name] = contract

    def register_verifier(
        self,
        name: str,
        verifier: Callable[[SkillExecutionResult], SkillExecutionResult],
    ) -> None:
        self._verifiers[name] = verifier

    def get(self, name: str) -> Optional[SkillContract]:
        return self._contracts.get(name)

    def all(self) -> Sequence[SkillContract]:
        return list(self._contracts.values())

    def verify(self, result: SkillExecutionResult) -> SkillExecutionResult:
        verifier = self._verifiers.get(result.skill)
        if verifier is None:
            if result.status == SkillStatus.SUCCESS_VERIFIED:
                # Cannot self-verify without a verifier present.
                return SkillExecutionResult(
                    skill=result.skill,
                    status=SkillStatus.SUCCESS_UNVERIFIED,
                    output=result.output,
                    receipt_id=result.receipt_id,
                    failure_reason="no verifier registered",
                )
            return result
        return verifier(result)

    def unverified_skills(self) -> List[str]:
        return [name for name in self._contracts if name not in self._verifiers]


_global_skills: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _global_skills
    if _global_skills is None:
        _global_skills = SkillRegistry()
    return _global_skills


def reset_skill_registry() -> None:
    global _global_skills
    _global_skills = None
