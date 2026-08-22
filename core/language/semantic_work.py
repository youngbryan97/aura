"""Typed work Aura inferred from one visible utterance.

This is an orchestration surface of the existing language substrate, not a
second interpreter.  ``PromptShape`` already preserves the individual asks in
a turn and the learned matchers can refine semantic decisions as receipts
accumulate.  What was missing was one typed object that downstream cognition
could consume without re-reading the prose differently in every phase.

The first contract is deliberately structural.  It decides only how much work
the answer surface must carry and whether that work belongs in the current
reply.  It does not select an external tool, classify a knowledge domain, or
ask a model to describe the request.  New learned decisions can therefore
amend this object after they clear the substrate's measured admission bar,
while an unseen wording keeps a deterministic, fail-open floor today.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from core.runtime.skill_task_bridge import looks_like_inline_answer_request
from core.runtime.structured_input import (
    analyze_prompt_shape,
    answer_surface_planning_tokens,
    answer_surface_token_floor,
)

__all__ = [
    "INLINE_REPLY",
    "NON_INLINE",
    "SemanticWorkContract",
    "build_semantic_work_contract",
]

INLINE_REPLY = "inline_reply"
NON_INLINE = "non_inline"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SemanticWorkContract:
    """Mechanically derived answer work shared by routing and generation."""

    delivery_mode: str = NON_INLINE
    obligations: tuple[str, ...] = ()
    obligation_count: int = 1
    requires_complete_reply: bool = False
    requires_deliberation: bool = False
    architecture_assistance_eligible: bool = False
    answer_token_floor: int = 256
    planning_token_estimate: int = 192
    decision_basis: tuple[str, ...] = ()
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SemanticWorkContract:
        raw = dict(payload or {})
        return cls(
            delivery_mode=str(raw.get("delivery_mode") or NON_INLINE),
            obligations=tuple(str(item) for item in (raw.get("obligations") or ())),
            obligation_count=max(1, int(raw.get("obligation_count") or 1)),
            requires_complete_reply=bool(raw.get("requires_complete_reply", False)),
            requires_deliberation=bool(raw.get("requires_deliberation", False)),
            architecture_assistance_eligible=bool(
                raw.get("architecture_assistance_eligible", False)
            ),
            answer_token_floor=max(1, int(raw.get("answer_token_floor") or 256)),
            planning_token_estimate=max(
                1, int(raw.get("planning_token_estimate") or 192)
            ),
            decision_basis=tuple(
                str(item) for item in (raw.get("decision_basis") or ())
            ),
            schema_version=int(raw.get("schema_version") or _SCHEMA_VERSION),
        )


def build_semantic_work_contract(text: str) -> SemanticWorkContract:
    """Return the current turn's typed answer-work contract.

    Deliberation is admitted by observable work, not a subject-name list.  A
    reply that must satisfy at least three independent obligations, or whose
    existing capacity estimator crosses the larger answer surface, earns the
    deliberate lane.  A short explanation remains reactive.  External effects
    never become architecture-assisted inline answers merely because they have
    several steps.
    """

    objective = str(text or "").strip()
    shape = analyze_prompt_shape(objective)
    obligations = tuple(str(item).strip() for item in shape.question_segments if str(item).strip())
    obligation_count = max(
        1,
        int(shape.question_parts or 1),
        int(shape.numbered_parts or 0),
        int(shape.imperative_parts or 0),
        len(obligations),
    )
    inline_reply = bool(looks_like_inline_answer_request(objective))
    delivery_mode = INLINE_REPLY if inline_reply else NON_INLINE
    token_floor = int(answer_surface_token_floor(objective))
    planning_tokens = int(answer_surface_planning_tokens(objective))
    requires_complete_reply = bool(
        inline_reply and shape.requires_single_reply_coverage
    )

    requires_deliberation = bool(
        inline_reply
        and (
            (requires_complete_reply and obligation_count >= 3)
            or (requires_complete_reply and token_floor >= 768)
            or planning_tokens >= 1024
        )
    )

    basis: list[str] = [delivery_mode]
    if obligation_count >= 2:
        basis.append("multipart")
    if requires_complete_reply:
        basis.append("single_reply_coverage")
    if token_floor >= 768:
        basis.append("answer_capacity")
    if requires_deliberation:
        basis.append("deliberate_work")

    return SemanticWorkContract(
        delivery_mode=delivery_mode,
        obligations=obligations,
        obligation_count=obligation_count,
        requires_complete_reply=requires_complete_reply,
        requires_deliberation=requires_deliberation,
        # This is eligibility for internal, side-effect-free support.  It is
        # intentionally not a capability dispatch and grants no authority.
        architecture_assistance_eligible=requires_deliberation,
        answer_token_floor=token_floor,
        planning_token_estimate=planning_tokens,
        decision_basis=tuple(basis),
    )
