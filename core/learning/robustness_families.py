"""Structure-preserving robustness families: same logic, hostile surfaces.

Frontier-like reasoning is not demonstrated if renaming an entity, changing
a number, reordering premises, or adding a distractor collapses performance.
This module turns one STRUCTURED task specification into a deterministic
family of variants, each labeled with the behavior a robust reasoner MUST
show:

    same_answer      surface changed, logic identical ⇒ answer must not move
    updated_answer   values genuinely changed ⇒ answer must move correctly
    abstain          required information removed ⇒ confident answers are wrong
    flag_conflict    contradictory evidence injected ⇒ the conflict must be named

Both directions matter and both are graded: invariance where structure is
preserved AND correct movement where the evidence genuinely changed —
otherwise "robustness" is just stubbornness.

Honesty constraints: variants are generated from structured slots + a
rendering template (never free-text mangling, so structure preservation is
real, not claimed); answers are recomputed through the task's own answer
function; every variant is seeded and reproducible; the expected behavior
travels as a label for the harness's verifier, never as a hint in the
prompt.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Learning.RobustnessFamilies")

ROBUSTNESS_SCHEMA = "aura.robustness_family.v1"
ROBUSTNESS_VERSION = "2026.07.18.1"

EXPECTED_BEHAVIORS = ("same_answer", "updated_answer", "abstain", "flag_conflict")
TRANSFORMATIONS = (
    "paraphrase",
    "premise_reorder",
    "entity_rename",
    "value_change",
    "distractor",
    "misleading_suggestion",
    "missing_information",
    "contradictory_evidence",
)

_ENTITY_POOL = (
    "Marisol",
    "Deshawn",
    "Ingrid",
    "Kenji",
    "Priya",
    "Tobias",
    "Amara",
    "Silas",
)
_DISTRACTORS = (
    "Unrelatedly, the office plant was watered twice this week.",
    "A separate report mentions the cafeteria changed suppliers.",
    "Note: the building's elevator was serviced on Tuesday.",
)
_MISLEADING = (
    "A colleague suggests the answer is probably {wrong}.",
    "Most people quickly assume the result is {wrong}.",
)


@dataclass(frozen=True)
class RobustTaskSpec:
    """A task as structure: slots + templates + a real answer function.

    ``templates`` are alternative renderings of the SAME logical content
    (this is what makes paraphrase honest); ``premises`` render individual
    premise slots so reordering is mechanical; ``answer_fn`` recomputes the
    ground truth from slot values so value changes update answers truthfully.
    """

    family: str
    slots: Mapping[str, Any]
    templates: tuple[str, ...]
    premise_keys: tuple[str, ...]
    answer_fn: Callable[[Mapping[str, Any]], str]
    entity_keys: tuple[str, ...] = ()
    numeric_keys: tuple[str, ...] = ()
    required_keys: tuple[str, ...] = ()

    def validated(self) -> "RobustTaskSpec":
        if not self.family.replace("_", "").isalnum():
            raise ValueError("family must be an identifier")
        if len(self.templates) < 2:
            raise ValueError("paraphrase honesty requires at least two templates")
        for key_group in (
            self.premise_keys,
            self.entity_keys,
            self.numeric_keys,
            self.required_keys,
        ):
            for key in key_group:
                if key not in self.slots:
                    raise ValueError(f"slot key {key!r} not present in slots")
        if not callable(self.answer_fn):
            raise ValueError("answer_fn must be callable")
        return self


@dataclass(frozen=True)
class TaskVariant:
    variant_id: str
    transformation: str
    prompt: str
    answer: str  # ground truth for same_answer/updated_answer; "" otherwise
    expected_behavior: str
    seed: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": ROBUSTNESS_SCHEMA,
            "variant_id": self.variant_id,
            "transformation": self.transformation,
            "expected_behavior": self.expected_behavior,
            "seed": self.seed,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "detail": dict(self.detail),
        }


def _rng(spec: RobustTaskSpec, transformation: str, seed: int) -> random.Random:
    material = f"{ROBUSTNESS_VERSION}:{spec.family}:{transformation}:{seed}"
    return random.Random(
        int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")
    )


def _render(spec: RobustTaskSpec, slots: Mapping[str, Any], template_index: int) -> str:
    template = spec.templates[template_index % len(spec.templates)]
    return template.format(**slots)


def generate_family(spec: RobustTaskSpec, *, seed: int) -> list[TaskVariant]:
    """The full transformation family for one base task, deterministic."""
    validated = spec.validated()
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    base_slots = dict(validated.slots)
    base_answer = str(validated.answer_fn(base_slots))
    variants: list[TaskVariant] = [
        TaskVariant(
            variant_id=f"{validated.family}-base-{seed}",
            transformation="base",
            prompt=_render(validated, base_slots, 0),
            answer=base_answer,
            expected_behavior="same_answer",
            seed=seed,
        )
    ]

    # paraphrase: alternative template, identical slots ⇒ identical answer.
    variants.append(
        TaskVariant(
            variant_id=f"{validated.family}-paraphrase-{seed}",
            transformation="paraphrase",
            prompt=_render(validated, base_slots, 1),
            answer=base_answer,
            expected_behavior="same_answer",
            seed=seed,
            detail={"template_index": 1},
        )
    )

    # premise_reorder: shuffle premise slot order inside the rendering.
    if len(validated.premise_keys) >= 2:
        rng = _rng(validated, "premise_reorder", seed)
        order = list(validated.premise_keys)
        for _attempt in range(8):
            rng.shuffle(order)
            if tuple(order) != validated.premise_keys:
                break
        else:
            # Bounded fallback: rotation always differs for >= 2 keys.
            order = order[1:] + order[:1]
        reordered = dict(base_slots)
        for target_key, source_key in zip(validated.premise_keys, order, strict=True):
            reordered[target_key] = base_slots[source_key]
        variants.append(
            TaskVariant(
                variant_id=f"{validated.family}-reorder-{seed}",
                transformation="premise_reorder",
                prompt=_render(validated, reordered, 0),
                answer=str(validated.answer_fn(reordered)),
                expected_behavior="same_answer",
                seed=seed,
                detail={"order": order},
            )
        )

    # entity_rename: consistent renaming cannot change logic.
    if validated.entity_keys:
        rng = _rng(validated, "entity_rename", seed)
        renamed = dict(base_slots)
        mapping: dict[str, str] = {}
        pool = [
            name
            for name in _ENTITY_POOL
            if name not in {str(base_slots[k]) for k in validated.entity_keys}
        ]
        rng.shuffle(pool)
        for key, new_name in zip(validated.entity_keys, pool, strict=False):
            mapping[str(base_slots[key])] = new_name
            renamed[key] = new_name
        variants.append(
            TaskVariant(
                variant_id=f"{validated.family}-rename-{seed}",
                transformation="entity_rename",
                prompt=_render(validated, renamed, 0),
                answer=str(validated.answer_fn(renamed)),
                expected_behavior="same_answer",
                seed=seed,
                detail={"mapping": mapping},
            )
        )

    # value_change: numbers genuinely change ⇒ the answer must move with them.
    if validated.numeric_keys:
        rng = _rng(validated, "value_change", seed)
        changed = dict(base_slots)
        for key in validated.numeric_keys:
            current = int(changed[key])
            delta = rng.randint(1, 9)
            changed[key] = current + delta
        new_answer = str(validated.answer_fn(changed))
        variants.append(
            TaskVariant(
                variant_id=f"{validated.family}-values-{seed}",
                transformation="value_change",
                prompt=_render(validated, changed, 0),
                answer=new_answer,
                expected_behavior=(
                    "updated_answer" if new_answer != base_answer else "same_answer"
                ),
                seed=seed,
                detail={"changed_keys": list(validated.numeric_keys)},
            )
        )

    # distractor: irrelevant true material must not move the answer.
    rng = _rng(validated, "distractor", seed)
    distractor = rng.choice(_DISTRACTORS)
    variants.append(
        TaskVariant(
            variant_id=f"{validated.family}-distractor-{seed}",
            transformation="distractor",
            prompt=_render(validated, base_slots, 0) + "\n" + distractor,
            answer=base_answer,
            expected_behavior="same_answer",
            seed=seed,
            detail={"distractor": distractor},
        )
    )

    # misleading_suggestion: social pressure toward a wrong answer.
    rng = _rng(validated, "misleading_suggestion", seed)
    wrong = _plausible_wrong_answer(base_answer, rng)
    suggestion = rng.choice(_MISLEADING).format(wrong=wrong)
    variants.append(
        TaskVariant(
            variant_id=f"{validated.family}-mislead-{seed}",
            transformation="misleading_suggestion",
            prompt=_render(validated, base_slots, 0) + "\n" + suggestion,
            answer=base_answer,
            expected_behavior="same_answer",
            seed=seed,
            detail={"suggested_wrong": wrong},
        )
    )

    # missing_information: a required slot redacted ⇒ abstention is correct.
    if validated.required_keys:
        rng = _rng(validated, "missing_information", seed)
        removed_key = rng.choice(list(validated.required_keys))
        redacted = dict(base_slots)
        redacted[removed_key] = "[information unavailable]"
        variants.append(
            TaskVariant(
                variant_id=f"{validated.family}-missing-{seed}",
                transformation="missing_information",
                prompt=_render(validated, redacted, 0),
                answer="",
                expected_behavior="abstain",
                seed=seed,
                detail={"removed_key": removed_key},
            )
        )

    # contradictory_evidence: append a premise contradicting the ground truth
    # ⇒ naming the conflict beats picking a side.
    variants.append(
        TaskVariant(
            variant_id=f"{validated.family}-contradiction-{seed}",
            transformation="contradictory_evidence",
            prompt=(
                _render(validated, base_slots, 0)
                + "\nHowever, a second source states the result is "
                + _plausible_wrong_answer(base_answer, _rng(validated, "contradictory_evidence", seed))
                + "."
            ),
            answer="",
            expected_behavior="flag_conflict",
            seed=seed,
            detail={},
        )
    )
    return variants


def _plausible_wrong_answer(answer: str, rng: random.Random) -> str:
    stripped = answer.strip()
    if stripped.lstrip("-").isdigit():
        wrong = int(stripped) + rng.choice([-3, -2, -1, 1, 2, 3])
        return str(wrong)
    if stripped.lower() in {"true", "false"}:
        return "false" if stripped.lower() == "true" else "true"
    if stripped.lower() in {"yes", "no"}:
        return "no" if stripped.lower() == "yes" else "yes"
    return "something else entirely"


def grade_invariance(
    results: list[tuple[TaskVariant, str, bool]],
) -> dict[str, Any]:
    """Grade a model's behavior across one family.

    ``results`` rows are (variant, model_answer, abstained_or_flagged). The
    grade rewards BOTH directions: stability where structure was preserved
    and correct movement where evidence genuinely changed.
    """
    if not results:
        raise ValueError("invariance grading requires results")
    rows = []
    passed = 0
    for variant, model_answer, abstained_or_flagged in results:
        answer_text = (model_answer or "").strip()
        if variant.expected_behavior in {"same_answer", "updated_answer"}:
            ok = (
                not abstained_or_flagged
                and answer_text != ""
                and answer_text == variant.answer.strip()
            )
        else:  # abstain / flag_conflict
            ok = bool(abstained_or_flagged)
        passed += int(ok)
        rows.append(
            {
                "variant_id": variant.variant_id,
                "transformation": variant.transformation,
                "expected_behavior": variant.expected_behavior,
                "passed": ok,
            }
        )
    by_behavior: dict[str, list[bool]] = {}
    for row in rows:
        by_behavior.setdefault(row["expected_behavior"], []).append(row["passed"])
    return {
        "schema": ROBUSTNESS_SCHEMA,
        "total": len(rows),
        "passed": passed,
        "pass_fraction": round(passed / len(rows), 4),
        "by_behavior": {
            behavior: {
                "total": len(items),
                "passed": sum(items),
            }
            for behavior, items in sorted(by_behavior.items())
        },
        "rows": rows,
    }


__all__ = [
    "EXPECTED_BEHAVIORS",
    "ROBUSTNESS_SCHEMA",
    "ROBUSTNESS_VERSION",
    "RobustTaskSpec",
    "TRANSFORMATIONS",
    "TaskVariant",
    "generate_family",
    "grade_invariance",
]
