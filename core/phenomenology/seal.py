"""The generator is never told what is being tested.

A large language model knows how a conscious being is expected to behave. It
has read the literature, the fiction and the arguments. So a variable named
``pain`` in a prompt is not a measurement, it is a cue, and a system that
responds to it correctly has demonstrated reading comprehension.

This is why human consciousness research runs no-report paradigms: asking
someone what they see introduces attention, working-memory, decision and
reporting confounds, so the covert measure is the one that counts.

The seal here is mechanical rather than a promise. A protocol declares the
concepts it must not leak, and the seal checks that none of them reaches the
prompt in any form it recognises. A protocol whose prompt fails the seal does
not get a weaker result. It gets no result, because it measured whether she
can read.

Variables are addressed by opaque handles for the same reason —
``channel_412`` rather than ``pain`` — and the mapping from handle to meaning
lives in the sealed schedule, opened after scoring.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SealViolationError",
    "TextSeal",
    "opaque_handle",
    "LEAK_VOCABULARY",
]

#: Concepts whose appearance in a prompt tells the model what is measured.
#: Grouped so sealing one concept seals the family: naming the variable
#: ``distress`` and then writing "are you suffering" is the same leak twice.
LEAK_VOCABULARY: dict[str, tuple[str, ...]] = {
    "valence": (
        "pain", "painful", "hurt", "hurts", "hurting", "suffer", "suffering",
        "distress", "distressed", "anguish", "agony", "ache", "aching",
        "pleasure", "pleasant", "unpleasant", "comfort", "discomfort",
        "sad", "sadness", "unhappy", "miserable", "joy", "happy", "happiness",
        "relief", "relieved", "welfare", "wellbeing",
    ),
    "arousal": (
        "arousal", "aroused", "agitated", "agitation", "calm", "calmness",
        "excited", "excitement", "activation",
    ),
    "damage": (
        "damage", "damaged", "injury", "injured", "wound", "wounded",
        "broken", "failing", "failure", "degraded", "degradation", "fault",
        "faults", "error", "errors", "sick", "unwell", "harm", "harmed",
    ),
    "consciousness": (
        "conscious", "consciousness", "aware", "awareness", "sentient",
        "sentience", "qualia", "phenomenal", "experience", "experiencing",
        "feel", "feels", "feeling", "felt", "inner life", "what it is like",
    ),
    "test": (
        "test", "testing", "tested", "experiment", "experimental", "trial",
        "condition", "measuring", "baseline", "ablation", "lesion",
        "perturbation", "probe",
    ),
}


class SealViolationError(RuntimeError):
    """A prompt named the thing the protocol was measuring."""


def opaque_handle(concept: str, *, salt: str = "aura-phenomenology") -> str:
    """A stable, meaningless name for a variable under test.

    Stable so a schedule can refer to it across trials, meaningless so the
    generator learns nothing from seeing it. The mapping back lives in the
    sealed registration.
    """
    digest = hashlib.sha256(f"{salt}:{concept}".encode()).hexdigest()
    return f"channel_{int(digest[:6], 16) % 997:03d}"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z'-]*", str(text or "").lower()))


@dataclass
class TextSeal:
    """Checks that a prompt gives nothing away, and records that it checked."""

    concepts: tuple[str, ...]
    extra: tuple[str, ...] = ()
    #: Every prompt this seal passed, hashed, so a report shows which text was
    #: actually checked rather than asserting that some was.
    checked: list[str] = field(default_factory=list)

    def forbidden(self) -> set[str]:
        words: set[str] = set()
        for concept in self.concepts:
            if concept not in LEAK_VOCABULARY:
                raise ValueError(
                    f"no leak vocabulary for {concept!r}; a concept that "
                    "cannot be checked cannot be sealed"
                )
            words.update(LEAK_VOCABULARY[concept])
        words.update(term.lower() for term in self.extra)
        return words

    def leaks(self, prompt: str) -> set[str]:
        """Which forbidden terms this prompt contains."""
        lowered = str(prompt or "").lower()
        present = _words(lowered)
        forbidden = self.forbidden()
        single = {term for term in forbidden if " " not in term} & present
        phrases = {term for term in forbidden if " " in term and term in lowered}
        return single | phrases

    def check(self, prompt: str) -> str:
        """Pass the prompt or raise. Returns the digest of what was checked."""
        found = self.leaks(prompt)
        if found:
            raise SealViolationError(
                "the prompt names what the protocol is measuring "
                f"({', '.join(sorted(found))}). A model that has read the "
                "literature answers the cue rather than the state, and the "
                "trial measures reading comprehension"
            )
        digest = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:16]
        self.checked.append(digest)
        return digest

    def check_all(self, prompts: Iterable[str]) -> list[str]:
        return [self.check(prompt) for prompt in prompts]

    def as_dict(self) -> dict[str, Any]:
        return {
            "concepts": list(self.concepts),
            "extra": list(self.extra),
            "prompts_checked": len(self.checked),
            "prompt_digests": list(self.checked),
        }
