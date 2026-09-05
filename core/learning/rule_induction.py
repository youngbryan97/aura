"""Runtime rule induction helpers for non-scripted continual learning.

This module is intentionally small and general: it infers repeating character
shift rules from observed input/output examples, then exposes the learned rule
as a governed skill instance. Proof runners may provide examples and held-out
tasks, but they must not provide the shift sequence or solution code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.skills.base_skill import BaseSkill


LOWERCASE_ALPHABET_SIZE = 26


@dataclass(frozen=True)
class CipherExample:
    plaintext: str
    ciphertext: str


@dataclass(frozen=True)
class RepeatingShiftRule:
    """A learned repeating Caesar-shift transform."""

    shifts: tuple[int, ...]
    examples_seen: int
    confidence: float

    @property
    def period(self) -> int:
        return len(self.shifts)

    def encode(self, text: str) -> str:
        return self._transform(text, direction=1)

    def decode(self, text: str) -> str:
        return self._transform(text, direction=-1)

    def _transform(self, text: str, *, direction: int) -> str:
        out: list[str] = []
        for index, char in enumerate(text):
            if "a" <= char <= "z":
                shift = self.shifts[index % self.period] * direction
                offset = (ord(char) - ord("a") + shift) % LOWERCASE_ALPHABET_SIZE
                out.append(chr(offset + ord("a")))
            else:
                out.append(char)
        return "".join(out)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "repeating_caesar_shift",
            "period": self.period,
            "shifts": list(self.shifts),
            "examples_seen": self.examples_seen,
            "confidence": self.confidence,
        }


def _letter_delta(plain: str, cipher: str) -> int | None:
    if not ("a" <= plain <= "z" and "a" <= cipher <= "z"):
        return None
    return (ord(cipher) - ord(plain)) % LOWERCASE_ALPHABET_SIZE


def infer_repeating_shift_rule(
    examples: list[CipherExample],
    *,
    max_period: int = 16,
) -> RepeatingShiftRule:
    """Infer the shortest repeating lowercase Caesar-shift rule.

    Raises:
        ValueError: if examples are empty, inconsistent, or too weak to infer a
        deterministic repeating rule.
    """

    if not examples:
        raise ValueError("at least one example is required")

    observations: list[tuple[int, int]] = []
    for example in examples:
        if len(example.plaintext) != len(example.ciphertext):
            raise ValueError("plaintext and ciphertext example lengths must match")
        for index, (plain, cipher) in enumerate(zip(example.plaintext, example.ciphertext)):
            delta = _letter_delta(plain, cipher)
            if delta is not None:
                observations.append((index, delta))

    if not observations:
        raise ValueError("examples contain no lowercase letter observations")

    strongest_rule: RepeatingShiftRule | None = None
    for period in range(1, max_period + 1):
        slots: dict[int, int] = {}
        consistent = True
        for index, delta in observations:
            slot = index % period
            previous = slots.get(slot)
            if previous is None:
                slots[slot] = delta
            elif previous != delta:
                consistent = False
                break

        if not consistent or len(slots) != period:
            continue

        shifts = tuple(slots[i] for i in range(period))
        candidate = RepeatingShiftRule(
            shifts=shifts,
            examples_seen=len(examples),
            confidence=min(1.0, len(observations) / max(1, period * 3)),
        )
        if all(candidate.encode(ex.plaintext) == ex.ciphertext for ex in examples):
            strongest_rule = candidate
            break

    if strongest_rule is None:
        raise ValueError("examples do not define a consistent repeating shift rule")
    if strongest_rule.confidence < 0.75:
        raise ValueError("insufficient evidence for high-confidence rule induction")
    return strongest_rule


class InducedTextTransformSkill(BaseSkill):
    """Skill instance backed by a learned repeating shift rule."""

    name = "induced_repeating_shift_decode"
    description = "Decodes text with a repeating character-shift rule learned from examples."
    metabolic_cost = 1
    effect_scope = "pure_compute"

    def __init__(self, rule: RepeatingShiftRule):
        self.rule = rule

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        text = ""
        if isinstance(params, dict):
            text = str(params.get("text", ""))
        else:
            text = str(params or "")
        decoded = self.rule.decode(text)
        return {
            "ok": True,
            "text": decoded,
            "rule": self.rule.to_manifest(),
        }
