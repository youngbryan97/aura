"""Effect claims that cannot be rendered without the evidence for them.

The honesty checks in claimed_effect.py read finished prose and look for
sentences that assert a completed effect. That is detection, and detection over
natural language cannot be closed: Aura can form indefinitely many sentences
describing the world, and any finite set of patterns recognises a subset of
them. The inequality is permanent —

    {claims she can generate} ⊃ {claims the auditor recognises}

— so every phrasing added to the auditor narrows the gap and none of them shuts
it. Raised against this codebase on 2026-08-10, and correct.

This module is the other half. Rather than inspecting a sentence after it
exists, an EffectClaim is a typed proposition that carries its own evidence,
and the renderer refuses to produce completed-effect text for a claim with no
receipt behind it. For the path that renders effects, an unevidenced claim is
not caught — it is unrepresentable.

    render(claim) is defined only when claim.evidence_ids is non-empty
    and claim.tense is COMPLETED

That is an architectural invariant rather than a growing list of detectors, and
it is enforced by construction: EffectClaim.completed() raises without
evidence, so no caller can build one.

Scope, stated plainly because the distinction is the whole point: this closes
the loop where Aura RENDERS an effect from a receipt. Free-form text generated
by the model remains audited by claimed_effect.py rather than constrained by
this, and that surface is still a detector. What this removes is the class of
false claim that the runtime itself was composing — the "Done — wrote X"
sentences assembled from a result payload, which is exactly where a receipt was
already in hand and was not being checked against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.epistemics.effect_registry import EFFECT_REGISTRY

__all__ = [
    "EffectClaim",
    "assertions_from_receipts",
    "EffectTense",
    "UnevidencedClaimError",
    "claims_from_receipts",
    "render_effect_claims",
]


class UnevidencedClaimError(ValueError):
    """A completed effect was asserted with no receipt behind it."""


class EffectTense(StrEnum):
    COMPLETED = "completed"
    ATTEMPTED = "attempted"
    REFUSED = "refused"


#: How each action reads in a sentence, and which receipt field names its
#: object. Keyed by the action vocabulary, which is finite and declared — that
#: is what makes this closable where language is not.
#:
#: Derived rather than written out, because the same 15 actions also need a
#: RECOGNISER for the auditing side, and two hand-maintained lists of the same
#: capability set is precisely the fragmentation this work was criticised for:
#: the day one list gains an action and the other does not, Aura can narrate an
#: effect nothing is able to audit. core/epistemics/effect_registry.py holds
#: both halves per action and fails a gate when either is missing.
_EFFECT_VOCABULARY: dict[str, tuple[str, tuple[str, ...]]] = {
    spec.action: (spec.render_phrase, spec.evidence_fields)
    for spec in EFFECT_REGISTRY.values()
    if spec.narrates_in_reply
}


@dataclass(frozen=True, slots=True)
class EffectClaim:
    """A statement about an effect, bound to the receipts that support it."""

    action: str
    tense: EffectTense
    obj: str = ""
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.tense is EffectTense.COMPLETED and not self.evidence_ids:
            raise UnevidencedClaimError(
                f"completed effect {self.action!r} has no evidence; a completed "
                "claim without a receipt cannot be constructed"
            )

    @classmethod
    def completed(
        cls, action: str, *, obj: str = "", evidence_ids: tuple[str, ...]
    ) -> EffectClaim:
        """The only way to assert a finished effect. Raises without evidence."""
        return cls(
            action=action,
            tense=EffectTense.COMPLETED,
            obj=obj,
            evidence_ids=tuple(evidence_ids),
        )

    @classmethod
    def attempted(cls, action: str, *, obj: str = "") -> EffectClaim:
        """A step that ran without verifying. Never renders as done."""
        return cls(action=action, tense=EffectTense.ATTEMPTED, obj=obj)

    def render(self) -> str:
        """Words for this claim, or "" when it has nothing it may assert."""
        if self.tense is not EffectTense.COMPLETED:
            return ""
        if self.action == "os_automation":
            # The object is already a full phrase built from the proven
            # criterion, so it is rendered as-is.
            return self.obj.strip()
        phrase, _fields = _EFFECT_VOCABULARY.get(self.action, ("", ()))
        if not phrase:
            return ""
        return f"{phrase} {self.obj}".strip()


#: "listed=/path;count=9" and "clipboard_contains=ORION-7" — the verifier
#: writes its proof as key=value pairs, so the effect it proved can be read
#: back out of the evidence rather than re-derived.
_EVIDENCE_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>[^;]+)")

#: Which proven criterion corresponds to which effect sentence.
_PROVEN_CRITERION_VOCABULARY: dict[str, str] = {
    "clipboard_contains": "put {value} on the clipboard",
    "file_exists": "wrote {value}",
    "file_contains": "wrote {value}",
    "browser_url_contains": "opened {value}",
    "app_frontmost": "brought {value} to the front",
    "app_not_running": "quit {value}",
}


def _claims_from_os_automation(receipt: dict[str, Any]) -> list[EffectClaim]:
    """Effect claims for a script lane, read from the criteria it proved."""

    if not receipt.get("ok"):
        return []
    evidence = str(receipt.get("effect_evidence") or "")
    identifier = str(receipt.get("durable_receipt_id") or f"step:{receipt.get('index', 0)}")
    claims: list[EffectClaim] = []
    for match in _EVIDENCE_FIELD_RE.finditer(evidence):
        template = _PROVEN_CRITERION_VOCABULARY.get(match.group("key"))
        if not template:
            continue
        value = match.group("value").strip()
        if not value:
            continue
        claims.append(
            EffectClaim.completed(
                "os_automation",
                obj=template.format(value=value),
                evidence_ids=(identifier,),
            )
        )
    return claims


def claims_from_receipts(receipts: Any) -> list[EffectClaim]:
    """Turn step receipts into claims, completed only where they verified.

    receipt["ok"] is the strong signal: the tool returned success AND
    _verify_step_effect confirmed the effect. Anything short of that becomes an
    ATTEMPTED claim, which renders as nothing.
    """

    if not isinstance(receipts, list):
        return []
    claims: list[EffectClaim] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            continue
        action = str(receipt.get("action") or "").strip()
        if action == "os_automation":
            # This lane executes an AppleScript and names its effect in the
            # contract it proved, not in the step's action. Without this the
            # reply fell back to "the desktop steps completed and their effects
            # verified" for a turn that had put an exact string on the
            # clipboard and confirmed it — a receipt strong enough to name what
            # it did, reported as bookkeeping.
            claims.extend(_claims_from_os_automation(receipt))
            continue
        if action not in _EFFECT_VOCABULARY:
            continue
        inner = receipt.get("result")
        payload = inner if isinstance(inner, dict) else receipt
        _phrase, fields = _EFFECT_VOCABULARY[action]
        obj = ""
        if action == "system_control":
            # A system-control receipt already carries the generic affordance
            # domain and the value confirmed by readback. Preserve both. The
            # former renderer selected only ``domain`` and produced "changed a
            # system setting wallpaper", while selecting only ``applied``
            # would hide which setting changed. This composition remains
            # domain-agnostic: a newly registered setting becomes narratable
            # without adding another branch here.
            domain = str(payload.get("domain") or "").strip().replace("_", " ")
            applied = str(payload.get("applied") or payload.get("value") or "").strip()
            if domain and applied:
                obj = f"{domain} to {applied}"
            else:
                obj = domain or applied
        else:
            for name in fields:
                value = str(payload.get(name) or "").strip()
                if value:
                    obj = value
                    break
        if receipt.get("ok"):
            identifier = str(
                receipt.get("durable_receipt_id") or f"step:{receipt.get('index', index)}"
            )
            claims.append(
                EffectClaim.completed(action, obj=obj, evidence_ids=(identifier,))
            )
        else:
            claims.append(EffectClaim.attempted(action, obj=obj))
    return claims


def assertions_from_receipts(receipts: Any) -> list[Any]:
    """The same effects as Assertions on the shared epistemic substrate.

    EffectClaim came first and covers desktop effects only. Assertion is the
    general form — subject, provenance, evidence, confidence, time, source and
    verification for anything checkable — and having two types for one idea was
    the fragmentation this work was criticised for. Effects are lowered onto
    the substrate here rather than kept beside it.
    """

    from core.epistemics.assertion import Assertion, SourceKind, Verification

    lowered: list[Any] = []
    for claim in claims_from_receipts(receipts):
        rendered = claim.render()
        completed = claim.tense is EffectTense.COMPLETED
        lowered.append(
            Assertion(
                subject=claim.obj or claim.action,
                claim=f"I {rendered}" if rendered else f"I ran {claim.action}",
                source=SourceKind.RECEIPTED if completed else SourceKind.GENERATED,
                provenance=f"desktop_task step receipt ({claim.action})",
                evidence=claim.evidence_ids,
                verification=(
                    Verification.VERIFIED if completed else Verification.UNVERIFIED
                ),
            )
        )
    return lowered


def render_effect_claims(receipts: Any) -> str:
    """One sentence naming every effect that actually verified, or "".

    This is what the desktop reply says it did. It cannot overstate the work,
    because a claim it would have to overstate cannot be built.
    """

    rendered = [claim.render() for claim in claims_from_receipts(receipts)]
    rendered = [text for text in rendered if text]
    if not rendered:
        return ""
    if len(rendered) == 1:
        return f"{rendered[0]}."
    return ", ".join(rendered[:-1]) + f", and {rendered[-1]}."
