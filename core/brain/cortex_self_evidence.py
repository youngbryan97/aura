"""Typed, evidence-bound facts about Aura's currently resident cortex.

This resolves the one validated active-cortex authority and the independently
verified qualified semantic-tissue authority, then emits compact assertions
for the existing language substrate. A narrow renderer serves closed factual
self-questions from those assertions; open reflection remains model-authored.
Missing evidence remains explicitly unmeasured.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.epistemics.assertion import (
    Assertion,
    AssertionResponse,
    SourceKind,
    Verification,
)


class CortexEvidenceRequest(StrEnum):
    """Closed self-evidence questions the runtime can answer without guessing."""

    IDENTITY = "identity"
    MEASURED_COMPARISON = "measured_comparison"
    BOUNDED_MECHANISM = "bounded_mechanism"


@dataclass(frozen=True, slots=True)
class CortexCampaignEvidence:
    """One independently verified, model-bound recurrent campaign."""

    cortex_label: str
    model_path: str
    task_count: int
    exact_by_arm: tuple[tuple[str, int], ...]
    gain_count: int
    regression_count: int
    paired_p_value: float
    elapsed_seconds: float
    artifact_receipt_sha256: str
    verification_receipt_sha256: str

    @property
    def decode_count(self) -> int:
        return self.task_count * len(self.exact_by_arm)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (
                self.artifact_receipt_sha256,
                self.verification_receipt_sha256,
            )
            if value
        )


@dataclass(frozen=True, slots=True)
class CortexSelfEvidence:
    resident_label: str
    model_type: str
    total_parameters: int
    native_context_tokens: int
    served_context_tokens: int
    promotion_verdict: str
    identity_behavior_changed: bool | None
    component_states: tuple[tuple[str, str], ...]
    semantic_active: bool
    semantic_verdict: str
    semantic_task_count: int
    semantic_exact_by_arm: tuple[tuple[str, int], ...]
    semantic_gain_count: int
    semantic_regression_count: int
    semantic_p_value: float | None
    semantic_activation_sha256: str = ""
    resident_descriptor_sha256: str = ""
    resident_model_path: str = ""
    campaigns: tuple[CortexCampaignEvidence, ...] = ()

    def resident_campaign(self) -> CortexCampaignEvidence | None:
        """The campaign bound to this exact resident checkpoint, if verified."""

        return next(
            (
                campaign
                for campaign in self.campaigns
                if campaign.model_path and campaign.model_path == self.resident_model_path
            ),
            None,
        )

    def assertions(self) -> tuple[str, ...]:
        """Render privacy-safe facts for the operational language substrate."""

        lines = [
            "Resident cortex: "
            f"{self.resident_label}, {self.model_type}, "
            f"{self.total_parameters:,} parameters; native context "
            f"{self.native_context_tokens:,} tokens and currently qualified serving "
            f"context {self.served_context_tokens:,} tokens.",
        ]
        if self.promotion_verdict:
            behavior = (
                " The migration authority records that model-generation identity behavior changed."
                if self.identity_behavior_changed is True
                else ""
            )
            lines.append(f"Cortex promotion evaluation: {self.promotion_verdict}.{behavior}")
        if self.component_states:
            dispositions = ", ".join(f"{name}={state}" for name, state in self.component_states)
            lines.append(f"Cortex migration components: {dispositions}.")
        if self.semantic_active and self.semantic_task_count > 0:
            arms = dict(self.semantic_exact_by_arm)
            lines.append(
                "Measured bounded recurrent semantic tissue: "
                f"{self.semantic_verdict}; treatment "
                f"{arms.get('treatment', 0)}/{self.semantic_task_count}, ordinary "
                f"decode {arms.get('ordinary_base', 0)}/{self.semantic_task_count}, "
                f"gain {self.semantic_gain_count}, regressions "
                f"{self.semantic_regression_count}"
                + (
                    f", paired exact p={self.semantic_p_value:.3g}."
                    if self.semantic_p_value is not None
                    else "."
                )
            )
        comparison = self.measured_campaign_comparison()
        if comparison:
            lines.append(comparison)
        lines.append(
            "Cortex comparison boundary: no paired evidence currently attributes "
            "differences in conversational style, association speed, broad reasoning, "
            "knowledge, or subjective experience to the model swap; those differences "
            "are unmeasured, not observations."
        )
        return tuple(lines)

    def measured_campaign_comparison(self) -> str:
        """Render the cross-generation measurement only when both receipts agree."""

        current = self.resident_campaign()
        previous = next(
            (campaign for campaign in self.campaigns if campaign is not current),
            None,
        )
        if current is None or previous is None:
            return ""
        if (
            current.task_count != previous.task_count
            or current.decode_count != previous.decode_count
            or tuple(name for name, _score in current.exact_by_arm)
            != tuple(name for name, _score in previous.exact_by_arm)
            or previous.elapsed_seconds <= 0.0
        ):
            return ""
        reduction = (
            100.0 * (previous.elapsed_seconds - current.elapsed_seconds) / previous.elapsed_seconds
        )
        return (
            "Recorded matched-shape recurrent campaigns: "
            f"{previous.cortex_label} {previous.elapsed_seconds:,.3f}s and "
            f"{current.cortex_label} {current.elapsed_seconds:,.3f}s for "
            f"{current.decode_count} decodes over {current.task_count} tasks; "
            f"the latter run was {reduction:.1f}% faster. This is bounded campaign "
            "throughput evidence, not an isolated broad-quality comparison."
        )


def _component_states(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        return ()
    states: list[tuple[str, str]] = []
    for name, component in sorted(value.items()):
        if not isinstance(component, dict):
            continue
        status = str(component.get("status") or "").strip()
        if status:
            states.append((str(name), status))
    return tuple(states)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN_PATHS: tuple[tuple[str, Path], ...] = (
    (
        "32B",
        _PROJECT_ROOT / "artifacts/closeout/latent_cortex/"
        "cp566_resident_mixed_multidomain_replication",
    ),
    (
        "27B",
        _PROJECT_ROOT / "artifacts/migration/27b/recovery/cp1003-semantic-canary",
    ),
)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verified_campaign(
    label: str,
    root: Path,
) -> CortexCampaignEvidence | None:
    """Load a campaign only when producer, verifier, and adjudicator agree."""

    result = _json_object(root / "result.json")
    verification = _json_object(root / "verification.json")
    adjudication = _json_object(root / "adjudication.json")
    if result is None or verification is None or adjudication is None:
        return None
    if verification.get("verified") is not True:
        return None
    if (
        adjudication.get("passed") is not True
        or str(adjudication.get("verdict") or "") != "BOUNDED_WOW_SIGNAL"
    ):
        return None

    result_receipt = str(result.get("receipt_sha256") or "")
    artifact_receipt = str(verification.get("artifact_receipt_sha256") or "")
    verification_receipt = str(verification.get("verification_receipt_sha256") or "")
    if not result_receipt or result_receipt != artifact_receipt or not verification_receipt:
        return None

    result_identity = result.get("model_identity")
    verification_identity = verification.get("model_identity")
    if not isinstance(result_identity, dict) or result_identity != verification_identity:
        return None
    model_path = str(result_identity.get("path") or "")
    if not model_path:
        return None

    result_arms = result.get("arms")
    verified_arms = verification.get("independent_exact_by_arm")
    adjudicated_arms = adjudication.get("independent_exact_by_arm")
    if not isinstance(result_arms, dict) or not isinstance(verified_arms, dict):
        return None
    if verified_arms != adjudicated_arms:
        return None
    result_exact = {
        str(name): int(values.get("exact") or 0)
        for name, values in result_arms.items()
        if isinstance(values, dict)
    }
    verified_exact = {str(name): int(value) for name, value in verified_arms.items()}
    if result_exact != verified_exact:
        return None

    task_count = int(verification.get("task_count") or 0)
    gain_count = int(verification.get("gain_count") or 0)
    regression_count = int(verification.get("regression_count") or 0)
    if (
        task_count <= 0
        or int(result.get("task_count") or 0) != task_count
        or int(adjudication.get("task_count") or 0) != task_count
        or int(result.get("gain_count") or 0) != gain_count
        or int(adjudication.get("gain_count") or 0) != gain_count
        or int(result.get("regression_count") or 0) != regression_count
        or int(adjudication.get("regression_count") or 0) != regression_count
    ):
        return None
    try:
        p_value = float(verification.get("paired_one_sided_exact_p"))
        adjudicated_p_value = float(adjudication.get("paired_one_sided_exact_p"))
        elapsed_seconds = float(result.get("elapsed_seconds"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        return None
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        return None
    if not math.isclose(p_value, adjudicated_p_value, rel_tol=0.0, abs_tol=0.0):
        return None

    return CortexCampaignEvidence(
        cortex_label=str(label),
        model_path=model_path,
        task_count=task_count,
        exact_by_arm=tuple(sorted(verified_exact.items())),
        gain_count=gain_count,
        regression_count=regression_count,
        paired_p_value=p_value,
        elapsed_seconds=elapsed_seconds,
        artifact_receipt_sha256=artifact_receipt,
        verification_receipt_sha256=verification_receipt,
    )


def _campaign_signature() -> tuple[tuple[str, int, int, int], ...]:
    """The file identity that makes a cached campaign observation current."""

    signature: list[tuple[str, int, int, int]] = []
    for _label, root in _CAMPAIGN_PATHS:
        for name in ("result.json", "verification.json", "adjudication.json"):
            path = root / name
            try:
                metadata = path.stat()
            except OSError:
                signature.append((str(path), -1, -1, -1))
            else:
                signature.append(
                    (
                        str(path),
                        int(metadata.st_size),
                        int(metadata.st_mtime_ns),
                        int(metadata.st_ctime_ns),
                    )
                )
    return tuple(signature)


@lru_cache(maxsize=4)
def _verified_cortex_campaigns_cached(
    _signature: tuple[tuple[str, int, int, int], ...],
) -> tuple[CortexCampaignEvidence, ...]:
    return tuple(
        campaign
        for label, root in _CAMPAIGN_PATHS
        if (campaign := _verified_campaign(label, root)) is not None
    )


def verified_cortex_campaigns() -> tuple[CortexCampaignEvidence, ...]:
    """Return source-bound campaigns; any evidence drift invalidates the cache."""

    return _verified_cortex_campaigns_cached(_campaign_signature())


_WORD_RE = re.compile(r"[a-z0-9]+")
_SELF_TERMS = frozenset({"aura", "you", "your", "yours"})
_CORTEX_TERMS = frozenset({"cortex", "model", "models", "checkpoint", "27b", "32b", "qwen"})
_MEASUREMENT_TERMS = frozenset(
    {
        "actual",
        "actually",
        "data",
        "distinguish",
        "evidence",
        "measurable",
        "measure",
        "measured",
        "observed",
        "prove",
        "proven",
        "result",
        "results",
        "test",
        "tested",
    }
)
_COMPARISON_TERMS = frozenset(
    {
        "after",
        "before",
        "changed",
        "compare",
        "compared",
        "comparison",
        "consequence",
        "current",
        "difference",
        "different",
        "former",
        "impact",
        "migration",
        "previous",
        "replaced",
        "replacing",
        "swap",
        "swapped",
        "upgrade",
        "upgraded",
    }
)
_IDENTITY_TERMS = frozenset({"active", "current", "now", "resident", "running", "using", "which"})
_MECHANISM_TERMS = frozenset({"rlc", "recurrent", "recurrence", "tissue", "wow"})


def _build_cortex_evidence_surface() -> object | None:
    """Generalize closed cortex-evidence requests beyond the lexical floor."""

    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        return LearnedMatcher(
            name="cortex_self_evidence_request",
            positives=(
                "What changed after replacing your former model that you can actually measure?",
                "Which cortex are you running now?",
                "What evidence proves that your recurrent tissue works?",
                "How did the resident-model migration affect measured performance?",
                "What do the receipts show about your current neural substrate?",
                "Tell me the verified result from the old and new cortex campaigns.",
            ),
            negatives=(
                "Do you feel different today?",
                "Compare two language models for me.",
                "Explain the cortex in biology.",
                "What is your opinion of the new interface?",
                "Write a story about a machine changing its mind.",
                "How are you doing right now?",
            ),
            features=embed_sentences,
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return None


_CORTEX_EVIDENCE_SURFACE = _build_cortex_evidence_surface()


def _request_tokens(text: str) -> frozenset[str]:
    try:
        from core.language.asking_clauses import asking_part

        text = asking_part(text)
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    return frozenset(_WORD_RE.findall(str(text or "").casefold()))


def classify_cortex_evidence_request(text: str) -> CortexEvidenceRequest | None:
    """Classify only self-questions whose answer is closed by typed evidence."""

    tokens = _request_tokens(text)
    if not tokens or not tokens.intersection(_SELF_TERMS):
        return None
    cortex_named = bool(tokens.intersection(_CORTEX_TERMS))
    mechanism_named = bool(tokens.intersection(_MECHANISM_TERMS))
    measured = bool(tokens.intersection(_MEASUREMENT_TERMS))
    compared = bool(tokens.intersection(_COMPARISON_TERMS))
    if measured and compared and (cortex_named or mechanism_named):
        return CortexEvidenceRequest.MEASURED_COMPARISON
    if measured and mechanism_named:
        return CortexEvidenceRequest.BOUNDED_MECHANISM
    if cortex_named and tokens.intersection(_IDENTITY_TERMS):
        return CortexEvidenceRequest.IDENTITY
    surface = _CORTEX_EVIDENCE_SURFACE
    if surface is not None:
        try:
            learned = surface.decide_without_waiting(str(text or ""))
        except (RuntimeError, TypeError, ValueError):
            learned = None
        if learned is True:
            if mechanism_named:
                return CortexEvidenceRequest.BOUNDED_MECHANISM
            if compared:
                return CortexEvidenceRequest.MEASURED_COMPARISON
            return CortexEvidenceRequest.IDENTITY
    return None


def render_cortex_evidence_response(
    user_message: str,
    *,
    evidence: CortexSelfEvidence | None = None,
) -> AssertionResponse | None:
    """Compose a closed cortex answer and retain its evidence through delivery."""

    request = classify_cortex_evidence_request(user_message)
    if request is None:
        return None
    observed = evidence or resolve_cortex_self_evidence()
    if observed is None:
        return None

    if request is CortexEvidenceRequest.IDENTITY:
        if not observed.resident_descriptor_sha256:
            return None
        text = (
            f"My resident cortex is {observed.resident_label} "
            f"({observed.model_type}), with {observed.total_parameters:,} parameters. "
            f"Its native context is {observed.native_context_tokens:,} tokens; the "
            f"currently qualified serving context is {observed.served_context_tokens:,}."
        )
        assertion = Assertion(
            subject="resident cortex identity",
            claim=text,
            source=SourceKind.MEASURED,
            provenance="active cortex descriptor and resident model registry",
            evidence=(observed.resident_descriptor_sha256,),
            verification=Verification.VERIFIED,
        )
        return AssertionResponse(
            family="cortex_self_evidence",
            text=text,
            assertions=(assertion,),
        )

    if request is CortexEvidenceRequest.BOUNDED_MECHANISM:
        if (
            not observed.semantic_active
            or observed.semantic_task_count <= 0
            or not observed.semantic_activation_sha256
        ):
            return None
        arms = dict(observed.semantic_exact_by_arm)
        assertion = Assertion(
            subject="active recurrent semantic tissue",
            claim=(
                "The result I can support is bounded: my active recurrent semantic tissue "
                f"is qualified as {observed.semantic_verdict}, with treatment "
                f"{arms.get('treatment', 0)}/{observed.semantic_task_count} against "
                f"ordinary decode {arms.get('ordinary_base', 0)}/{observed.semantic_task_count}, "
                f"{observed.semantic_gain_count} gains, {observed.semantic_regression_count} "
                "regressions, and lesion-dependent controls"
            ),
            source=SourceKind.MEASURED,
            provenance="active semantic-neural serving qualification",
            evidence=(observed.semantic_activation_sha256,),
            verification=Verification.VERIFIED,
        )
        text = (
            f"{assertion.render()}. That does not establish open-domain or frontier reasoning."
        )
        return AssertionResponse(
            family="cortex_self_evidence",
            text=text,
            assertions=(assertion,),
        )

    comparison = observed.measured_campaign_comparison()
    if not comparison:
        return None
    current = observed.resident_campaign()
    previous = next(
        (campaign for campaign in observed.campaigns if campaign is not current),
        None,
    )
    if current is None or previous is None:
        return None
    reduction = (
        100.0 * (previous.elapsed_seconds - current.elapsed_seconds) / previous.elapsed_seconds
    )
    assertion = Assertion(
        subject="matched-shape recurrent campaign throughput",
        claim=(
            "One difference I can actually measure is bounded campaign throughput. "
            f"The former {previous.cortex_label} run took {previous.elapsed_seconds:,.3f} "
            f"seconds; the current {current.cortex_label} run took "
            f"{current.elapsed_seconds:,.3f} seconds for the same "
            f"{current.decode_count}-decode, {current.task_count}-task campaign shape, "
            f"so the recorded run was {reduction:.1f}% faster"
        ),
        source=SourceKind.MEASURED,
        provenance="independently verified CP566 and CP1011 campaign receipts",
        evidence=(*previous.evidence_ids, *current.evidence_ids),
        verification=Verification.VERIFIED,
    )
    text = (
        f"{assertion.render()}. The cohorts were separately seeded, so that is not "
        "a model-only quality benchmark. I do not have paired evidence that the swap "
        "changed my conversational style, broad reasoning, knowledge, association "
        "speed, or subjective experience; those remain unmeasured."
    )
    return AssertionResponse(
        family="cortex_self_evidence",
        text=text,
        assertions=(assertion,),
    )


def render_cortex_evidence_reply(
    user_message: str,
    *,
    evidence: CortexSelfEvidence | None = None,
) -> str:
    """Compatibility text view over the typed assertion response."""

    response = render_cortex_evidence_response(user_message, evidence=evidence)
    return response.text if response is not None else ""


def resolve_cortex_self_evidence() -> CortexSelfEvidence | None:
    """Resolve current cortex facts only from validated runtime authorities."""

    from core.brain.llm.model_registry import (
        get_active_cortex_spec,
        resident_model_identity,
    )
    from core.brain.llm.semantic_neural_serving import (
        semantic_neural_serving_status,
    )

    spec = get_active_cortex_spec()
    if spec is None or not spec.exact_identity:
        return None
    identity = resident_model_identity()
    migration = spec.migration_contract() or {}
    evaluation = spec.evaluation() or {}
    semantic_status = semantic_neural_serving_status(spec.model_path)
    semantic_receipt = (
        semantic_status.get("receipt") if semantic_status.get("active") is True else {}
    )
    qualification = (
        semantic_receipt.get("qualification") if isinstance(semantic_receipt, dict) else {}
    )
    if not isinstance(qualification, dict):
        qualification = {}
    exact_by_arm = qualification.get("independent_exact_by_arm")
    exact_items = (
        tuple(sorted((str(name), int(score)) for name, score in exact_by_arm.items()))
        if isinstance(exact_by_arm, dict)
        else ()
    )
    p_value = qualification.get("paired_one_sided_exact_p")
    return CortexSelfEvidence(
        resident_label=str(identity.get("label") or spec.size_class or "resident"),
        model_type=str(identity.get("model_type") or "unknown"),
        total_parameters=int(identity.get("total_parameters") or 0),
        native_context_tokens=int(identity.get("native_context_window") or 0),
        served_context_tokens=int(identity.get("served_context_tokens") or 0),
        promotion_verdict=str(evaluation.get("verdict") or ""),
        identity_behavior_changed=(
            evaluation.get("identity_behavior_changed")
            if isinstance(evaluation.get("identity_behavior_changed"), bool)
            else None
        ),
        component_states=_component_states(migration.get("components")),
        semantic_active=semantic_status.get("active") is True,
        semantic_verdict=str(qualification.get("verdict") or ""),
        semantic_task_count=int(qualification.get("task_count") or 0),
        semantic_exact_by_arm=exact_items,
        semantic_gain_count=int(qualification.get("gain_count") or 0),
        semantic_regression_count=int(qualification.get("regression_count") or 0),
        semantic_p_value=float(p_value) if isinstance(p_value, (int, float)) else None,
        semantic_activation_sha256=(
            str(semantic_receipt.get("activation_sha256") or "")
            if isinstance(semantic_receipt, dict)
            else ""
        ),
        resident_descriptor_sha256=str(identity.get("descriptor_sha256") or ""),
        resident_model_path=str(spec.model_path or ""),
        campaigns=verified_cortex_campaigns(),
    )


__all__ = [
    "CortexCampaignEvidence",
    "CortexEvidenceRequest",
    "CortexSelfEvidence",
    "classify_cortex_evidence_request",
    "render_cortex_evidence_reply",
    "render_cortex_evidence_response",
    "resolve_cortex_self_evidence",
    "verified_cortex_campaigns",
]
