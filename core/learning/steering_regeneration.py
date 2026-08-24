"""Regenerating CAA steering vectors for a checkpoint that replaced the last one.

Aura's affective steering is causal: a vector is added to the residual stream at
chosen layers and the tokens change. That is why it cannot be carried across a
model swap. The 45 retained vectors are 5120-wide and the new checkpoint's
residual stream is also 5120 wide, so every one of them loads. They describe
directions in a space that no longer exists.

The repository already refuses to serve them --
``model_bound_steering.resolve_active_generation`` reports
``active_cortex_descriptor_mismatch`` -- and the tissue inventory already
classifies the bundle ``activation_basis`` / ``retrain``. What was missing is
the other half: the plan for what "retrain" concretely means here, written
before capture rather than discovered during it.

Two things are specific to this checkpoint and are the reason this module is not
just the old recipe pointed at a new path.

**Target layers are no longer interchangeable.** The extraction recipe picks a
depth band. On a dense checkpoint every layer in that band is the same kind of
layer. On the 27B, three in four carry ``linear_attn``, whose state advances
along the sequence rather than being re-read from a K/V cache, so an injection
there propagates differently from one at an attention layer. The plan records
the kind of every target layer, so a later result can be read against where it
was actually applied.

**Authority is not granted by capture.** A regenerated vector that exists, is
the right width, and was extracted from the right model has still not been shown
to do anything. Serving authority requires a causal A/B against a matched no-op
and a lesion that removes the effect, on this checkpoint. Anything less is a
vector with good provenance and unmeasured behaviour, which is exactly what
looks safest and is not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

STEERING_REGENERATION_SCHEMA: Final = "aura.caa.steering_regeneration_plan.v1"

#: The depth band the extraction recipe targets, as fractions of total depth.
#: Kept as the recipe's own number rather than re-derived here.
TARGET_LAYER_FRACTION: Final = (0.40, 0.65)

#: Evidence a regenerated vector needs before it may steer a served answer.
#: Every entry is a measurement on the new checkpoint; none may be inherited.
REQUIRED_EVIDENCE: Final = (
    "extraction_bound_to_active_descriptor",
    "causal_ab_vs_matched_noop",
    "lesion_removes_the_effect",
    "no_regression_on_the_control_prompts",
)


class SteeringRegenerationError(RuntimeError):
    """The plan does not describe the checkpoint it claims to."""


@dataclass(frozen=True)
class TargetLayer:
    """One injection site, and what kind of layer it turned out to be."""

    index: int
    carries_attention: bool

    @property
    def kind(self) -> str:
        return "full_attention" if self.carries_attention else "linear_attention"


@dataclass
class SteeringRegenerationPlan:
    """What will be captured, from which model, and what it may not yet do."""

    schema: str
    descriptor_fingerprint: str
    model_path: str
    num_hidden_layers: int
    hidden_size: int
    target_layers: tuple[TargetLayer, ...]
    dimensions: tuple[str, ...]
    reuses_previous_vectors: bool = False
    serving_authority: bool = False
    required_evidence: tuple[str, ...] = REQUIRED_EVIDENCE
    superseded_bundle: dict[str, Any] = field(default_factory=dict)

    def attention_targets(self) -> tuple[int, ...]:
        return tuple(t.index for t in self.target_layers if t.carries_attention)

    def linear_targets(self) -> tuple[int, ...]:
        return tuple(t.index for t in self.target_layers if not t.carries_attention)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_layers"] = [
            {"index": t.index, "kind": t.kind} for t in self.target_layers
        ]
        payload["attention_targets"] = list(self.attention_targets())
        payload["linear_targets"] = list(self.linear_targets())
        payload["expected_vector_count"] = len(self.target_layers) * len(
            self.dimensions
        )
        return payload


def resolve_target_layers(
    geometry: Any, fraction: tuple[float, float] = TARGET_LAYER_FRACTION
) -> tuple[TargetLayer, ...]:
    """The depth band, annotated with what each layer in it actually is."""
    low, high = fraction
    if not 0.0 <= low < high <= 1.0:
        raise SteeringRegenerationError("target layer fraction is not a band in [0,1]")
    layers = int(geometry.num_hidden_layers)
    start = int(layers * low)
    stop = max(start + 1, int(layers * high))
    return tuple(
        TargetLayer(index=index, carries_attention=geometry.carries_attention(index))
        for index in range(start, min(stop, layers))
    )


def build_plan(
    *,
    descriptor_fingerprint: str,
    model_path: str,
    geometry: Any,
    hidden_size: int,
    dimensions: tuple[str, ...],
    superseded_bundle: dict[str, Any] | None = None,
) -> SteeringRegenerationPlan:
    if not descriptor_fingerprint:
        raise SteeringRegenerationError("a plan must name the checkpoint it targets")
    if not dimensions:
        raise SteeringRegenerationError("a plan must name the dimensions to capture")
    return SteeringRegenerationPlan(
        schema=STEERING_REGENERATION_SCHEMA,
        descriptor_fingerprint=descriptor_fingerprint,
        model_path=model_path,
        num_hidden_layers=int(geometry.num_hidden_layers),
        hidden_size=int(hidden_size),
        target_layers=resolve_target_layers(geometry),
        dimensions=tuple(dimensions),
        reuses_previous_vectors=False,
        serving_authority=False,
        superseded_bundle=dict(superseded_bundle or {}),
    )


def authority_errors(
    plan: dict[str, Any], evidence: dict[str, Any] | None = None
) -> list[str]:
    """Why a regenerated bundle may not steer a served answer yet.

    Called with no evidence, this returns the full requirement list. That is the
    intended reading before a capture runs: nothing has been shown, so nothing
    is granted.
    """
    errors: list[str] = []
    if plan.get("reuses_previous_vectors"):
        errors.append("plan_reuses_vectors_from_another_checkpoint")
    if not plan.get("descriptor_fingerprint"):
        errors.append("plan_names_no_checkpoint")

    present = evidence or {}
    for requirement in plan.get("required_evidence", REQUIRED_EVIDENCE):
        if present.get(requirement) is not True:
            errors.append(f"missing_evidence:{requirement}")

    fingerprint = present.get("descriptor_fingerprint")
    if fingerprint and fingerprint != plan.get("descriptor_fingerprint"):
        errors.append("evidence_measured_on_a_different_checkpoint")
    return errors


def may_serve(plan: dict[str, Any], evidence: dict[str, Any] | None = None) -> bool:
    """Fail closed. Absent evidence is never a pass."""
    return not authority_errors(plan, evidence)
