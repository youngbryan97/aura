"""Cross-family and cross-geometry transfer of one neural operation head."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from core.learning.semantic_program_basis import (
    establish_semantic_training_representation_compatibility,
)
from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
from core.learning.semantic_program_feature_materialization import LoadedSemanticFeatureBundle
from core.learning.semantic_program_transducer import LinearClassifierHead, _fit_classifier

SEMANTIC_OPERATION_TRANSFER_SCHEMA: Final = "aura.semantic_operation_transfer.v1"
_ARITHMETIC_PRIMITIVES: Final = frozenset({"add", "sub", "mul", "idiv"})
_FINAL_CHANNEL: Final = "final_causal_hidden"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SemanticOperationObservation:
    family: str
    split: str
    example_id: str
    step: int
    label: str
    feature: np.ndarray
    token_surface: tuple[int, ...]
    geometry_feature: np.ndarray

    def __post_init__(self) -> None:
        feature = np.asarray(self.feature, dtype=np.float32).reshape(-1)
        geometry = np.asarray(self.geometry_feature, dtype=np.float32).reshape(-1)
        if (
            not self.family
            or self.split not in {"train", "validation", "test"}
            or not self.example_id
            or type(self.step) is not int
            or self.step < 0
            or self.label not in _ARITHMETIC_PRIMITIVES
            or feature.size < 1
            or geometry.shape != (5,)
            or not np.all(np.isfinite(feature))
            or not np.all(np.isfinite(geometry))
            or not self.token_surface
        ):
            raise ValueError("semantic operation observation is invalid")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "geometry_feature", geometry)


def _normalized_mean(hidden: np.ndarray) -> np.ndarray:
    vector = np.mean(hidden, axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def operation_observations_from_bundle(
    family: str,
    bundle: LoadedSemanticFeatureBundle,
) -> tuple[SemanticOperationObservation, ...]:
    """Read only final-channel operation-span evidence from a verified bundle."""

    examples = training_examples_from_feature_bundle(bundle)
    result: list[SemanticOperationObservation] = []
    for raw, example in zip(bundle.examples, examples, strict=True):
        if _FINAL_CHANNEL not in example.hidden_channels:
            raise ValueError("semantic operation transfer has no final hidden channel")
        channel_index = example.hidden_channels.index(_FINAL_CHANNEL)
        lower = sum(example.hidden_channel_widths[:channel_index])
        upper = lower + example.hidden_channel_widths[channel_index]
        final_hidden = example.hidden_states[:, lower:upper]
        for step, instruction in enumerate(example.ir.instructions):
            if instruction.op not in _ARITHMETIC_PRIMITIVES:
                continue
            span = instruction.operation_span
            result.append(
                SemanticOperationObservation(
                    family=family,
                    split=example.split,
                    example_id=str(raw.metadata["example_id"]),
                    step=step,
                    label=instruction.op,
                    feature=_normalized_mean(final_hidden[span.start : span.end]),
                    token_surface=tuple(example.ir.source_token_ids[span.start : span.end]),
                    geometry_feature=np.asarray(
                        (
                            example.ir.n_inputs,
                            len(example.ir.instructions),
                            len(instruction.args),
                            step,
                            span.end - span.start,
                        ),
                        dtype=np.float32,
                    ),
                )
            )
    if not result:
        raise ValueError(f"semantic operation family has no arithmetic evidence: {family}")
    return tuple(result)


def _lesion(head: LinearClassifierHead) -> LinearClassifierHead:
    return LinearClassifierHead(head.labels, np.zeros_like(head.weight), head.bias)


def _permuted_labels(labels: Sequence[str]) -> list[str]:
    support = sorted(set(labels))
    if len(support) < 2:
        raise ValueError("semantic operation permutation lacks label support")
    mapping = {label: support[(index + 1) % len(support)] for index, label in enumerate(support)}
    return [mapping[label] for label in labels]


def _fit(rows: Sequence[SemanticOperationObservation], *, geometry: bool = False) -> LinearClassifierHead:
    selected = tuple(item for item in rows if item.split == "train")
    if {item.label for item in selected} != _ARITHMETIC_PRIMITIVES:
        raise ValueError("semantic operation source lacks complete arithmetic support")
    return _fit_classifier(
        np.stack(
            [item.geometry_feature if geometry else item.feature for item in selected]
        ),
        [item.label for item in selected],
    )


def _fit_permuted(rows: Sequence[SemanticOperationObservation]) -> LinearClassifierHead:
    selected = tuple(item for item in rows if item.split == "train")
    return _fit_classifier(
        np.stack([item.feature for item in selected]),
        _permuted_labels([item.label for item in selected]),
    )


def _token_lookup(
    training: Sequence[SemanticOperationObservation],
) -> tuple[dict[tuple[int, ...], str], str]:
    votes: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
    priors: Counter[str] = Counter()
    for item in training:
        if item.split == "train":
            votes[item.token_surface][item.label] += 1
            priors[item.label] += 1
    lookup = {
        surface: sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
        for surface, counts in votes.items()
    }
    fallback = sorted(priors.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
    return lookup, fallback


def _exact_pair(treatment: Mapping[str, bool], control: Mapping[str, bool]) -> dict[str, Any]:
    if treatment.keys() != control.keys():
        raise ValueError("semantic operation paired programs differ")
    treatment_only = sum(treatment[key] and not control[key] for key in treatment)
    control_only = sum(control[key] and not treatment[key] for key in treatment)
    discordant = treatment_only + control_only
    p_value = (
        sum(math.comb(discordant, value) for value in range(treatment_only, discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": discordant,
        "one_sided_exact_p": p_value,
    }


def _evaluate_direction(
    *,
    source: Sequence[SemanticOperationObservation],
    target: Sequence[SemanticOperationObservation],
) -> dict[str, Any]:
    treatment = _fit(source)
    heads = {
        "treatment": treatment,
        "coefficient_lesion": _lesion(treatment),
        "label_permutation": _fit_permuted(source),
        "geometry_only": _fit(source, geometry=True),
    }
    lookup, fallback = _token_lookup(source)
    source_surfaces = {
        item.token_surface for item in source if item.split == "train"
    }
    splits: dict[str, Any] = {}
    for split in ("validation", "test"):
        selected = tuple(item for item in target if item.split == split)
        if not selected:
            raise ValueError(f"semantic operation target split is empty: {split}")
        predictions: dict[str, list[str]] = {
            arm: [
                head.predict(item.geometry_feature if arm == "geometry_only" else item.feature)[0]
                for item in selected
            ]
            for arm, head in heads.items()
        }
        predictions["token_lookup"] = [
            lookup.get(item.token_surface, fallback) for item in selected
        ]
        exact_by_arm: dict[str, dict[str, bool]] = {}
        for arm, values in predictions.items():
            by_example: dict[str, list[bool]] = defaultdict(list)
            for item, prediction in zip(selected, values, strict=True):
                by_example[item.example_id].append(prediction == item.label)
            exact_by_arm[arm] = {
                key: all(decisions) for key, decisions in sorted(by_example.items())
            }
        splits[split] = {
            "operation_count": len(selected),
            "program_count": len(exact_by_arm["treatment"]),
            "surface_overlap_count": sum(
                item.token_surface in source_surfaces for item in selected
            ),
            "surface_unseen_count": sum(
                item.token_surface not in source_surfaces for item in selected
            ),
            "arms": {
                arm: {
                    "operation_exact": sum(
                        prediction == item.label
                        for item, prediction in zip(selected, values, strict=True)
                    ),
                    "program_exact": sum(exact_by_arm[arm].values()),
                }
                for arm, values in predictions.items()
            },
            "paired_program_tests": {
                arm: _exact_pair(exact_by_arm["treatment"], exact_by_arm[arm])
                for arm in predictions
                if arm != "treatment"
            },
        }
    return {
        "source_training_operation_count": sum(item.split == "train" for item in source),
        "source_training_surface_count": len(source_surfaces),
        "splits": splits,
    }


def run_semantic_operation_transfer(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
) -> dict[str, Any]:
    """Run fixed D-to-A/B and A+B-to-D semantic transfer directions."""

    required = {"arithmetic", "fork_join", "sequence_binary"}
    if set(bundles) != required:
        raise ValueError("semantic operation transfer bundle inventory differs")
    compatibility = establish_semantic_training_representation_compatibility(
        {name: bundle.manifest for name, bundle in bundles.items()}
    )
    observations = {
        name: operation_observations_from_bundle(name, bundle)
        for name, bundle in bundles.items()
    }
    directions = {
        "sequence_to_arithmetic": _evaluate_direction(
            source=observations["sequence_binary"],
            target=observations["arithmetic"],
        ),
        "sequence_to_fork_join": _evaluate_direction(
            source=observations["sequence_binary"],
            target=observations["fork_join"],
        ),
        "arithmetic_fork_join_to_sequence": _evaluate_direction(
            source=(*observations["arithmetic"], *observations["fork_join"]),
            target=observations["sequence_binary"],
        ),
    }
    body = {
        "schema": SEMANTIC_OPERATION_TRANSFER_SCHEMA,
        "feature_manifest_sha256s": {
            name: bundle.manifest["manifest_sha256"]
            for name, bundle in sorted(bundles.items())
        },
        "representation_compatibility": compatibility,
        "directions": directions,
        "gold_operation_spans_available": True,
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_evaluation": False,
        "family_identity_available_to_classifier": False,
        "geometry_available_to_treatment_classifier": False,
        "serving_authority": False,
        "claim_boundary": (
            "bounded gold-operation-span semantic transfer across synthetic arithmetic "
            "and sequence language; not end-to-end program acquisition"
        ),
    }
    return {**body, "report_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_OPERATION_TRANSFER_SCHEMA",
    "SemanticOperationObservation",
    "operation_observations_from_bundle",
    "run_semantic_operation_transfer",
]
