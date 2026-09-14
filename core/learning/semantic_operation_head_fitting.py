"""Fitting the multi-view operation head.

One fitter, taken out of `semantic_program_transducer` because that module
reached the 2000-line ceiling a module not already in the baseline is never
grandfathered past. It is a leaf: it reads training examples and returns a
head, and nothing in it decides anything the rest of the module needs to see.

The names it needs come back through the module it came out of, so a test that
patches one of them there still reaches this.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # For the signature only. At run time the same names are imported inside
    # the function, from the module this came out of, so the cycle never forms
    # and a patch installed there is still the one this sees.
    from core.learning.semantic_program_transducer import (
        MultiViewClassifierHead,
        SemanticTransducerTrainingExample,
    )


def _fit_multiview_operation_head(
    rows: Sequence[tuple[SemanticTransducerTrainingExample, int]],
    *,
    hidden_channels: tuple[str, ...],
    hidden_channel_widths: tuple[int, ...],
) -> tuple[MultiViewClassifierHead, dict[str, Any]]:
    """Select semantic views using construction-held-out training evidence only."""
    # Per call, from the module this came out of. A module-level import
    # would be a cycle, module __getattr__ does not answer a bare global
    # lookup inside a function, and a name bound once stops honouring a
    # patch the caller installs.
    from core.learning.semantic_program_transducer import (  # noqa: PLC0415
        LinearClassifierHead,
        MultiViewClassifierHead,
        SemanticTransducerTrainingExample,
        _MAX_OPERATION_VIEWS,
        _MIDDLE_CAUSAL_CHANNEL,
        _OPERATION_FEATURE_MODES_V2,
        _OPERATION_FEATURE_MODES_V3,
        _fit_classifier,
        _multiview_prediction,
        _operation_feature,
        _operation_feature_width,
    )


    constructions = sorted({item.construction_id for item, _ in rows})
    labels = sorted({item.ir.instructions[step].op for item, step in rows})
    if len(constructions) < 2 or len(labels) < 2:
        raise ValueError("semantic multiview selection lacks construction or label support")
    candidate_modes = (
        _OPERATION_FEATURE_MODES_V3
        if _MIDDLE_CAUSAL_CHANNEL in hidden_channels
        else _OPERATION_FEATURE_MODES_V2
    )
    candidates = tuple(
        modes
        for count in range(1, min(_MAX_OPERATION_VIEWS, len(candidate_modes)) + 1)
        for modes in combinations(candidate_modes, count)
    )
    labels_by_row = tuple(item.ir.instructions[step].op for item, step in rows)
    features_by_mode = {
        mode: tuple(
            _operation_feature(
                item.hidden_states,
                item.ir.instructions[step].operation_span,
                mode=mode,
                hidden_channels=hidden_channels,
                hidden_channel_widths=hidden_channel_widths,
            )
            for item, step in rows
        )
        for mode in candidate_modes
    }
    fold_indices: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    fold_heads: dict[tuple[str, str], LinearClassifierHead] = {}
    for held_out_construction in constructions:
        fit_indices = tuple(
            index
            for index, (item, _) in enumerate(rows)
            if item.construction_id != held_out_construction
        )
        held_out_indices = tuple(
            index
            for index, (item, _) in enumerate(rows)
            if item.construction_id == held_out_construction
        )
        if {labels_by_row[index] for index in fit_indices} != set(labels) or not held_out_indices:
            continue
        fold_indices[held_out_construction] = (fit_indices, held_out_indices)
        for mode in candidate_modes:
            fold_heads[(held_out_construction, mode)] = _fit_classifier(
                np.stack([features_by_mode[mode][index] for index in fit_indices]),
                [labels_by_row[index] for index in fit_indices],
            )
    scored: list[tuple[int, int, int, tuple[str, ...]]] = []
    for modes in candidates:
        correct = 0
        total = 0
        valid = True
        for held_out_construction in constructions:
            fold = fold_indices.get(held_out_construction)
            if fold is None:
                valid = False
                break
            _, held_out_indices = fold
            heads = tuple(fold_heads[(held_out_construction, mode)] for mode in modes)
            for index in held_out_indices:
                features = tuple(features_by_mode[mode][index] for mode in modes)
                correct += int(_multiview_prediction(heads, features) == labels_by_row[index])
                total += 1
        if valid and total:
            feature_width = sum(
                _operation_feature_width(
                    mode,
                    hidden_channels=hidden_channels,
                    hidden_channel_widths=hidden_channel_widths,
                )
                for mode in modes
            )
            scored.append((correct, -len(modes), -feature_width, modes))
    if not scored:
        raise ValueError("semantic multiview construction folds are not identifiable")
    correct, _, _, selected_modes = max(scored)
    selected_heads = tuple(
        _fit_classifier(
            np.stack(features_by_mode[mode]),
            labels_by_row,
        )
        for mode in selected_modes
    )
    receipt = {
        "modes": list(selected_modes),
        "leave_one_construction_out_correct": correct,
        "leave_one_construction_out_total": len(rows),
        "candidate_ensembles_evaluated": len(scored),
    }
    return MultiViewClassifierHead(selected_modes, selected_heads), receipt
