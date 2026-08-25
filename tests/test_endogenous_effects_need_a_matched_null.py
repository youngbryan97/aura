"""An effect with no comparison is not a small effect. It is an unmeasured one.

Every intervention here reports the same-sized intervention on peer dimensions
beside it, because "moving this dimension changed the bias" and "this head
reacts to any perturbation" produce identical numbers until the second is
ruled out.
"""

from __future__ import annotations

import numpy as np

from core.brain.llm.endogenous_intervention import (
    build_arm,
    channel_influence_map,
    measure_ablation,
    measure_contrast,
    measure_intervention,
    sweep_dimension,
    text_distance,
)
from core.brain.llm.endogenous_state import (
    CHANNELS,
    FEATURE_INDEX,
    STATE_DIM,
    empty_state,
    layout_digest,
)
from core.brain.llm.endogenous_vocab_head import EndogenousVocabHead


def _head(seed: int = 1, vocab: int = 128) -> EndogenousVocabHead:
    rng = np.random.default_rng(seed)
    return EndogenousVocabHead(
        weights=rng.standard_normal((vocab, STATE_DIM)).astype(np.float32) * 0.2,
        bias=np.zeros(vocab, dtype=np.float32),
        vocab_size=vocab,
        layout=layout_digest(),
        tokenizer="sig",
        trained=True,
    )


def _state():
    return empty_state().do(
        **{
            "uncertainty.confidence": 0.5,
            "goal.active": 1.0,
            "goal.priority": 0.6,
            "memory.recall_hits": 0.5,
        }
    )


def test_without_a_head_an_effect_reports_no_bias_shift():
    effect = measure_intervention(_state(), "uncertainty.confidence", 0.95)
    assert effect.bias_shift == 0.0
    assert effect.exceeds_null is False
    assert "UNCERTAINTY" in effect.code_lines_moved


def test_an_effect_with_no_nulls_never_claims_to_exceed_them():
    effect = measure_intervention(
        _state(), "uncertainty.confidence", 0.95, head=_head(), null_count=0
    )
    assert effect.null_bias_shifts == ()
    assert effect.exceeds_null is False


def test_a_contrast_moves_the_code_even_when_the_state_already_sat_there():
    """Comparing against the state as found reports nothing when it matches."""
    already_high = empty_state().do(**{"uncertainty.confidence": 0.95})
    single = measure_intervention(already_high, "uncertainty.confidence", 0.95)
    contrast = measure_contrast(already_high, "uncertainty.confidence", 0.05, 0.95)
    assert single.code_lines_moved == ()
    assert contrast.code_lines_moved == ("UNCERTAINTY",)


def test_the_nulls_come_from_other_channels():
    effect = measure_intervention(
        _state(), "uncertainty.confidence", 0.9, head=_head(), null_count=6
    )
    assert len(effect.null_bias_shifts) == 6
    assert effect.bias_shift > 0.0


def test_an_ablation_is_compared_against_ablating_every_other_channel():
    effect = measure_ablation(_state(), "memory", head=_head())
    assert len(effect.null_bias_shifts) == len(CHANNELS) - 1
    assert effect.feature == "channel:memory"


def test_ablating_an_unknown_channel_raises():
    import pytest

    with pytest.raises(KeyError):
        measure_ablation(_state(), "not_a_channel")


def test_the_influence_map_names_the_channels_a_head_ignores():
    head = _head(seed=4)
    state = _state()
    influence = channel_influence_map(state, head)
    assert set(influence["channels_with_influence"]) | set(
        influence["channels_ignored"]
    ) == set(CHANNELS)
    # Channels nothing answered for cannot move a bias, and must be reported
    # as ignored rather than quietly counted as influential.
    assert "attention" in influence["channels_ignored"]


def test_a_sweep_records_where_the_readout_changed():
    result = sweep_dimension(
        _state(), "uncertainty.confidence", [0.05, 0.5, 0.95], head=_head()
    )
    moved = [step["moved_from_previous"] for step in result["steps"]]
    assert moved[0] == []
    assert any(step for step in moved[1:]), "a full sweep moved nothing"
    assert all(step["bias_norm"] is not None for step in result["steps"])


def test_an_arm_carries_its_own_interventions():
    arm = build_arm("treated", _state().do(**{"goal.priority": 0.1}), head=_head())
    payload = arm.as_dict()
    assert payload["has_bias"] is True
    assert any(i["feature"] == "goal.priority" for i in payload["interventions"])


def test_text_distance_is_zero_for_identical_text():
    assert text_distance("a b c", "a b c") == 0.0
    assert text_distance("", "") == 0.0
    assert text_distance("a b", "c d") == 1.0


def test_intervening_on_an_unknown_dimension_raises():
    import pytest

    with pytest.raises(KeyError):
        measure_intervention(_state(), "nope.nope", 1.0)
    assert "uncertainty.confidence" in FEATURE_INDEX
