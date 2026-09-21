"""Requested operation views preserve their exact numerical definition."""

import numpy as np
import pytest

from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import (
    _operation_feature, _normalized_feature, _OPERATION_FEATURE_MODES,
)

CHANNELS = ('input_token_embedding', 'middle_causal_hidden', 'final_causal_hidden')
WIDTHS = (3, 5, 4)


@pytest.mark.parametrize('mode', _OPERATION_FEATURE_MODES)
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_requested_view_is_bit_identical_to_eager_definition(mode, dtype):
    hidden = np.random.default_rng(19).normal(size=(15, sum(WIDTHS))).astype(dtype)
    lexical, middle, contextual = hidden[2:9, :3], hidden[2:9, 3:8], hidden[2:9, 8:]
    mean = lambda array: np.mean(array, axis=0, dtype=np.float32)
    expected = {
        'span_mean': mean(hidden[2:9]), 'lexical_mean': mean(lexical),
        'middle_mean': mean(middle), 'middle_last': middle[-1],
        'contextual_mean': mean(contextual), 'contextual_last': contextual[-1],
        'lexical_mean_contextual_last': np.concatenate((mean(lexical), contextual[-1])),
        'lexical_mean_contextual_mean_contextual_last': np.concatenate(
            (mean(lexical), mean(contextual), contextual[-1])),
    }
    actual = _operation_feature(hidden, TokenSpan(2, 9), mode=mode,
        hidden_channels=CHANNELS, hidden_channel_widths=WIDTHS)
    np.testing.assert_array_equal(actual, _normalized_feature(expected[mode]))


def test_last_view_does_not_reduce_unused_channels(monkeypatch):
    hidden = np.ones((4, sum(WIDTHS)), dtype=np.float32)
    monkeypatch.setattr(np, 'mean', lambda *a, **k: pytest.fail('unused view reduced'))
    value = _operation_feature(hidden, TokenSpan(1, 3), mode='contextual_last',
        hidden_channels=CHANNELS, hidden_channel_widths=WIDTHS)
    assert value.shape == (4,)


@pytest.mark.parametrize('mode', ['middle_mean', 'middle_last', 'unknown'])
def test_missing_middle_and_unknown_modes_remain_unsupported(mode):
    with pytest.raises(ValueError, match='unsupported'):
        _operation_feature(np.ones((3, 7)), TokenSpan(0, 2), mode=mode,
            hidden_channels=(CHANNELS[0], CHANNELS[2]), hidden_channel_widths=(3, 4))
