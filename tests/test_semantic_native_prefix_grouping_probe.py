"""The numeric probe preserves exactly the frozen token supervision."""

import pytest

from tools.probe_semantic_native_prefix_grouping import supervision_sequences


def test_probe_preserves_tokens_and_identity_without_retokenization():
    row = {"source": "s", "program_sha256": "p", "tokens": [1, 2, 3, 4],
           "continuation_start": 2, "semantic_positions": [2, 3]}
    sequence = supervision_sequences({"rows": [row]})[("s", "p")]
    assert sequence.tokens == (1, 2, 3, 4)
    assert sequence.continuation_start == 2
    assert sequence.semantic_positions == (2, 3)
    for rows in ([], [row, row]):
        with pytest.raises(ValueError, match="unique"):
            supervision_sequences({"rows": rows})
