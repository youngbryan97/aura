"""Quotation offsets preserve speech without treating contractions as delimiters."""

import pytest

from core.language.typography import quotation_spans


@pytest.mark.parametrize("text,quotes", [
    ('You said "First sentence. Second sentence." Then answered.', ['"First sentence. Second sentence."']),
    ("You said 'I don't want that.'", ["'I don't want that.'"]),
    ("You're saying \u201cI don't want that.\u201d", ["\u201cI don't want that.\u201d"]),
    ("The readers' notes say \u2018we\u2019re ready.\u2019", ["\u2018we\u2019re ready.\u2019"]),
    ('She said "Call it \'amber\' today."', ['"Call it \'amber\' today."']),
    ('She said "Call it \\"amber\\" today."', ['"Call it \\"amber\\" today."']),
    ('"First." And "Second."', ['"First."', '"Second."']),
    ('An unfinished "quotation. More text.', []),
])
def test_balanced_speech_offsets(text, quotes):
    assert [text[start:end] for start, end in quotation_spans(text)] == quotes
