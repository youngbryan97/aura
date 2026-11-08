"""What her ear wrote down with no more certainty than it had of babble is not given to her as heard.

Measured 8 November 2026 through her ear's own streaming recognizer: no worded five seconds of a narration played
backwards scored above 0.871. The same night a cartoon's invented language came out as slurs she said back.
"""

from core.skills.watching_along import HER_EAR_SURE_ABOVE, her_ear_was_sure, how_sure


def _phrase(words, confidence, tokens=4):
    return {"at": 1.0, "words": words, "confidence": confidence, "tokens": tokens}


def test_babble_written_down_as_words_is_not_heard():
    babble = [_phrase("her in a sense value", 0.79), _phrase("of it, sure is perish", 0.78)]
    assert not her_ear_was_sure(babble)


def test_speech_her_ear_was_sure_of_is():
    speech = [_phrase("each with 100 to 1,000", 0.96), _phrase("billion stars in it", 0.95)]
    assert her_ear_was_sure(speech)


def test_too_few_words_to_go_on_and_an_ear_that_kept_no_certainty_are_left_as_they_are():
    assert her_ear_was_sure([_phrase("Oh.", 0.5)])
    assert her_ear_was_sure([{"at": 1.0, "words": "a line from an older ear with no score"}])


def test_each_phrase_counts_by_its_tokens():
    mixed = [_phrase("a long clear stretch of narration here", 0.96, tokens=12), _phrase("ah", 0.5, tokens=1)]
    assert how_sure(mixed) > HER_EAR_SURE_ABOVE
