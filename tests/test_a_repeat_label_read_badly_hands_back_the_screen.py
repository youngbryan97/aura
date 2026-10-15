"""Navigation and the realtime actor share the bounded reading of a repeat label."""
import pytest

from core.agency.playing_as_it_happens import SAID_IT_IS_OVER, _Run, _said_it_is_over
from core.language.how_a_game_ended import how_it_ended, says_a_round_is_over

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('label', ['TRYJAGAIN J', 'PLAYAGAlN', 'TRY ABAIN', 'UUTRY-AGAIN'])
def test_an_asserted_reading_with_counters_and_a_damaged_repeat_label_ends_play(label):
    words = 'Attempts: 3 Accuracy: 0% ' + label
    assert says_a_round_is_over(words)
    assert how_it_ended(words) != 'won'
    run = _Run(began=10., keys=[])
    run.words_read = [(10.1, 'Attempts: 0'), (14., words)]
    assert _said_it_is_over(run) == SAID_IT_IS_OVER
    assert run.over_read_at == 14.


@pytest.mark.parametrize('words', [
    'If you miss, TRYJAGAIN J', 'You may PLAYAGAlN', 'Previous game TRYJAGAIN J',
    'Your turn. TRYJAGAIN J', 'Accuracy: 0% Training', 'Please play the piano',
])
def test_rules_past_results_and_current_turns_do_not_end_play(words):
    assert not says_a_round_is_over(words)


def test_a_repeat_label_already_present_as_play_began_is_not_a_new_end():
    run = _Run(began=10., keys=[])
    run.words_read = [(10.1, 'TRYJAGAIN J'), (14., 'TRYJAGAIN J')]
    assert not _said_it_is_over(run)


def test_the_bounded_reading_and_its_claim_are_measured():
    from core.language.how_a_game_ended import _damaged_repeat_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _damaged_repeat_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'damaged_repeat_labels_supply_no_winner')
    assert check.predict(None) is True
