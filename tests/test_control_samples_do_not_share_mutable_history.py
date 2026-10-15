"""Selecting the responding body cannot make one captured velocity count twice."""
import pytest

from core.agency.which_one_answers_to_her import _control_sample_custody_invariant, _Speeds

pytestmark = pytest.mark.unit


def test_selection_copies_trial_history_instead_of_sharing_live_sample_lists():
    from tests.test_which_thing_answers_to_her import _run

    hers = _run(trying=True, ball_follows_keys=False)
    assert hers.number == 1
    independent = hers._by_thing[1]
    inherited = independent.timed.keys() & hers._hers.timed.keys()
    assert inherited
    for press in inherited:
        assert independent.timed[press] is not hers._hers.timed[press]
    for record in (independent, hers._hers):
        for samples in record.timed.values():
            times = [sample[0] for sample in samples]
            assert len(times) == len(set(times))
    press = next(iter(inherited))
    before = list(hers._hers.timed[press])
    independent.timed[press].clear()
    assert hers._hers.timed[press] == before


@pytest.mark.parametrize('extra', [.4, .3, float('nan'), float('inf')])
def test_three_measurements_and_a_duplicate_cannot_supply_four_samples_for_acceleration(extra):
    record = _Speeds()
    for at in (.2, .3, .4, extra):
        record.add('q', 0., at * 10., press=0., when=at)
    assert len(record.by_press[('q', 0.)]) == 3
    assert not record.pushes()


def test_sample_custody_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _control_sample_custody_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'control_samples_have_distinct_times')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
