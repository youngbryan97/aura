"""Source method arithmetic reaches current probe ordering with its uncertainty."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.how_a_game_works import STEERED, Answer, HowItWorks, understood
from core.agency.probes_from_source_arithmetic import (
    _conditional_source_probe_invariant,
    source_motion_trial_keys,
)
from core.agency.searching_before_a_body_is_known import PixelSearchState, eligible_pixel_search
from core.cognition.a_guide_to_a_place import Guide
from core.perception.motion_inside_source_methods import MethodMotion, SourceMethodMotion


def _guide():
    vector = MethodMotion('j', 'Opaque.z', 'vector_increment', 'Opaque.p', 'Opaque.p.q', ('Opaque.p.r', 'Opaque.p.s'),
                          (-1., 1.), (-18., 18.), (), ('unresolved control flow and physical role',), (1,))
    turn = replace(vector, key='k', kind='turn')
    motion = SourceMethodMotion((turn, vector), source='swf:' + 'a' * 64)
    return SimpleNamespace(code_says=SimpleNamespace(source_motion=motion), controls={'j': object(), 'k': object()})


def test_unknown_units_and_sign_can_only_order_already_authorized_probe_keys():
    guide = _guide()
    assert _conditional_source_probe_invariant() == ()
    assert source_motion_trial_keys(guide, ('k', 'j')) == ('j',)
    assert source_motion_trial_keys(guide, ('k',)) == ()
    how = HowItWorks(me=Answer(STEERED, 1.), body_keys_from_code=('k', 'j'), source_trial_keys=('j',))
    run = SimpleNamespace(_pixel_search=PixelSearchState(), how=how, keys=['k', 'j'], trying=8, held='', body_trial=None)
    hers, moves = SimpleNamespace(number=None, follows_pointer=False), SimpleNamespace(seeing=True)
    assert eligible_pixel_search(run, hers, moves, 10., 30.) == ('j', 'k')
    assert hers.number is None and not guide.code_says.source_motion.effects[1].unconditional


@pytest.mark.parametrize('problem', ['missing', 'bad-source', 'incomplete', 'nonfinite', 'zero', 'not-a-body-control'])
def test_unavailable_or_unbound_source_arithmetic_cannot_reorder_probes(problem):
    guide = _guide()
    reading = guide.code_says.source_motion
    if problem == 'missing':
        guide.code_says.source_motion = None
    elif problem == 'bad-source':
        guide.code_says.source_motion = replace(reading, source='unowned')
    elif problem == 'incomplete':
        guide.code_says.source_motion = replace(reading, unknown=('incomplete source',))
    elif problem in ('nonfinite', 'zero'):
        vector = replace(reading.effects[1], steps=(float('nan') if problem == 'nonfinite' else 0.,))
        guide.code_says.source_motion = replace(reading, effects=(vector,))
    else:
        del guide.controls['j']
    assert source_motion_trial_keys(guide, ('j', 'k')) == ()


def test_assigned_source_survives_the_real_program_guide_cache_and_probe_path():
    from core.perception.reading_a_program import read_program
    from core.skills.sovereign_browser_the_program import _from_kept, _kept

    path = Path.home() / '.aura/flash-campaign/offline-bundle/programs/9.swf'
    if not path.exists():
        pytest.skip('the assigned public source is unavailable')
    read = read_program(path.read_bytes())
    assert read.action_sources.complete and read.action_sources.source == read.plays.source_motion.source
    assert len(read.plays.source_motion.effects) == 4
    assert all(not effect.unconditional for effect in read.plays.source_motion.effects)
    assert not read.world.hers.drives and not read.world.hers.turns and not read.world.hers.paces
    restored = _from_kept(_kept(read))
    assert restored.plays.source_motion == read.plays.source_motion
    guide = Guide()
    guide.take_in_program(restored, 1.)
    how = understood(guide)
    assert how.source_trial_keys == ('up', 'down')
    assert how.translating_keys_from_code == ()
