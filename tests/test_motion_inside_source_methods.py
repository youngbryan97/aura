"""The source's arithmetic survives opaque names without inventing branch proof."""
import json
import math
from types import SimpleNamespace

import pytest

from core.perception.motion_inside_source_methods import (
    DampedRecovery,
    kept,
    motion_in_source_methods,
    unkept,
)


def _method(name, params, *lines):
    return [f'  // anonymous function ({params})', *('  ' + line for line in lines),
            f'{name} = function ({params}) {{...}}']


def _source():
    return [
        *_method('Opaque.a0', '', 'this.C1 = 1', 'this.C2 = 10', 'this.C3 = 0', 'this.C4 = 1',
                 'this.C5 = 0.95', 'this.C6 = 0.5'),
        *_method('Opaque.a1', 'choice', 'var thing = this.part',
                 'var q = (Math.cos(((thing.f * 3.14159) / 180)) * this.C1)',
                 'var r = (Math.sin(((thing.f * 3.14159) / 180)) * this.C1)',
                 'if !(choice == this.C4) goto +40', 'q = (q * -1)', 'r = (r * -1)',
                 'var nextq = (thing.u + q)', 'var nextr = (thing.v + r)',
                 'thing.u = nextq', 'thing.v = nextr'),
        *_method('Opaque.a2', 'choice', 'var thing = this.part', 'if !thing.ready goto +342',
                 'if !(choice == this.C3) goto +27', 'var offset = (360 - this.C2)',
                 'goto +16', 'var offset = this.C2', 'thing.pending = ((thing.f + offset) % 360)'),
        *_method('Opaque.a3', '', 'if !Opaque.enabled goto +260', 'if !Key.isDown(65) goto +30',
                 'Opaque.a1(Opaque.C3)'),
    ]


def _keys():
    return {'a': SimpleNamespace(by='Opaque.a1(Opaque.C3)'),
            'b': SimpleNamespace(by='Opaque.a1(Opaque.C4)'),
            'c': SimpleNamespace(by='Opaque.a2(Opaque.C3)'),
            'd': SimpleNamespace(by='Opaque.a2(Opaque.C4)')}


def test_actual_trig_and_modular_arithmetic_read_with_opaque_names():
    result = motion_in_source_methods(_source(), _keys(), 18.)
    assert len(result.effects) == 4
    a, b, c, d = result.effects
    assert (a.kind, a.body, a.heading) == ('vector_increment', 'Opaque.part', 'Opaque.part.f')
    assert a.steps == b.steps == (-1., 1.)
    assert a.rates == b.rates == (-18., 18.)
    assert c.kind == d.kind == 'turn'
    assert c.steps == pytest.approx((-math.pi / 18, math.pi / 18))
    assert c.rates == pytest.approx((-math.pi, math.pi))
    assert c.changed == ('Opaque.part.pending',)
    assert all(not effect.unconditional for effect in result.effects)
    assert any('ready' in p.expression for p in c.predicates)
    assert any('Opaque.enabled' in p.expression for p in a.predicates)
    assert all('unresolved source control flow' in effect.unknown for effect in result.effects)
    assert ('Opaque.C2', 10.) in result.constants


def test_no_key_or_method_words_are_required():
    code = [*_method('Zzz.opaque', '', 'var o = this.z',
                     'o.k = (o.k + (Math.cos(o.q) * 2))',
                     'o.m = (o.m + (Math.sin(o.q) * 2))')]
    effects = motion_in_source_methods(code, {'?': SimpleNamespace(by='Zzz.opaque()')}, 24.).effects
    assert len(effects) == 1
    assert effects[0].key == '?'
    assert effects[0].steps == (2.,)
    assert effects[0].rates == (48.,)
    assert not effects[0].unconditional
    assert 'incremented fields have not been linked to rendered position' in effects[0].unknown


def test_modular_counter_does_not_become_heading_without_trig_connection():
    code = _method('Opaque.counter', '', 'this.n = ((this.n + 10) % 360)')
    assert not motion_in_source_methods(code, {'a': SimpleNamespace(by='Opaque.counter()')}, 18.).effects


def test_an_unrelated_heading_unit_does_not_qualify_a_turn():
    code = [*_method('Zzz.a', '', 'var o = this.z',
                     'o.u = (o.u + Math.cos(o.q))', 'o.v = (o.v + Math.sin(o.q))'),
            *_method('Zzz.b', '', 'this.z.pending = ((this.z.q + 10) % 360)')]
    effects = motion_in_source_methods(code, {'a': SimpleNamespace(by='Zzz.a()'),
                                            'b': SimpleNamespace(by='Zzz.b()')}, 18.).effects
    assert [e.kind for e in effects] == ['vector_increment']


def test_unsupported_computation_and_missing_cadence_do_not_get_a_model():
    code = _method('Opaque.method', '', 'this.x = (this.x + Arbitrary.execute())')
    assert not motion_in_source_methods(code, {'a': SimpleNamespace(by='Opaque.method()')}, 18.).effects
    assert motion_in_source_methods(_source(), _keys(), None).unknown == ('unmeasured program frame rate',)


def test_duplicate_method_assignments_are_ambiguous():
    source = _source()
    source.extend(_method('Opaque.a1', 'choice', 'this.part.u = 0'))
    result = motion_in_source_methods(source, _keys(), 18.)
    assert not any(effect.kind == 'vector_increment' for effect in result.effects)


def test_source_vector_decay_and_strict_stop_test_are_retained():
    code = [*_method('Unknown.c0', '', 'this.K = 0.95', 'this.Q = 0.5'),
            *_method('Unknown.c1', '', 'this.a.u = (this.a.u * this.K)',
                     'this.a.v = (this.a.v * this.K)',
                     'if !(Math.sqrt(((this.a.u * this.a.u) + (this.a.v * this.a.v))) < this.Q) goto +182')]
    result = motion_in_source_methods(code, {}, 18.)
    assert len(result.recoveries) == 1
    recovery = result.recoveries[0]
    assert recovery.retained_per_frame == .95
    assert recovery.stopping_speed == .5
    assert recovery.frames_below(1) == 14
    assert recovery.frames_below(2) == 28
    assert recovery.frames_below(8 / 3) == 33
    assert recovery.seconds_below(2) == pytest.approx(28 / 18)
    assert recovery.predicates


def test_decay_of_other_vector_cannot_supply_an_unrelated_stop_test():
    code = [*_method('Unknown.c0', '', 'this.K = 0.95', 'this.Q = 0.5'),
            *_method('Unknown.c1', '', 'this.a.u = (this.a.u * this.K)',
                     'this.a.v = (this.a.v * this.K)',
                     'if !(Math.sqrt(((this.b.u * this.b.u) + (this.b.v * this.b.v))) < this.Q) goto +182')]
    assert not motion_in_source_methods(code, {}, 18.).recoveries


def test_unknown_branch_decay_alternatives_remain_separate():
    code = [*_method('Unknown.c0', '', 'this.K = 0.95', 'this.J = 0.98', 'this.Q = 0.5'),
            *_method('Unknown.c1', '', 'if !unknown goto +30',
                     'this.a.u = (this.a.u * this.K)', 'this.a.v = (this.a.v * this.K)',
                     'goto +30', 'this.a.u = (this.a.u * this.J)', 'this.a.v = (this.a.v * this.J)',
                     'if !(Math.sqrt(((this.a.u * this.a.u) + (this.a.v * this.a.v))) < this.Q) goto +182')]
    result = motion_in_source_methods(code, {}, 18.)
    assert {r.retained_per_frame for r in result.recoveries} == {.95, .98}
    assert all(r.predicates for r in result.recoveries)


@pytest.mark.parametrize('decay,stop,fps', [(1., .5, 18.), (0., .5, 18.), (.95, 0., 18.),
                                         (.95, .5, 0.), (math.nan, .5, 18.)])
def test_invalid_recovery_model_is_refused(decay, stop, fps):
    with pytest.raises(ValueError):
        DampedRecovery(decay, stop, fps, ())


def test_recovery_uses_strict_threshold_and_refuses_unknown_speed():
    recovery = DampedRecovery(.95, .5, 18., ())
    assert recovery.frames_below(.49) == 0
    assert recovery.frames_below(.5) == 1
    assert recovery.frames_below(math.nan) is None
    assert recovery.frames_below(-1) is None


def test_saved_reading_retains_source_hash_and_unknown_predicates():
    source = 'swf:' + 'a' * 64
    reading = motion_in_source_methods(_source(), _keys(), 18., source=source)
    restored = unkept(json.loads(json.dumps(kept(reading))))
    assert restored == reading
    assert restored.source == source
    assert all(not effect.unconditional for effect in restored.effects)


def test_saved_reading_refuses_invalid_numeric_or_custody_fields():
    data = kept(motion_in_source_methods(_source(), _keys(), 18.))
    data['effects'][0]['steps'] = [math.nan]
    assert unkept(data) is None


def test_saved_vector_increment_cannot_become_physics_by_removing_its_warnings():
    data = kept(motion_in_source_methods(_source(), _keys(), 18.))
    data['effects'][0]['unknown'] = []
    data['effects'][0]['predicates'] = []
    restored = unkept(data)
    assert restored is not None and not restored.effects[0].unconditional


def test_candidate_constant_from_an_uncalled_or_guarded_method_is_not_proven():
    code = [*_method('Unknown.init', '', 'if !unknown goto +30', 'this.K = 2'),
            *_method('Unknown.effect', '', 'this.a.u = (this.a.u + (Math.cos(this.a.q) * this.K))',
                     'this.a.v = (this.a.v + (Math.sin(this.a.q) * this.K))')]
    result = motion_in_source_methods(code, {'x': SimpleNamespace(by='Unknown.effect()')}, 18.)
    assert result.effects
    assert not result.effects[0].unconditional
    assert 'source constant initialization is unproved' in result.effects[0].unknown


def test_flat_return_does_not_prove_reachability_of_later_arithmetic():
    code = _method('Unknown.effect', '', 'return undefined',
                   'this.a.u = (this.a.u + Math.cos(this.a.q))',
                   'this.a.v = (this.a.v + Math.sin(this.a.q))')
    result = motion_in_source_methods(code, {'x': SimpleNamespace(by='Unknown.effect()')}, 18.)
    assert result.effects
    assert not result.effects[0].unconditional


def test_source_bounds_apply_before_unbounded_marker_scans():
    code = ['  // anonymous function ()'] * 4001
    assert not motion_in_source_methods(code, {}, 18.).effects
    result = motion_in_source_methods(['x' * 4097], {}, 18.)
    assert result.unknown == ('source text exceeds bounded method reader',)
    data = kept(motion_in_source_methods(_source(), _keys(), 18.))
    data['source'] = 'made-up-owner'
    assert unkept(data) is None
