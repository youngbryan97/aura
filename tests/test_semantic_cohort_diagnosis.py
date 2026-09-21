"""A cohort audit cannot turn missing or resumed evidence into correctness."""

from types import SimpleNamespace

import pytest

from core.learning import semantic_cohort_diagnosis as audit


def setup(monkeypatch):
    monkeypatch.setattr(audit, 'validation_implementation_identity', lambda: 'implementation')
    monkeypatch.setattr(audit, 'validation_identity', lambda *a, **k: 'observations')
    model = SimpleNamespace(receipt_sha256='candidate')
    examples = tuple(SimpleNamespace(ir=SimpleNamespace(source_text_sha256=f'source{i}'),
        split='train', construction_id='construction', topology_id='topology') for i in range(2))
    return model, examples


def test_complete_semantics_does_not_claim_execution_and_resume_does_not_decode(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: {'semantic_status': 'equivalent'})
    result = audit.audit_semantic_cohort(model, examples, directory=tmp_path)
    assert result['stages'] == {'semantic_success_downstream_unmeasured': 2}
    assert result['coverage_complete'] and not result['serving_authority']
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: pytest.fail('cached row decoded again'))
    assert audit.audit_semantic_cohort(model, examples, directory=tmp_path) == result
    with pytest.raises(ValueError, match='identity differs'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, max_charts=17)


def test_interruption_keeps_completed_row_without_claiming_complete(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: {'semantic_status': 'equivalent'})
    def interrupt(row):
        if 'failure_stage' in row:
            raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, progress=interrupt)
    assert len(list(tmp_path.glob('*.json'))) == 1
    result = audit.audit_semantic_cohort(model, examples, directory=tmp_path)
    assert result['observed_count'] == result['expected_count'] == 2


def test_grounding_failure_is_not_mislabeled_selection(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: {'semantic_status': 'unmeasured', 'source_grounding_aligned': False})
    monkeypatch.setattr(audit, 'decode_semantic_candidates', lambda *a, **k: pytest.fail('searched past grounding failure'))
    assert audit.audit_semantic_cohort(model, examples, directory=tmp_path)['stages'] == {'grounding': 2}


def test_decode_refusal_keeps_the_budget_and_does_not_repeat_search(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    def observe(*args, solve_time_limit_s):
        assert solve_time_limit_s == 0.5
        return {'semantic_status': 'decode_refused', 'source_grounding_aligned': None,
                'refusal': 'decode_search_budget_exhausted'}
    monkeypatch.setattr(audit, '_observe', observe)
    monkeypatch.setattr(audit, 'decode_semantic_candidates', lambda *a, **k: pytest.fail('repeated refused decode'))
    events = []
    result = audit.audit_semantic_cohort(model, examples, directory=tmp_path,
        solve_time_limit_s=0.5, progress=events.append)
    assert result['stages'] == {'decode_unavailable': 2}
    assert events[0]['stage'] == 'observation_started'
    assert result['coverage_complete'] and not result['serving_authority']


def test_test_split_and_duplicate_sources_are_refused(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    with pytest.raises(ValueError, match='unique development'):
        audit.audit_semantic_cohort(model, (examples[0], examples[0]), directory=tmp_path)
    examples[0].split = 'test'
    with pytest.raises(ValueError, match='unique development'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path)


def test_real_candidate_path_keeps_downstream_evidence_unmeasured(tmp_path):
    from tests.test_semantic_relation_graph_learning import model_examples
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    result = audit.audit_semantic_cohort(model, examples[:2], directory=tmp_path,
        max_charts=2, max_graphs_per_chart=2, solve_time_limit_s=1.)
    assert result['observed_count'] == 2
    assert sum(result['stages'].values()) == 2
    assert result['learning_performed'] is False
    for row in result['rows']:
        if row['diagnosis'] is not None:
            assert row['diagnosis']['execution_correct'] is None
            assert row['diagnosis']['emission_correct'] is None
            assert row['diagnostic_attempts'][-1]['bank']['receipt_sha256'] == row['diagnosis']['bank_receipt_sha256']
        assert row['failure_stage'] != 'success'


@pytest.mark.parametrize('first_reachable,complete,expected', [
    (True, False, [(1, 1)]),
    (False, True, [(1, 1)]),
    (None, False, [(1, 1), (4, 3)]),
])
def test_diagnosis_expands_only_without_a_witness_or_exhaustion(tmp_path, monkeypatch,
    first_reachable, complete, expected):
    model, examples = setup(monkeypatch)
    item = examples[0]
    item.ir.source_token_ids = (1, 2)
    item.ir.model_basis_receipt_sha256 = 'basis'
    item.hidden_states = None
    item.public_inputs = (2, 3)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k:
        {'semantic_status': 'different', 'source_grounding_aligned': True})
    calls = []
    events = []
    def generate(model, **kwargs):
        assert set(kwargs) == {'source_token_ids', 'hidden_states', 'public_inputs',
            'source_text_sha256', 'model_basis_sha256', 'max_charts',
            'max_graphs_per_chart', 'solve_time_limit_s', 'progress'}
        calls.append((kwargs['max_charts'], kwargs['max_graphs_per_chart']))
        events.append('generated')
        return SimpleNamespace(receipt={'search_complete': complete,
            'receipt_sha256': f'bank{len(calls)}'})
    def diagnose(bank, item):
        assert events[-1] == 'generated'
        events.append('diagnosed')
        return {'correct_reachable': first_reachable if len(calls) == 1 else True,
            'failure_stage': 'selection', 'bank_receipt_sha256': bank.receipt['receipt_sha256']}
    monkeypatch.setattr(audit, 'decode_semantic_candidates', generate)
    monkeypatch.setattr(audit, 'diagnose_semantic_candidate_bank', diagnose)
    result = audit.audit_semantic_cohort(model, (item,), directory=tmp_path,
        max_charts=4, max_graphs_per_chart=3)
    assert calls == expected
    assert len(result['rows'][0]['diagnostic_attempts']) == len(expected)
    assert result['diagnostic_budget_uses_targets_after_bank_completion'] is True
    assert result['serving_authority'] is False


@pytest.mark.parametrize('options', [dict(max_charts=0), dict(max_graphs_per_chart=True),
    dict(solve_time_limit_s=float('inf'))])
def test_invalid_search_contract_is_rejected_even_for_correct_rows(tmp_path, monkeypatch, options):
    model, examples = setup(monkeypatch)
    with pytest.raises(ValueError, match='positive and finite'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, **options)


@pytest.mark.parametrize('status,stage', [
    ('different', 'semantic_failure_unattributed'),
    ('unmeasured', 'semantic_comparison_unresolved'),
    ('equivalent', 'semantic_success_downstream_unmeasured'),
])
def test_observation_only_finishes_cohort_without_claiming_attribution(tmp_path, monkeypatch, status, stage):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k:
        {'semantic_status': status, 'source_grounding_aligned': True})
    monkeypatch.setattr(audit, 'decode_semantic_candidates', lambda *a, **k:
        pytest.fail('ordinary measurement searched diagnostic alternatives'))
    result = audit.audit_semantic_cohort(model, examples, directory=tmp_path, diagnose_failures=False)
    assert result['stages'] == {stage: 2}
    assert result['coverage_complete'] and not result['serving_authority']
    assert result['diagnostic_policy'] == 'ordinary_observation_only_v1'
    assert result['diagnostic_budget_uses_targets_after_bank_completion'] is False
    assert all(row['diagnosis'] is None and row['diagnostic_attempts'] == [] for row in result['rows'])
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: pytest.fail('cached row decoded again'))
    assert audit.audit_semantic_cohort(model, examples, directory=tmp_path, diagnose_failures=False) == result
    with pytest.raises(ValueError, match='identity differs'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path)


def test_observation_policy_requires_boolean(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    with pytest.raises(ValueError, match='boolean'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, diagnose_failures='false')


@pytest.mark.parametrize('field,value', [('split','validation'), ('construction_id','other'),
                                       ('topology_id','other')])
def test_cohort_membership_labels_are_part_of_cache_identity(tmp_path, monkeypatch, field, value):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: {'semantic_status':'equivalent'})
    audit.audit_semantic_cohort(model, examples, directory=tmp_path)
    setattr(examples[0], field, value)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: pytest.fail('changed membership reused'))
    with pytest.raises(ValueError, match='identity differs'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path)


def test_resealed_row_cannot_change_its_bound_split(tmp_path, monkeypatch):
    import json

    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a, **k: {'semantic_status':'equivalent'})
    audit.audit_semantic_cohort(model, examples, directory=tmp_path)
    path = tmp_path/'source0.json'
    body = json.loads(path.read_text())
    body.pop('receipt_sha256')
    body['split'] = 'validation'
    path.chmod(0o600)
    path.write_text(json.dumps({**body,'receipt_sha256':audit._sha(body)}))
    with pytest.raises(ValueError, match='identity differs'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path)
