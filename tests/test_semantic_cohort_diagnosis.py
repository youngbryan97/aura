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
    monkeypatch.setattr(audit, '_observe', lambda *a: {'semantic_status': 'equivalent'})
    result = audit.audit_semantic_cohort(model, examples, directory=tmp_path)
    assert result['stages'] == {'semantic_success_downstream_unmeasured': 2}
    assert result['coverage_complete'] and not result['serving_authority']
    monkeypatch.setattr(audit, '_observe', lambda *a: pytest.fail('cached row decoded again'))
    assert audit.audit_semantic_cohort(model, examples, directory=tmp_path) == result
    with pytest.raises(ValueError, match='identity differs'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, max_charts=17)


def test_interruption_keeps_completed_row_without_claiming_complete(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a: {'semantic_status': 'equivalent'})
    def interrupt(row):
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, progress=interrupt)
    assert len(list(tmp_path.glob('*.json'))) == 1
    result = audit.audit_semantic_cohort(model, examples, directory=tmp_path)
    assert result['observed_count'] == result['expected_count'] == 2


def test_grounding_failure_is_not_mislabeled_selection(tmp_path, monkeypatch):
    model, examples = setup(monkeypatch)
    monkeypatch.setattr(audit, '_observe', lambda *a: {'semantic_status': 'unmeasured', 'source_grounding_aligned': False})
    monkeypatch.setattr(audit, 'decode_semantic_candidates', lambda *a, **k: pytest.fail('searched past grounding failure'))
    assert audit.audit_semantic_cohort(model, examples, directory=tmp_path)['stages'] == {'grounding': 2}


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
        assert row['failure_stage'] != 'success'


@pytest.mark.parametrize('options', [dict(max_charts=0), dict(max_graphs_per_chart=True),
    dict(solve_time_limit_s=float('inf'))])
def test_invalid_search_contract_is_rejected_even_for_correct_rows(tmp_path, monkeypatch, options):
    model, examples = setup(monkeypatch)
    with pytest.raises(ValueError, match='positive and finite'):
        audit.audit_semantic_cohort(model, examples, directory=tmp_path, **options)
