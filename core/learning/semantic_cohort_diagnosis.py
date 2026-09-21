"""Map every declared development observation before selecting a repair."""

import json
import math
from collections import Counter
from pathlib import Path

from core.learning.semantic_candidate_bank import decode_semantic_candidates
from core.learning.semantic_failure_diagnosis import diagnose_semantic_candidate_bank
from core.learning.semantic_graph_trial import _observe
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_validation_checkpoint import validation_identity, validation_implementation_identity
from core.runtime.atomic_writer import atomic_write_bytes_if_absent


def audit_semantic_cohort(model, examples, *, directory, max_charts=16,
                         max_graphs_per_chart=16, solve_time_limit_s=3., progress=None,
                         diagnose_failures=True):
    """Audit a fixed cohort; annotated targets never enter candidate generation."""
    examples = tuple(examples)
    if type(diagnose_failures) is not bool:
        raise ValueError('diagnose_failures must be boolean')
    if (any(type(value) is not int or value < 1 for value in (max_charts, max_graphs_per_chart))
            or type(solve_time_limit_s) not in (int, float)
            or not math.isfinite(solve_time_limit_s) or solve_time_limit_s <= 0):
        raise ValueError('cohort audit search allowances must be positive and finite')
    ids = [item.ir.source_text_sha256 for item in examples]
    if not ids or len(ids) != len(set(ids)) or any(item.split not in {'train', 'validation'} for item in examples):
        raise ValueError('cohort audit requires unique development observations')
    implementation = validation_implementation_identity()
    options = dict(max_charts=max_charts, max_graphs_per_chart=max_graphs_per_chart,
                   solve_time_limit_s=solve_time_limit_s)
    policy = 'completed_prefix_then_expand_v1' if diagnose_failures else 'ordinary_observation_only_v1'
    identity = _sha({'observations': validation_identity({'candidate': model}, examples,
        scoring='source_anchors_v2', implementation=implementation), 'options': options,
        'diagnostic_policy': policy})
    directory = Path(directory)
    rows = []
    for item in examples:
        path = directory / f'{item.ir.source_text_sha256}.json'
        if path.exists():
            document = json.loads(path.read_text())
            body = {k: v for k, v in document.items() if k != 'receipt_sha256'}
            if (document.get('receipt_sha256') != _sha(body) or body.get('identity') != identity
                    or body.get('source_text_sha256') != item.ir.source_text_sha256):
                raise ValueError('cohort audit checkpoint identity differs')
        else:
            if progress:
                progress({'stage': 'observation_started', 'source': item.ir.source_text_sha256,
                          'completed': len(rows), 'total': len(examples)})
            observation = _observe(model, item, solve_time_limit_s=solve_time_limit_s)
            diagnosis = None
            attempts = []
            if observation['semantic_status'] == 'equivalent':
                stage = 'semantic_success_downstream_unmeasured'
            elif observation['source_grounding_aligned'] is False:
                stage = 'grounding'
            elif observation['semantic_status'] == 'decode_refused':
                stage = 'decode_unavailable'
            elif not diagnose_failures:
                stage = ('semantic_failure_unattributed' if observation['semantic_status'] == 'different'
                         else 'semantic_comparison_unresolved')
            else:
                # One equivalent alternative proves reachability. Complete each
                # target-blind bank before deciding whether diagnostics need more.
                limits = tuple(dict.fromkeys(((1, 1), (max_charts, max_graphs_per_chart))))
                for charts, graphs in limits:
                    bank = decode_semantic_candidates(model,
                        source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
                        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
                        model_basis_sha256=item.ir.model_basis_receipt_sha256,
                        max_charts=charts, max_graphs_per_chart=graphs,
                        solve_time_limit_s=solve_time_limit_s,
                        progress=(lambda event: progress({**event, 'source': item.ir.source_text_sha256}))
                        if progress else None)
                    diagnosis = diagnose_semantic_candidate_bank(bank, item)
                    attempts.append({'bank': bank.receipt, 'diagnosis': diagnosis})
                    if diagnosis['correct_reachable'] is True or bank.receipt['search_complete']:
                        break
                stage = diagnosis['failure_stage']
            body = {'schema': 'aura.semantic_cohort_diagnosis_row.v2', 'identity': identity,
                    'source_text_sha256': item.ir.source_text_sha256, 'split': item.split,
                    'construction_id': item.construction_id, 'topology_id': item.topology_id,
                    'observation': observation, 'diagnosis': diagnosis, 'failure_stage': stage,
                    'diagnostic_attempts': attempts,
                    'serving_authority': False}
            document = {**body, 'receipt_sha256': _sha(body)}
            payload = (json.dumps(document, sort_keys=True, allow_nan=False) + '\n').encode()
            if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
                raise FileExistsError(path)
        rows.append(document)
        if progress:
            progress({'completed': len(rows), 'total': len(examples), 'source': ids[len(rows)-1],
                      'failure_stage': document['failure_stage']})
    if implementation != validation_implementation_identity():
        raise ValueError('cohort audit implementation changed')
    body = {'schema': 'aura.semantic_cohort_diagnosis.v2', 'identity': identity,
            'implementation': implementation, 'candidate': model.receipt_sha256,
            'diagnostic_policy': policy, 'search_allowances': options,
            'diagnostic_budget_uses_targets_after_bank_completion': diagnose_failures,
            'expected_count': len(examples), 'observed_count': len(rows), 'coverage_complete': True,
            'stages': dict(Counter(row['failure_stage'] for row in rows)),
            'rows': rows, 'test_examples_used': 0, 'serving_authority': False,
            'fresh_transfer_claim': False, 'learning_performed': False}
    return {**body, 'receipt_sha256': _sha(body)}
