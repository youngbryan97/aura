"""Complete-program energies match actual public chart edge computations."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from core.learning.semantic_conditioned_relations import conditional_choice_scores
from core.learning.semantic_grounded_program_objective import (
    GroundedProgramSupervision,
    ProgramChartChoices,
    ProgramGraphChoice,
    mine_grounded_program_supervision,
    program_chart_edge_scores,
    program_objective_contract,
)
from core.learning.semantic_native_operation_field import NativeOperationField
from tools.semantic_grounded_score_execution import conditional_role_update


@pytest.mark.parametrize("offset", [0., 100., -100.])
def test_differentiable_choice_update_matches_runtime_and_cancels_offsets(offset):
    baseline, learned = (2., -1., .2), (.7, -.8, 1.2)
    actual = conditional_choice_scores(mx.array(baseline), mx.array(learned) + offset, weight=.6)
    expected, _receipt = conditional_role_update(baseline, learned, weight=.6)
    np.testing.assert_allclose(actual.tolist(), expected, atol=5e-6)
    gradient = mx.grad(lambda value: mx.sum(conditional_choice_scores(mx.array(baseline), value)))(mx.array(learned))
    assert abs(mx.sum(gradient).item()) < 1e-5 and mx.any(gradient != 0).item()


@pytest.mark.parametrize("rounds", [0, 2])
@pytest.mark.parametrize("projection", [False, True])
def test_program_loss_scores_every_public_edge_identically_to_batched_decoder(rounds, projection):
    from tests.test_semantic_grounded_chart_bridge import fixture
    from tools.semantic_grounded_batched_chart import BatchedGroundedBindingChartSolver
    bridge, chart, public = fixture()
    pointer = bridge.engine.pointer
    pointer.rounds = rounds
    if projection:
        bridge.engine.nuisance_projection = SimpleNamespace(basis=np.array([[1.], [0.], [0.], [0.]]))
    proposal = ProgramChartChoices(public["operation_nodes"], chart, ())
    actual = program_chart_edge_scores(pointer, bridge.depth_states, proposal,
        source_id=public["source_id"], input_spans=public["input_spans"], inputs=public["inputs"],
        nuisance_projection=bridge.engine.nuisance_projection, batch_size=2)
    runtime = BatchedGroundedBindingChartSolver(bridge.engine, bridge.source_id, bridge.depth_states,
        score_policy="conditional_likelihood")
    runtime(chart, **public)
    expected = [row["learned_relation_score"] for row in runtime.last_resolution["edge_evidence"]]
    np.testing.assert_allclose(actual.tolist(), expected, atol=1e-5, rtol=1e-5)
    gradient = mx.grad(lambda states: mx.sum(program_chart_edge_scores(pointer, states, proposal,
        source_id=public["source_id"], input_spans=public["input_spans"], inputs=public["inputs"],
        nuisance_projection=bridge.engine.nuisance_projection)))(bridge.depth_states)
    assert mx.all(mx.isfinite(gradient)).item() and mx.any(gradient != 0).item()


def source_pool():
    from tests.test_semantic_grounded_chart_bridge import fixture
    bridge, chart, public = fixture()
    chart = chart.with_conditional_choices()
    graphs = []
    for args, status in ((((1, 0),), "equivalent"), (((0, 1),), "different")):
        # This fixture has exactly one mention per register in each role.
        restricted = chart.restrict_arguments(args)
        result = restricted.solve()
        indices = tuple(tuple(next(index for index, option in enumerate(pool) if option[1] == register)
            for pool, register in zip(node, target, strict=True))
            for node, target in zip(chart.options, args, strict=True))
        graphs.append(ProgramGraphChoice(indices, result[0], {"status": status, "method": "fixture_role_swap"}))
    proposal = ProgramChartChoices(public["operation_nodes"], chart, tuple(graphs))
    source = GroundedProgramSupervision(public["source_id"], public["input_spans"], public["inputs"],
        (proposal,), {"fixture": True})
    field = NativeOperationField(4, depths=1, labels=("sub", "add"), relation_width=8, max_span_tokens=1)
    source.validate(field, len(bridge.depth_states))
    return bridge, field, source


def test_whole_program_loss_has_finite_gradients_to_roles_operations_and_source_states():
    bridge, field, source = source_pool()
    bridge.engine.pointer.lora_b *= .01
    other = replace(source.charts[0], nodes=tuple(SimpleNamespace(operation="add", span=node.span)
        for node in source.charts[0].nodes), graphs=(replace(source.charts[0].graphs[1],
            comparison={"status": "different", "method": "fixture_primitive_contrast"}),))
    source = replace(source, charts=(*source.charts, other))
    model = nn.Module()
    model.pointer, model.operation_field = bridge.engine.pointer, field
    score, grads = nn.value_and_grad(model,
        lambda owner: source.source_loss(owner.pointer, owner.operation_field, bridge.depth_states))(model)
    assert mx.isfinite(score).item()
    from mlx.utils import tree_flatten
    assert all(mx.all(mx.isfinite(value)).item() for _name, value in tree_flatten(grads))
    assert any(mx.any(value != 0).item() for name, value in tree_flatten(grads) if name.startswith("pointer."))
    assert any(mx.any(value != 0).item() for name, value in tree_flatten(grads) if name.startswith("operation_field."))
    gradient = mx.grad(lambda states: source.source_loss(model.pointer, model.operation_field, states))(bridge.depth_states)
    assert mx.all(mx.isfinite(gradient)).item() and mx.any(gradient != 0).item()


def test_program_custody_changes_with_selected_factors_and_rejects_fabricated_baseline():
    bridge, field, source = source_pool()
    contract = program_objective_contract({source.source_id: source}, 1.)
    assert not contract["all_programs_covered"] and not contract["whole_program_calibration_proven"]
    changed = replace(source.charts[0].graphs[0], baseline_score=999.)
    drift = replace(source, charts=(replace(source.charts[0], graphs=(changed, source.charts[0].graphs[1])),))
    assert program_objective_contract({source.source_id: drift}, 1.) != contract
    with pytest.raises(ValueError, match="selected public factors"):
        drift.validate(field, len(bridge.depth_states))
    with pytest.raises(ValueError, match="unproved labels"):
        replace(source, charts=(replace(source.charts[0], graphs=(replace(changed,
            comparison={"status": "unknown"}),)),)).validate(field, len(bridge.depth_states))


def test_a_correct_only_pool_has_zero_contrast_loss_without_an_artificial_blocker():
    bridge, field, source = source_pool()
    source = replace(source, charts=(replace(source.charts[0], graphs=(source.charts[0].graphs[0],)),))
    source.validate(field, len(bridge.depth_states))
    assert source.source_loss(bridge.engine.pointer, field, bridge.depth_states).item() == 0.


def test_public_mining_precedes_source_labels_and_refuses_held_examples():
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding
    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    item = next(item for item in examples if item.split == "train")
    pool = mine_grounded_program_supervision(parent, item, max_charts=2, max_graphs=2, max_seconds=10.)
    assert pool.mining["proven_positive_graphs"] and pool.mining["witnessed_negative_graphs"]
    assert not pool.mining["target_available_to_runtime"] and not pool.mining["complete_grammar_partition"]
    field = NativeOperationField(4, depths=1, labels=parent.operation_head.labels,
        max_span_tokens=parent.max_span_tokens, max_steps=parent.max_steps)
    pool.validate(field, len(item.ir.source_token_ids))
    with pytest.raises(ValueError, match="source-training"):
        mine_grounded_program_supervision(parent, replace(item, split="test"))


def test_program_pool_roundtrip_is_exact_and_reuses_no_search(monkeypatch, tmp_path):
    import json

    import tools.semantic_grounded_program_pool as module
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    item = next(item for item in examples if item.split == "train")
    options = dict(max_charts=2, max_graphs=2, max_seconds=10.)
    pool, reused = module.source_program_pool(parent, item, tmp_path, **options)
    assert not reused
    monkeypatch.setattr(module, "mine_grounded_program_supervision", lambda *_args, **_kwargs:
        pytest.fail("repeated completed program search"))
    loaded, reused = module.source_program_pool(parent, item, tmp_path, **options)
    assert reused and json.dumps(loaded.receipt(), sort_keys=True) == json.dumps(pool.receipt(), sort_keys=True)
    with pytest.raises(ValueError, match="custody"):
        module.source_program_pool(parent, item, tmp_path, **{**options, "max_charts": 3})
    path = tmp_path / f"{item.ir.source_text_sha256}.json"
    receipt = json.loads(path.read_text())
    receipt["programs"]["mining"]["public_charts"] += 1
    path.chmod(0o600)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="custody"):
        module.source_program_pool(parent, item, tmp_path, **options)


def test_population_preflight_keeps_all_failures_and_does_not_claim_semantic_success(monkeypatch, tmp_path):
    import json

    import tools.semantic_grounded_program_pool as module

    seen = []
    def fail(_parent, item, _directory, **_bounds):
        seen.append(item.ir.source_text_sha256)
        raise ValueError("unreachable source role")
    monkeypatch.setattr(module, "source_program_pool", fail)
    items = tuple(SimpleNamespace(ir=SimpleNamespace(source_text_sha256=str(index), instructions=(1, 2)),
        construction_id="fit-only-family") for index in range(3))
    with pytest.raises(ValueError, match="3/3"):
        module.prepare_program_population(None, items, tmp_path)
    assert seen == ["0", "1", "2"]
    report = json.loads(next(tmp_path.glob("population-*.json")).read_text())
    assert report["failed"] == report["population"] == 3 and not report["fit_ready"]
    assert not report["model_weights_loaded"] and report["semantic_success"] is None
