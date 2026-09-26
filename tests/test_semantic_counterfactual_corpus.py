"""Language contrasts keep register identity, source order and meaning separate."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace
import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_counterfactual_corpus import (
    augment_source_programs, cross_construction_relation_controls,
    equivalent_recompositions, render_bound_program,
)
from core.learning.semantic_program_corpus import build_semantic_program_corpus
from core.learning.semantic_program_floor import compile_source_independent_program_to_floor, execute_semantic_floor_program


def source():
    return next(item for item in build_semantic_program_corpus(seed=24) if item.split == 'train')


def test_reordering_and_renaming_preserve_bound_program_and_every_reference():
    item = source()
    names = ('amber', 'juniper', 'cobalt', 'tulip', 'willow')
    variant = render_bound_program(item, names=names, clause_order=(1, 0))
    assert variant.program == item.program
    assert variant.instructions[1].operation_span.start < variant.instructions[0].operation_span.start
    for name, span in zip(names, variant.register_definition_spans, strict=True):
        assert variant.source_text[span.start:span.end] == name
    for annotated in variant.instructions:
        for register, span in zip(annotated.instruction.args, annotated.argument_spans, strict=True):
            assert variant.source_text[span.start:span.end] == names[register]
    for value, span in zip(item.inputs, variant.input_spans, strict=True):
        assert variant.source_text[span.start:span.end] == str(value)


def test_subtraction_word_order_does_not_swap_the_computational_operands():
    item = source()
    program = Program(3, (Instruction('sub', (0, 1)), Instruction('sub', (2, 3))))
    variant = render_bound_program(item, names=('first', 'second', 'third', 'interim', 'final'),
                                   clause_order=(0, 1), program=program)
    assert 'subtract second from first' in variant.source_text
    assert 'subtract interim from third' in variant.source_text
    assert variant.program == program


def test_generated_contrasts_are_source_only_and_independently_replay():
    examples = build_semantic_program_corpus(seed=31)
    training = tuple(item for item in examples if item.split == 'train')[:12]
    excluded = tuple(item for item in examples if item.split != 'train')[:4]
    rows, receipt = augment_source_programs((*training, *excluded), seed=12)
    assert len(rows) >= 3 * len(training)
    assert receipt['ignored_nontraining_examples'] == len(excluded)
    assert receipt['test_examples_used'] == 0
    assert {row.split for row in rows} == {'train'}
    assert {row.contrast_id for row in rows} == {row.example_id for row in training}
    assert len({row.example_id for row in rows}) == len(rows)
    sources, generated = {row.example_id: row for row in training}, {row.example_id: row for row in rows}
    for record in receipt['records']:
        original, variant = sources[record['source']], generated[record['example']]
        if record['status'] == 'equivalent':
            if record.get('transformation') == 'associative_recomposition':
                assert variant.program != original.program
                assert record['comparison']['method'] == 'floor_integer_polynomial_v1'
            else:
                assert variant.program == original.program
            continue
        witness = record['comparison']['witness']
        outputs = []
        for program in (original.program, variant.program):
            compiled = compile_source_independent_program_to_floor(program, tuple(witness['inputs']),
                                                                    provenance_receipt_sha256='c' * 64)
            result = execute_semantic_floor_program(compiled)
            assert result.result == program.run(witness['inputs'])
            outputs.append(result.result)
        assert outputs == witness['outputs'] and outputs[0] != outputs[1]
    replay, repeated = augment_source_programs((*training, *excluded), seed=12)
    assert repeated == receipt and replay == rows


def test_cross_construction_controls_share_relation_and_witness_role_change():
    from core.learning.semantic_program_floor import semantic_program_structural_key

    examples = tuple(item for item in build_semantic_program_corpus(seed=24)
                     if item.split == 'train')
    controls = cross_construction_relation_controls(examples)
    by_id = {item.example_id: item for item in examples}
    assert controls['cross_construction_pairs'] > 0
    assert controls == cross_construction_relation_controls(examples)
    for pair in controls['pairs']:
        left, right = by_id[pair['left']], by_id[pair['right']]
        assert left.construction_id != right.construction_id
        assert left.contrast_id != right.contrast_id
        assert semantic_program_structural_key(left.program) == semantic_program_structural_key(right.program)
        assert pair['negative_witness']['outputs'][0] != pair['negative_witness']['outputs'][1]
    with pytest.raises(ValueError, match='source training only'):
        cross_construction_relation_controls((*examples[:1], next(item for item in
            build_semantic_program_corpus(seed=24) if item.split == 'validation')))


def test_source_variants_remain_in_their_origin_construction_fold():
    from core.learning.semantic_construction_folds import construction_folds

    by_construction = {}
    for item in build_semantic_program_corpus(seed=31):
        if item.split == 'train':
            by_construction.setdefault(item.construction_id, item)
    sources = tuple(by_construction.values())
    variants, _receipt = augment_source_programs(sources, seed=12, lineage_version=2)
    projected = tuple(SimpleNamespace(
        ir=SimpleNamespace(source_text_sha256=hashlib.sha256(
            item.source_text.encode('utf-8')).hexdigest()),
        split=item.split, construction_id=item.construction_id,
        contrast_id=item.contrast_id) for item in (*sources, *variants))
    folds = construction_folds(projected, count=3)
    for variant in variants:
        child = hashlib.sha256(variant.source_text.encode('utf-8')).hexdigest()
        assert folds['assignments'][child] == folds['assignments'][variant.contrast_id]
    assert len(set(folds['assignments'].values())) == 3


def test_evaluation_constructions_cannot_become_training_contrasts():
    item = source()
    with pytest.raises(ValueError, match='reserved'):
        augment_source_programs((item,), forbidden_constructions=(item.construction_id,))
    rows, receipt = augment_source_programs((replace(item, split='test'),))
    assert rows == () and receipt['ignored_nontraining_examples'] == 1


@pytest.mark.parametrize('names,order', [
    (('a', 'a', 'c', 'd', 'e'), (0, 1)),
    (('a', 'b', 'c', 'd', 'e'), (0, 0)),
    (('a', 'b', 'c', 'd', 'e'), (False, True)),
    (('a', 'b', 'c', 'd', 'do work'), (0, 1)),
])
def test_invalid_alias_or_order_is_not_rendered(names, order):
    with pytest.raises(ValueError):
        render_bound_program(source(), names=names, clause_order=order)


def test_sequence_scalar_type_swap_is_rejected_before_rendering():
    item = replace(source(), inputs=((1, 2, 1), 1, 3))
    correct = Program(3, (Instruction('count_of', (0, 1)), Instruction('add', (3, 2))))
    names = ('data', 'target', 'extra', 'tally', 'answer')
    variant = render_bound_program(item, names=names, clause_order=(1, 0), program=correct)
    assert 'count how often target occurs in data' in variant.source_text
    broken = replace(correct, instructions=(Instruction('count_of', (1, 0)), correct.instructions[1]))
    with pytest.raises(ValueError, match='type contract'):
        render_bound_program(item, names=names, clause_order=(1, 0), program=broken)


def test_typed_mutation_still_has_to_execute_on_the_printed_inputs():
    item = replace(source(), inputs=(4, 4, 3))
    program = Program(3, (Instruction('sub', (0, 1)), Instruction('idiv', (2, 3))))
    with pytest.raises(ValueError, match='printed inputs'):
        render_bound_program(item, names=('a', 'b', 'c', 'd', 'e'), clause_order=(0, 1), program=program)


@pytest.mark.parametrize('operation', ['add', 'mul'])
def test_associative_recomposition_changes_graph_not_computation(operation):
    program = Program(3, (Instruction(operation, (0, 1)), Instruction(operation, (3, 2))))
    variants = tuple(equivalent_recompositions(program))
    assert len(variants) == 1 and variants[0] != program
    for a in range(-3, 4):
        for b in range(-3, 4):
            for c in range(-3, 4):
                assert variants[0].run((a, b, c)) == program.run((a, b, c))


def test_recomposition_does_not_assume_subtraction_associativity_or_duplicate_shared_work():
    subtraction = Program(3, (Instruction('sub', (0, 1)), Instruction('sub', (3, 2))))
    shared = Program(3, (Instruction('add', (0, 1)), Instruction('add', (3, 2)),
                         Instruction('sub', (3, 4))))
    assert tuple(equivalent_recompositions(subtraction)) == ()
    assert tuple(equivalent_recompositions(shared)) == ()


def test_every_generated_natural_example_has_an_independent_execution():
    from core.learning.semantic_counterfactual_corpus import build_semantic_counterfactual_source_corpus

    for seed in (41, 43, 47):
        for item in build_semantic_counterfactual_source_corpus(seed=seed):
            compiled = compile_source_independent_program_to_floor(item.program, item.inputs,
                                                                    provenance_receipt_sha256='c' * 64)
            assert execute_semantic_floor_program(compiled).result == item.program.run(item.inputs)


def test_existing_feature_materializer_can_acquire_counterfactual_training():
    from core.learning.semantic_program_feature_materialization import (
        SemanticFeatureConfig, FAMILY_FEATURE_CONFIG_SCHEMA, build_semantic_program_corpus_for_config,
        COUNTERFACTUAL_SOURCE_CORPUS_KIND,
    )
    from core.learning.semantic_program_corpus_natural import build_semantic_program_natural_source_corpus

    config = SemanticFeatureConfig(seed=41, corpus_kind=COUNTERFACTUAL_SOURCE_CORPUS_KIND,
                                   schema=FAMILY_FEATURE_CONFIG_SCHEMA, max_examples=4096)
    generated = build_semantic_program_corpus_for_config(config)
    original = build_semantic_program_natural_source_corpus(seed=41)
    training_ids = {row.example_id for row in original if row.split == 'train'}
    assert generated and len(generated) <= config.max_examples
    assert {row.split for row in generated} == {'train'}
    assert {row.contrast_id for row in generated} == training_ids
    assert not {row.source_text for row in generated} & {row.source_text for row in original}


def test_v2_feature_materializer_uses_source_hash_lineage_without_changing_v1():
    from core.learning.semantic_program_feature_materialization import (
        COUNTERFACTUAL_SOURCE_CORPUS_KIND, COUNTERFACTUAL_SOURCE_CORPUS_V2_KIND,
        FAMILY_FEATURE_CONFIG_SCHEMA, SemanticFeatureConfig,
        build_semantic_program_corpus_for_config,
    )
    from core.learning.semantic_program_corpus_natural import build_semantic_program_natural_source_corpus

    def generated(kind):
        return build_semantic_program_corpus_for_config(SemanticFeatureConfig(
            seed=41, corpus_kind=kind, schema=FAMILY_FEATURE_CONFIG_SCHEMA,
            max_examples=4096))

    legacy, corrected = generated(COUNTERFACTUAL_SOURCE_CORPUS_KIND), generated(
        COUNTERFACTUAL_SOURCE_CORPUS_V2_KIND)
    origins = {row.example_id: row for row in build_semantic_program_natural_source_corpus(seed=41)
               if row.split == 'train'}
    assert [row.source_text for row in legacy] == [row.source_text for row in corrected]
    assert {row.contrast_id for row in legacy} == set(origins)
    assert {row.contrast_id for row in corrected} == {
        hashlib.sha256(row.source_text.encode('utf-8')).hexdigest() for row in origins.values()}
    assert all(row.construction_id.startswith('counterfactual-bound-v2:') for row in corrected)


def test_counterfactual_features_roundtrip_through_the_existing_bundle(tmp_path):
    import asyncio
    from core.learning.semantic_program_feature_materialization import (
        SemanticFeatureConfig, FAMILY_FEATURE_CONFIG_SCHEMA, build_semantic_program_corpus_for_config,
        COUNTERFACTUAL_SOURCE_CORPUS_KIND, materialize_semantic_program_features,
        load_standard_semantic_feature_bundle,
    )
    from tests.test_semantic_program_feature_materialization import (
        _FeatureClient, _CharacterTokenizer, _lane_receipt, _tokenizer_identity,
    )
    checkpoint = tmp_path / 'model'
    checkpoint.mkdir()
    output = tmp_path / 'features'
    config = SemanticFeatureConfig(seed=43, corpus_kind=COUNTERFACTUAL_SOURCE_CORPUS_KIND,
                                   schema=FAMILY_FEATURE_CONFIG_SCHEMA, max_examples=4096)
    corpus = build_semantic_program_corpus_for_config(config)
    client = _FeatureClient(checkpoint)
    result = asyncio.run(materialize_semantic_program_features(client=client,
        tokenizer=_CharacterTokenizer(), checkpoint=checkpoint, output_directory=output,
        corpus=corpus, config=config, lane_ownership_receipt=_lane_receipt(checkpoint),
        tokenizer_identity=_tokenizer_identity(checkpoint)))
    assert result.complete
    bundle = load_standard_semantic_feature_bundle(output)
    assert len(bundle.examples) == client.calls == len(corpus)
    assert bundle.manifest['config']['corpus_kind'] == COUNTERFACTUAL_SOURCE_CORPUS_KIND
    assert bundle.manifest['split_counts'] == {'train': len(corpus)}
