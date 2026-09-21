"""Program-first language variations with independently checked semantic contrasts."""

from dataclasses import replace
from collections.abc import Iterator
import random
import string

from core.learning.procedure_induction import Instruction
from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_corpus import (
    SemanticInstructionAnnotation, _AnnotatedText, _append_natural_binary_operation,
)
from core.learning.semantic_program_floor import (
    compile_source_independent_program_to_floor, execute_semantic_floor_program,
    semantic_primitive_type_signature, semantic_program_structural_key,
)
from typing import Any

_BINARY_LANGUAGE = ('add', 'sub', 'mul', 'idiv', 'at', 'count_of')


def render_bound_program(
    source: Any,
    *,
    names: Any,
    clause_order: list[Any],
    program: Any=None,
) -> Any:
    """Render named dependencies independently of their textual clause order."""
    program = source.program if program is None else program
    count = program.n_inputs
    if (source.split != 'train' or count != len(source.inputs)
            or semantic_program_structural_key(program) is None
            or len(names) != count + len(program.instructions)
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.isascii() or not name.isalpha() for name in names)
            or any(type(index) is not int for index in clause_order)
            or sorted(clause_order) != list(range(len(program.instructions)))
            or any(ins.op not in _BINARY_LANGUAGE for ins in program.instructions)):
        raise ValueError('counterfactual source or rendering contract is unsupported')
    types = ['integer_sequence' if isinstance(value, tuple) else 'integer' for value in source.inputs]
    for ins in program.instructions:
        signature = semantic_primitive_type_signature(ins.op)
        if tuple(types[index] for index in ins.args) != signature[0]:
            raise ValueError('counterfactual program violates the floor type contract')
        types.append(signature[1])
    try:
        compiled = compile_source_independent_program_to_floor(program, source.inputs,
            provenance_receipt_sha256=_sha({'source': source.example_id, 'program': program.sha()}))
        execution = execute_semantic_floor_program(compiled)
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
        raise ValueError(f'counterfactual program cannot execute its printed inputs: {exc}') from exc
    if execution.result != program.run(source.inputs):
        raise ValueError('counterfactual floor and primitive execution disagree')
    text = _AnnotatedText()
    text.append('Use these recorded values: ')
    for index, value in enumerate(source.inputs):
        if index:
            text.append('; ')
        text.append(names[index], label=f'definition:{index}')
        text.append(' = ')
        literal = '[' + ', '.join(map(str, value)) + ']' if isinstance(value, tuple) else str(value)
        text.append(literal, label=f'input:{index}')
    text.append('. ')
    annotations = {}
    for ordinal in clause_order:
        ins = program.instructions[ordinal]
        text.append('To obtain ')
        text.append(names[count + ordinal], label=f'definition:{count + ordinal}')
        text.append(', ')
        left, right = (names[index] for index in ins.args)
        labels = (f'argument:{ordinal}:0', f'argument:{ordinal}:1')
        if ins.op in ('at', 'count_of'):
            text.append('select the item at' if ins.op == 'at' else 'count', label=f'natural:operation:{ordinal}')
            text.append(' ' if ins.op == 'at' else ' how often ')
            text.append(right, label=labels[1])
            text.append(' in ' if ins.op == 'at' else ' occurs in ')
            text.append(left, label=labels[0])
        else:
            _append_natural_binary_operation(text, op=ins.op, ordinal=ordinal,
                left_text=left, left_label=labels[0], right_text=right, right_label=labels[1])
        text.append('. ')
        annotations[ordinal] = SemanticInstructionAnnotation(ins, text.span(f'natural:operation:{ordinal}'),
            tuple(text.span(label) for label in labels), tuple(sorted({arg - count for arg in ins.args if arg >= count})))
    text.append('Return ' + names[-1] + '.')
    identity = _sha({'source': source.example_id, 'text': text.text, 'program': program.sha()})
    return replace(source, example_id=identity, construction_id='counterfactual-bound-v1',
        topology_id=_sha(semantic_program_structural_key(program)), source_text=text.text,
        input_spans=tuple(text.span(f'input:{index}') for index in range(count)),
        instructions=tuple(annotations[index] for index in range(len(annotations))),
        report_value=count + len(annotations) - 1, contrast_id=source.example_id,
        register_definition_spans=tuple(text.span(f'definition:{index}') for index in range(len(names))))


def _names(rng: Any, count: Any) -> tuple[Any, ...]:
    names = []
    while len(names) < count:
        value = ''.join(rng.choice(string.ascii_lowercase) for _ in range(8))
        if value not in names:
            names.append(value)
    return tuple(names)


def equivalent_recompositions(program: Any) -> Iterator[Any]:
    """Rotate single-use associative subtrees without changing public inputs."""
    for child, instruction in enumerate(program.instructions):
        if instruction.op not in ('add', 'mul'):
            continue
        register = program.n_inputs + child
        uses = [(index, slot) for index, ins in enumerate(program.instructions)
                for slot, argument in enumerate(ins.args) if argument == register]
        if len(uses) != 1:
            continue
        parent, slot = uses[0]
        parent_instruction = program.instructions[parent]
        if parent_instruction.op != instruction.op:
            continue
        other = parent_instruction.args[1 - slot]
        if other >= register:
            continue
        left, right = instruction.args
        changed = list(program.instructions)
        changed[child] = Instruction(instruction.op, (right, other))
        changed[parent] = Instruction(instruction.op, (left, register))
        candidate = replace(program, instructions=tuple(changed))
        if (candidate != program
                and compare_program_meanings(program, candidate, ())['status'] == 'equivalent'):
            yield candidate


def augment_source_programs(
    examples: tuple[Any, ...],
    *,
    seed: int=0,
    variations: int=2,
    forbidden_constructions: tuple[Any, ...]=(),
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rename/reorder source programs and retain only witnessed meaning changes."""
    if type(seed) is not int or type(variations) is not int or variations < 1:
        raise ValueError('counterfactual generation settings are invalid')
    examples = tuple(examples)
    sources = tuple(item for item in examples if item.split == 'train')
    if len({item.example_id for item in sources}) != len(sources):
        raise ValueError('counterfactual sources repeat an identity')
    if any(item.construction_id in forbidden_constructions for item in sources):
        raise ValueError('counterfactual source construction is reserved for evaluation')
    rows, records = [], []
    for source in sources:
        rng = random.Random(_sha({'seed': seed, 'source': source.example_id}))
        names = _names(rng, len(source.inputs) + len(source.instructions))
        for variant in range(variations):
            order = list(range(len(source.instructions)))
            if variant:
                if variant == 1:
                    order.reverse()
                else:
                    rng.shuffle(order)
                names = _names(rng, len(names))
            try:
                rendered = render_bound_program(source, names=names, clause_order=order)
            except ValueError as exc:
                records.append({'source': source.example_id, 'status': 'unsupported', 'reason': str(exc)})
                break
            proof = compare_program_meanings(source.program, rendered.program, ())
            assert proof['status'] == 'equivalent'
            rows.append(rendered)
            records.append({'source': source.example_id, 'example': rendered.example_id,
                            'status': 'equivalent', 'comparison': proof})
        else:
            for program in equivalent_recompositions(source.program):
                rendered = render_bound_program(source, names=names, clause_order=order, program=program)
                rows.append(rendered)
                records.append({'source': source.example_id, 'example': rendered.example_id,
                    'status': 'equivalent', 'transformation': 'associative_recomposition',
                    'comparison': compare_program_meanings(source.program, program, ())})
            # Mutations operate on typed SSA edges and primitives, never on answer labels.
            mutations = []
            for index, annotation in enumerate(source.instructions):
                ins = annotation.instruction
                for operation in _BINARY_LANGUAGE:
                    if operation != ins.op and semantic_primitive_type_signature(operation) == semantic_primitive_type_signature(ins.op):
                        mutations.append((index, Instruction(operation, ins.args)))
                if len(set(ins.args)) == 2:
                    mutations.append((index, Instruction(ins.op, tuple(reversed(ins.args)))))
            rng.shuffle(mutations)
            for index, instruction in mutations:
                program = replace(source.program, instructions=tuple(
                    instruction if ordinal == index else ann.instruction for ordinal, ann in enumerate(source.instructions)))
                try:
                    rendered = render_bound_program(source, names=names, clause_order=order, program=program)
                except ValueError:
                    continue
                comparison = compare_program_meanings(source.program, program, counterfactual_inputs(source.inputs))
                if comparison['status'] != 'different':
                    continue
                rows.append(rendered)
                records.append({'source': source.example_id, 'example': rendered.example_id,
                                'status': 'different', 'comparison': comparison})
                break
            else:
                records.append({'source': source.example_id, 'status': 'no_witnessed_mutation'})
    body = {'schema': 'aura.semantic_counterfactual_corpus.v1', 'seed': seed, 'variations': variations,
        'source_training_ids': sorted(item.example_id for item in sources),
        'ignored_nontraining_examples': sum(item.split != 'train' for item in examples),
        'forbidden_constructions': sorted(forbidden_constructions), 'records': records,
        'generated_examples': len(rows), 'test_examples_used': 0, 'serving_authority': False}
    return tuple(rows), {**body, 'receipt_sha256': _sha(body)}


def build_semantic_counterfactual_source_corpus(
    *,
    seed: int=0,
    examples_per_schema_domain: int=1,
) -> Any:
    """Augment existing natural source training, never its validation/test domains."""
    from core.learning.semantic_program_corpus_natural import build_semantic_program_natural_source_corpus

    sources = build_semantic_program_natural_source_corpus(
        seed=seed, examples_per_schema_domain=examples_per_schema_domain)
    rows, receipt = augment_source_programs(sources, seed=seed)
    if any(row['status'] == 'unsupported' for row in receipt['records']):
        raise ValueError('declared counterfactual source corpus has unsupported programs')
    return rows
