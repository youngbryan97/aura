"""A named result is read back to the step that named it, not to an input."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_argument_antecedent import (
    FEATURES,
    ArgumentAntecedent,
    argument_antecedent_from_dict,
    fit_argument_antecedent,
    register_stretches,
)
from core.learning.semantic_program_ir import TokenSpan

CHANNELS = ("input_token_embedding", "middle_causal_hidden")
WIDTH = 24


def _vectors(words: list[str], seed: int) -> np.ndarray:
    """Each word one fixed direction; the middle layer adds a little context."""
    vocabulary = sorted(set(words))
    directions = np.random.default_rng(7).normal(size=(len(vocabulary), WIDTH))
    embedding = np.stack([directions[vocabulary.index(word)] for word in words])
    noise = np.random.default_rng(seed).normal(scale=0.2, size=embedding.shape)
    return np.concatenate([embedding, embedding + noise], axis=1)


def _request(split: str, names: tuple[str, str], seed: int):
    """'a = 1 ; b = 2 ; c = 3 ; add a and b . save as N0 . sub c from b . save as N1 . mul N0 by N1'."""
    first, second = names
    words = ("a = 1 ; b = 2 ; c = 3 ; add a and b . save as " + first + " . sub c from b . save as "
             + second + " . mul " + first + " by " + second).split()
    at = {index: word for index, word in enumerate(words)}

    def span(start: int) -> TokenSpan:
        return TokenSpan(start, start + 1)

    inputs = (span(2), span(6), span(10))
    add, sub, mul = (next(i for i, w in at.items() if w == op) for op in ("add", "sub", "mul"))
    instructions = (
        SimpleNamespace(operation_span=span(add), args=(0, 1), argument_spans=(span(add + 1), span(add + 3))),
        SimpleNamespace(operation_span=span(sub), args=(1, 2), argument_spans=(span(sub + 3), span(sub + 1))),
        SimpleNamespace(operation_span=span(mul), args=(3, 4), argument_spans=(span(mul + 1), span(mul + 3))),
    )
    ir = SimpleNamespace(input_spans=inputs, instructions=instructions, n_inputs=3,
                         source_text_sha256=f"{split}-{first}-{second}-{seed}")
    return SimpleNamespace(split=split, ir=ir, hidden_states=_vectors(words, seed),
                           hidden_channels=CHANNELS, hidden_channel_widths=(WIDTH, WIDTH))


TRAINING = [_request("train", names, seed) for seed, names in enumerate(
    [("lead", "side"), ("primary", "auxiliary"), ("first", "second"), ("upper", "lower")]
)]


def test_each_register_owns_its_declaration_or_its_clause() -> None:
    stretches = register_stretches(
        [TokenSpan(2, 3), TokenSpan(6, 7)], [TokenSpan(10, 11), TokenSpan(20, 21)], 30
    )
    assert stretches == ((0, 3), (3, 7), (10, 20), (20, 30))
    # Inline literals after the operations: a declaration holds no operation.
    nested = register_stretches(
        [TokenSpan(12, 13), TokenSpan(17, 18)], [TokenSpan(2, 5), TokenSpan(7, 10)], 24
    )
    assert nested[:2] == ((10, 13), (13, 18))


def test_the_fitter_sees_training_rows_only() -> None:
    with pytest.raises(ValueError, match="training rows only"):
        fit_argument_antecedent([*TRAINING, _request("validation", ("x", "y"), 9)])
    fitted = fit_argument_antecedent(TRAINING)
    assert fitted.fit_receipt["splits_used"] == ["train"]
    assert argument_antecedent_from_dict(fitted.to_dict()).identity_sha256 == fitted.identity_sha256


def test_a_name_never_seen_in_training_is_read_back_to_the_step_that_gave_it() -> None:
    fitted = fit_argument_antecedent(TRAINING)
    item = _request("test", ("refined", "spare"), 11)
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    multiply = item.ir.instructions[2]
    for mention, named in zip(multiply.argument_spans, multiply.args, strict=True):
        scores = scorer.log_probabilities(mention)
        assert int(np.argmax(scores)) == named
        assert math.isclose(sum(math.exp(value) for value in scores), 1.0, rel_tol=1e-9)
    # An input's name read back to its declaration.
    subtract = item.ir.instructions[1]
    assert int(np.argmax(scorer.log_probabilities(subtract.argument_spans[0]))) == 1


def test_a_mention_with_no_earlier_window_spreads_its_probability() -> None:
    fitted = fit_argument_antecedent(TRAINING)
    item = TRAINING[0]
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    first_token = scorer.log_probabilities(TokenSpan(0, 1))
    assert max(first_token) - min(first_token) < 1.0


def test_decode_scores_arguments_with_antecedents() -> None:
    from core.learning.semantic_argument_ownership import fit_argument_ownership
    from core.learning.semantic_operation_peaks import (
        PeakRecognitionTransducer,
        fit_peak_operation_recognizer,
        peak_recognition_transducer_from_dict,
    )
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    training = tuple(item for item in examples if item.split == "train")
    base = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    recognizer = fit_peak_operation_recognizer(training)
    # In this fixture every mention follows every operation, so which operation
    # owns a mention is the ownership readout's to decide; antecedents ground it.
    ownership = fit_argument_ownership(training)
    antecedent = fit_argument_antecedent(training)
    candidate = PeakRecognitionTransducer(base, recognizer, ownership, antecedent)
    assert candidate.receipt_sha256 != PeakRecognitionTransducer(base, recognizer, ownership).receipt_sha256
    restored = peak_recognition_transducer_from_dict(
        candidate.to_dict(), restore_base=compositional_semantic_program_transducer_from_dict
    )
    assert restored.receipt_sha256 == candidate.receipt_sha256
    item = next(item for item in examples if item.split == "test" and len(item.ir.instructions) == 3)
    arguments = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
                     public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
                     model_basis_sha256=base.model_basis_sha256)
    decoded = candidate.decode(**arguments)
    assert decoded.ir is not None and decoded.ir.to_program() == item.ir.to_program()
    # A readout that sends every mention away from where its name was given moves the program.
    perverse = ArgumentAntecedent(tuple(-50.0 * value for value in antecedent.weight), 0.0, antecedent.fit_receipt)
    moved = PeakRecognitionTransducer(base, recognizer, ownership, perverse).decode(**arguments)
    assert moved.ir is None or moved.ir.to_program() != item.ir.to_program()


def test_an_inputs_literal_value_gets_no_antecedent_evidence() -> None:
    """The literal grammar binds it exactly; the readout is neither fitted on nor applied to it."""
    fitted = fit_argument_antecedent(TRAINING)
    item = TRAINING[1]
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    assert set(scorer.log_probabilities(item.ir.input_spans[1])) == {0.0}


def test_relative_scoring_says_which_register_and_not_which_span() -> None:
    """Every span's best register scores zero; only a worse register pays."""
    from dataclasses import replace

    fitted = replace(fit_argument_antecedent(TRAINING), scoring="relative")
    assert argument_antecedent_from_dict(fitted.to_dict()).scoring == "relative"
    item = TRAINING[2]
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    multiply = item.ir.instructions[2]
    for mention, named in zip(multiply.argument_spans, multiply.args, strict=True):
        scores = [scorer.score(mention, register) for register in range(len(scorer.stretches))]
        assert scores[named] == 0.0 and all(value <= 0.0 for value in scores)
    with pytest.raises(ValueError, match="absolute or relative"):
        replace(fitted, scoring="sideways")


def test_a_conditional_fit_puts_each_mentions_own_register_first() -> None:
    """The readout is used as each mention's distribution over registers, and can be fitted as one."""
    fitted = fit_argument_antecedent(TRAINING, objective="conditional")
    assert fitted.fit_receipt["objective"] == "conditional" and fitted.bias == 0.0
    item = _request("test", ("refined", "spare"), 11)
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = fitted.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    multiply = item.ir.instructions[2]
    for mention, named in zip(multiply.argument_spans, multiply.args, strict=True):
        assert int(np.argmax(scorer.log_probabilities(mention))) == named
    with pytest.raises(ValueError, match="pairwise or conditionally"):
        fit_argument_antecedent(TRAINING, objective="sideways")


def test_a_mention_that_names_an_operations_own_result_is_not_its_input() -> None:
    from dataclasses import replace

    fitted = fit_argument_antecedent(TRAINING)
    ruled = replace(fitted, own_result_is_not_an_input=True)
    assert argument_antecedent_from_dict(ruled.to_dict()).own_result_is_not_an_input is True
    item = TRAINING[0]
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    for readout, expected in ((fitted, False), (ruled, True)):
        scorer = readout.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
        multiply = item.ir.instructions[2]
        mention, named = multiply.argument_spans[0], multiply.args[0]
        # The multiply's first mention names the add's result: that is the add's own result.
        assert scorer.names_own_result(mention, named) is expected
        assert scorer.names_own_result(item.ir.input_spans[0], 0) is False


def test_an_input_named_elsewhere_is_used_through_its_name() -> None:
    from dataclasses import replace

    ruled = replace(fit_argument_antecedent(TRAINING), named_inputs_are_used_by_name=True)
    assert argument_antecedent_from_dict(ruled.to_dict()).named_inputs_are_used_by_name is True
    item = _request("test", ("refined", "spare"), 11)
    operations = [instruction.operation_span for instruction in item.ir.instructions]
    scorer = ruled.scorer(item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations)
    # "b" is named in the add and the subtraction, so its literal "2" is a declaration, not a use.
    mentions = [span for step in item.ir.instructions for span in step.argument_spans]
    named = scorer.inputs_used_by_name(mentions)
    assert 1 in named
    assert scorer.is_declaration_of(item.ir.input_spans[1], named)
    plain = fit_argument_antecedent(TRAINING).scorer(
        item.hidden_states, CHANNELS, (WIDTH, WIDTH), item.ir.input_spans, operations
    )
    assert plain.inputs_used_by_name(mentions) == frozenset()


def test_an_operation_owns_its_whole_sentence_when_sentence_ends_are_known() -> None:
    """ "Form the lead calculation by subtract ..." names the result before the operation's word.

    Tokens: 0-9 "inputs are a = 7 ; b = 3 ." (a@4, b@8), 10-13 "Form the lead calculation by"
    (10-14), 15 subtract, 16-19 "b from a .", 20-25 "Save that as the result .", 26 "Then",
    27 add, 28 "it", 29 multiply ...
    """
    a, b = TokenSpan(4, 5), TokenSpan(8, 9)
    subtract, add, multiply = TokenSpan(15, 16), TokenSpan(27, 28), TokenSpan(29, 30)
    sentences = (0, 10, 20, 26)

    stretches = register_stretches((a, b), (subtract, add, multiply), 34, sentences)

    assert stretches[2] == (10, 27), "the subtraction owns its lead-in and the sentence naming its result"
    assert stretches[3] == (27, 29), "operations sharing a sentence each start at their own word"
    assert stretches[4] == (29, 34)
    # Without sentence ends, as before.
    assert register_stretches((a, b), (subtract, add, multiply), 34)[2] == (15, 27)


def test_a_full_stop_inside_a_literal_ends_no_sentence() -> None:
    from core.learning.semantic_argument_antecedent import sentence_starts

    period = 13
    tokens = [1, 2, 3, period, 4, 5, period, 6, 7]
    assert sentence_starts(tokens, (period,), (TokenSpan(2, 5),)) == (0, 7)


def test_the_readout_keeps_its_sentence_ends_through_serialization() -> None:
    readout = ArgumentAntecedent(tuple(0.0 for _ in FEATURES), 0.0, {}, sentence_end_token_ids=(13, 30))

    assert argument_antecedent_from_dict(readout.to_dict()).sentence_end_token_ids == (13, 30)


def test_no_argument_option_starts_before_its_operations_sentence() -> None:
    """scalar_branch_weave_five-0-0: "add intake flow and return flow" took a declaration two sentences back."""
    from core.learning.semantic_argument_antecedent import within_sentence

    add, subtract = SimpleNamespace(span=TokenSpan(64, 65)), SimpleNamespace(span=TokenSpan(81, 82))
    declaration, name, later = TokenSpan(30, 34), TokenSpan(68, 70), TokenSpan(90, 92)
    options = [((declaration, 1.0), (name, 0.5), (later, 0.2)), ((TokenSpan(24, 25), 1.0), (TokenSpan(85, 87), 0.4))]

    bounded = within_sentence(options, (add, subtract), sentences=(0, 59, 78))

    assert bounded[0] == ((name, 0.5), (later, 0.2)), "what comes after the clause is kept"
    assert bounded[1] == ((TokenSpan(85, 87), 0.4),)


def _chain(split: str, steps: int, seed: int):
    """'v0 = 1 ; ... ; first add v0 and v1 . then sub v2 from it . then mul it by v3 ...': each "it" is the step before."""
    ops = ("add", "sub", "mul")
    words = []
    for index in range(steps + 1):
        words += [f"v{index}", "=", str(index + 1), ";"]
    inputs = [TokenSpan(4 * index + 2, 4 * index + 3) for index in range(steps + 1)]
    instructions = []
    for step in range(steps):
        op_at = len(words) + 1
        if step == 0:
            words += ["first", ops[0], "v0", "and", "v1", "."]
            instructions.append(SimpleNamespace(operation_span=TokenSpan(op_at, op_at + 1), args=(0, 1),
                argument_spans=(TokenSpan(op_at + 1, op_at + 2), TokenSpan(op_at + 3, op_at + 4))))
        else:
            words += ["then", ops[step % 3], "it", "with", f"v{step + 1}", "."]
            instructions.append(SimpleNamespace(operation_span=TokenSpan(op_at, op_at + 1),
                args=(steps + 1 + step - 1, step + 1),
                argument_spans=(TokenSpan(op_at + 1, op_at + 2), TokenSpan(op_at + 3, op_at + 4))))
    ir = SimpleNamespace(input_spans=tuple(inputs), instructions=tuple(instructions), n_inputs=steps + 1,
                         source_text_sha256=f"{split}-chain-{steps}-{seed}")
    return SimpleNamespace(split=split, ir=ir, hidden_states=_vectors(words, seed),
                           hidden_channels=CHANNELS, hidden_channel_widths=(WIDTH, WIDTH))


def test_recency_counts_the_operations_between_a_register_and_the_mention() -> None:
    from core.learning.semantic_argument_antecedent import _item_stretches, _Similarities, antecedent_features

    item = _chain("train", 4, 0)
    stretches = _item_stretches(item, ())
    last = item.ir.instructions[-1]
    rows = antecedent_features(_Similarities(item.hidden_states, CHANNELS, (WIDTH, WIDTH)),
                               last.argument_spans[0], stretches, item.ir.n_inputs, recency=True)
    assert all(len(row) == len(FEATURES) + 2 for row in rows)
    between = [row[-2] for row in rows[item.ir.n_inputs:]]
    # Registers of steps 0..3 as seen from step 3's "it": the step before has only step 3 between.
    assert between == [3.0, 2.0, 1.0, 0.0]
    first = item.ir.instructions[1]
    early = antecedent_features(_Similarities(item.hidden_states, CHANNELS, (WIDTH, WIDTH)),
                                first.argument_spans[0], stretches, item.ir.n_inputs, recency=True)
    assert [row[-1] for row in early[item.ir.n_inputs:]] == [0.0, 0.0, 1.0, 1.0]


def test_a_recency_readout_fitted_on_short_chains_reads_long_ones_back_one_step() -> None:
    """Chains of three to five steps, where the first clause to match and the step before differ."""
    training = [_chain("train", steps, seed) for seed, steps in enumerate((3, 4, 5) * 4)]
    readout = fit_argument_antecedent(training, objective="conditional", recency=True)
    assert readout.weight[-2] < 0
    assert readout.to_dict()["schema"].endswith(".v3")
    assert argument_antecedent_from_dict(readout.to_dict()) == readout
    long = _chain("train", 7, 99)
    scorer = readout.scorer(long.hidden_states, CHANNELS, (WIDTH, WIDTH), long.ir.input_spans,
                            tuple(ins.operation_span for ins in long.ir.instructions))
    n = long.ir.n_inputs
    for step, instruction in enumerate(long.ir.instructions[1:], 1):
        values = scorer.log_probabilities(instruction.argument_spans[0])
        others = [value for register, value in enumerate(values) if register != n + step]
        assert values[n + step - 1] == max(others)


def test_a_readout_without_recency_keeps_its_schema() -> None:
    plain = fit_argument_antecedent(TRAINING, objective="conditional")
    assert plain.to_dict()["schema"].endswith(".v2") and not plain.recency
    assert argument_antecedent_from_dict(plain.to_dict()) == plain
