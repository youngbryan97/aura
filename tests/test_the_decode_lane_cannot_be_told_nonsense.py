"""Values nobody checked reached the model, the sampler, and the receipt.

**Token IDs.** ``token_ids`` was copied verbatim into the embedding lookup. A
float, a bool, or a negative index got there and the backend decided what it
meant — after the input commitment had already been hashed over values the
model would not read the same way.

**Prompt size.** The full token list was materialized and JSON-serialized into
the commitment payload before anything asked whether the episode could pay for
the prefill. The refusal arrived after the expensive part.

**Non-finite numbers.** Sampling consumed logits directly, so NaN produced a
backend-dependent token and the episode still reported ok. A logits digest over
a NaN vector fingerprints nothing. And a caller-supplied verifier could return
a bool or an infinity straight into branch ordering and the JSON receipt.

**The newline cap.** The module advertised at most two consecutive pure-newline
tokens, then masked only the one token it had just sampled and gave up after
four tries — returning a newline anyway. A model with several newline-family
tokens walked from one to the next under a rule that said two.

**An empty answer.** EOS as the first sampled token appended nothing and
terminated as "eos", which the acceptance set read as a complete decode.

**The optimizer flag.** ``latent_opt_no_accepted_step`` was emitted from the
attempt count, so a run that proposed forty edits and accepted none was
receipted as applied with no rejection flag anywhere.
"""
from __future__ import annotations

import ast
import inspect
import math

import pytest

import core.brain.llm.latent_cortex.engine as engine_mod
from core.brain.llm.latent_cortex.engine import (
    LatentCortexEngine,
    _validated_verifier_result,
)

# ─────────────────────────────── token ids are indices, or they are refused


class _Weight:
    def __init__(self, rows: int) -> None:
        self.shape = (rows, 8)


class _Embed:
    def __init__(self, rows: int) -> None:
        self.weight = _Weight(rows)


class _Inner:
    def __init__(self, rows: int) -> None:
        self.embed_tokens = _Embed(rows)


class _Model:
    def __init__(self, rows: int = 32, window: int = 0) -> None:
        self.model = _Inner(rows)
        if window:
            self.args = type("Args", (), {"max_position_embeddings": window})()


def _engine(*, vocab: int = 32, window: int = 0, layers: int = 4):
    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    model = _Model(vocab, window)
    engine.model = model
    # The engine reads the vocabulary off the DECODER — the inner module that
    # owns embed_tokens — which __init__ resolves through the layer view. This
    # fixture predated that and set only `model`, so every case in this file
    # died on AttributeError before reaching what it was testing.
    engine.decoder = model.model
    # And the position window off the LANGUAGE MODEL, which is the module
    # carrying `args`. Same refactor, same fixture gap.
    engine.language_model = model
    engine.tokenizer = None
    engine.n_layers = layers
    engine._vocab_size = 0
    engine._input_token_ceiling = None
    return engine


def test_valid_token_ids_pass_through_unchanged():
    engine = _engine()

    assert engine._validate_token_ids([0, 5, 31]) == [0, 5, 31]


@pytest.mark.parametrize("bad", [True, False])
def test_a_boolean_is_not_a_token_id(bad):
    engine = _engine()

    with pytest.raises(TypeError, match="not an exact integer"):
        engine._validate_token_ids([1, bad])


def test_a_float_is_not_a_token_id():
    engine = _engine()

    with pytest.raises(TypeError):
        engine._validate_token_ids([1.0])


def test_a_negative_id_is_refused_not_wrapped():
    """Negative indexing into an embedding table reads the far end of the
    vocabulary and calls it the caller's token."""
    engine = _engine()

    with pytest.raises(ValueError, match="outside the serving vocabulary"):
        engine._validate_token_ids([-1])


def test_an_out_of_vocabulary_id_is_refused():
    engine = _engine(vocab=32)

    with pytest.raises(ValueError, match=r"\[0, 32\)"):
        engine._validate_token_ids([32])


def test_the_position_of_the_bad_id_is_named():
    engine = _engine()

    with pytest.raises(ValueError, match=r"token_ids\[2\]"):
        engine._validate_token_ids([0, 1, 99])


# ─────────────────────────────── the prompt is admitted before it commits


def _budget(max_layer_apps: int = 1_000_000):
    from core.brain.llm.latent_cortex.types import ComputeBudget

    return ComputeBudget(max_layer_apps=max_layer_apps)


def test_a_prompt_inside_both_bounds_is_admitted():
    engine = _engine(window=4096)

    engine._admit_input_length(100, _budget())


def test_a_prompt_past_the_position_window_is_refused():
    engine = _engine(window=512)

    with pytest.raises(ValueError, match="512-token position window"):
        engine._admit_input_length(513, _budget())


def test_a_prompt_the_budget_cannot_pay_for_is_refused():
    from core.brain.llm.latent_cortex.loop_core import ComputeBudgetUnaffordable

    engine = _engine(window=100_000, layers=64)

    with pytest.raises(ComputeBudgetUnaffordable, match="layer applications"):
        engine._admit_input_length(50_000, _budget(max_layer_apps=1_000))


def test_a_sentinel_context_length_is_not_a_window():
    """Tokenizers ship model_max_length=1e30 to mean "unset". Read as a
    window it authorizes any prompt at all."""
    engine = _engine()
    engine.tokenizer = type("T", (), {"model_max_length": int(1e30)})()

    assert engine.input_token_ceiling() == 0


def test_the_model_window_outranks_the_tokenizer_value():
    engine = _engine(window=8192)
    engine.tokenizer = type("T", (), {"model_max_length": 2048})()

    assert engine.input_token_ceiling() == 8192


def test_the_admission_check_runs_before_the_commitment_payload():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    admit_line = None
    commit_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_admit_input_length"
        ):
            admit_line = node.lineno
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "encoded_tokens"
                for target in node.targets
            )
            and "json.dumps" in (ast.get_source_segment(source, node.value) or "")
        ):
            commit_line = node.lineno

    assert admit_line is not None, "nothing admits the input length"
    assert commit_line is not None, "the commitment payload was not found"
    assert admit_line < commit_line, (
        "the episode serializes every token before asking whether it can pay"
    )


# ─────────────────────────────── a verdict that cannot be ordered


def test_an_ordinary_score_passes_through():
    assert _validated_verifier_result(0.75) == 0.75
    assert _validated_verifier_result(0) == 0


def test_a_boolean_verdict_is_refused():
    """True reads as 1.0 and joins a scale it was never on."""
    with pytest.raises(TypeError, match="boolean"):
        _validated_verifier_result(True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_refused(value):
    with pytest.raises(ValueError, match="non-finite"):
        _validated_verifier_result(value)


def test_a_structured_verdict_is_left_alone():
    """Only real numbers are scores; richer results are validated by their
    own contracts."""
    receipt = {"score": 0.5, "reason": "ok"}

    assert _validated_verifier_result(receipt) is receipt


def test_every_verifier_result_crosses_the_same_boundary():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "charge"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "_validated_verifier_result(callback(rendered))" in rendered
        return
    raise AssertionError("the metered verifier boundary was not found")


def test_nan_cannot_win_by_losing_every_comparison():
    """NaN makes < and > both false, so a NaN branch survives every
    'is it worse?' test that guards selection."""
    assert not (float("nan") < 0.0)
    assert not (float("nan") > 0.0)

    with pytest.raises(ValueError):
        _validated_verifier_result(float("nan"))


# ─────────────────────────────── non-finite logits stop the episode


def test_the_sampler_checks_finiteness_before_it_samples():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_sample"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        guard = rendered.index("_require_finite_logits")
        sample = min(
            rendered.index("mx.random.categorical"),
            rendered.index("mx.argmax"),
        )
        assert guard < sample, "the guard runs after the token is drawn"
        return
    raise AssertionError("_sample was not found")


def test_token_suppression_cannot_overflow_float16_into_infinity():
    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import _mask_token_logits

    logits = mx.array([8.0, 4.0, -3.0], dtype=mx.float16)
    masked = _mask_token_logits(logits, mx.array([0]))
    mx.eval(masked)

    assert bool(mx.all(mx.isfinite(masked)).item())
    assert int(mx.argmax(masked).item()) == 1


def test_a_non_finite_forward_pass_cannot_be_fingerprinted():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_logits_digest"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "mx.isfinite" in rendered
        assert "NonFiniteLogitsError" in rendered
        return
    raise AssertionError("_logits_digest was not found")


def test_the_non_finite_error_fails_the_episode_rather_than_escaping():
    from core.brain.llm.latent_cortex.engine import (
        _LATENT_PHASE_ERRORS,
        NonFiniteLogitsError,
    )

    assert issubclass(NonFiniteLogitsError, _LATENT_PHASE_ERRORS)


# ─────────────────────────────── the newline cap is a cap


def test_the_whole_newline_family_is_masked_not_one_token():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "sample_disciplined"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "self._newline_token_ids()" in rendered
        assert "newline_discipline_exhausted" in rendered
        return
    raise AssertionError("sample_disciplined was not found")


def test_the_resample_attempt_budget_is_gone():
    """Four tries then emit it anyway is not a maximum."""
    source = inspect.getsource(engine_mod)

    assert "_NEWLINE_RESAMPLE_ATTEMPTS" not in source


def test_the_newline_family_is_derived_from_the_serving_vocabulary():
    engine = _engine(vocab=6)
    pieces = {0: "a", 1: "\n", 2: "\n\n", 3: "b", 4: " ", 5: "\r\n"}
    engine.tokenizer = type(
        "T", (), {"decode": staticmethod(lambda ids: pieces[ids[0]])}
    )()
    engine._newline_token_cache = {}
    engine._newline_token_ids_cache = None

    assert engine._newline_token_ids() == (1, 2, 5)


def test_the_newline_family_is_scanned_once():
    engine = _engine(vocab=4)
    calls: list[int] = []

    def decode(ids):
        calls.append(ids[0])
        return "\n" if ids[0] == 1 else "x"

    engine.tokenizer = type("T", (), {"decode": staticmethod(decode)})()
    engine._newline_token_cache = {}
    engine._newline_token_ids_cache = None

    engine._newline_token_ids()
    before = len(calls)
    engine._newline_token_ids()

    assert len(calls) == before


def test_an_exhausted_newline_discipline_terminates_the_decode():
    source = inspect.getsource(engine_mod)

    assert '"newline_discipline_exhausted"' in source
    assert '"newline_discipline_exhausted_before_decode"' in source


# ─────────────────────────────── an empty answer is not an answer


def test_a_decode_that_produced_nothing_is_not_a_success():
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "decode_incomplete:no_tokens_generated" not in rendered:
            continue
        assert "not out_tokens" in (ast.get_source_segment(source, node.test) or "")
        assert "decode_produced_no_tokens" in rendered
        return
    raise AssertionError("the empty-answer invariant was not found")


def test_the_empty_answer_reason_is_not_in_the_accepted_terminations():
    """If it were, the classification would undo itself."""
    source = inspect.getsource(engine_mod)

    assert '"no_tokens_generated",' not in source


# ─────────────────────────────── the flag names what it measures


def _flag_guard(source: str, tree: ast.AST, flag: str) -> str:
    """The test of the innermost ``if`` whose whole body raises this flag.

    Walking for any ``if`` that merely CONTAINS the flag text matches every
    enclosing block too, and the outermost one always wins.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            continue
        body = ast.get_source_segment(source, node.body[0]) or ""
        if body.strip() != f'receipt.flag("{flag}")':
            continue
        return ast.get_source_segment(source, node.test) or ""
    raise AssertionError(f"the {flag} guard was not found")


def test_the_no_accepted_step_flag_reads_accepted_steps():
    source = inspect.getsource(engine_mod)
    guard = _flag_guard(source, ast.parse(source), "latent_opt_no_accepted_step")

    assert "accepted" in guard, f"the rejection flag is still keyed off {guard!r}"
    assert "attempts" not in guard


def test_the_not_attempted_flag_reads_attempts():
    source = inspect.getsource(engine_mod)
    guard = _flag_guard(source, ast.parse(source), "latent_opt_not_attempted")

    assert "attempts" in guard


def test_a_run_that_never_started_gets_its_own_flag():
    source = inspect.getsource(engine_mod)

    assert 'receipt.flag("latent_opt_not_attempted")' in source


def test_forty_proposals_and_no_acceptances_is_not_applied_silently():
    """The two flags are distinct, so a consumer can tell 'the verifier
    rejected everything' from 'the optimizer never ran'."""
    source = inspect.getsource(engine_mod)

    assert source.count('"latent_opt_no_accepted_step"') == 1
    assert source.count('"latent_opt_not_attempted"') == 1


# ─────────────────────────────── provenance is carried, not reconstructed


def test_a_context_seed_states_which_item_it_came_from():
    from core.brain.llm.latent_cortex.workspace import _validated_context_seed

    assert _validated_context_seed((3, "memory", object()))[0] == 3


def test_the_two_element_seed_form_is_refused():
    from core.brain.llm.latent_cortex.workspace import _validated_context_seed

    with pytest.raises(TypeError, match="two-element form"):
        _validated_context_seed(("memory", object()))


def test_a_negative_seed_index_is_refused():
    from core.brain.llm.latent_cortex.workspace import _validated_context_seed

    with pytest.raises(ValueError):
        _validated_context_seed((-1, "memory", object()))


def test_the_workspace_uses_the_carried_index_not_the_row_count():
    import inspect as _inspect

    import core.brain.llm.latent_cortex.workspace as workspace_mod

    source = _inspect.getsource(workspace_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.DictComp):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "context_by_slot" in rendered or "1 + j" not in rendered:
            continue
        assert "int(index)" in rendered, (
            "the context index is still recomputed from the row count"
        )
        return

    assert "for j, (index, source, vector) in enumerate(seeds)" in source


def test_a_skipped_item_no_longer_shifts_its_neighbours_provenance():
    """The embedder skips an item that encodes to nothing. With the position
    inferred from the row count, every later slot recorded the wrong text."""
    source = inspect.getsource(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_embed_cognitive_context"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "for position, item in enumerate(items)" in rendered
        assert 'seeds.append((position, item["source"], pooled))' in rendered
        return
    raise AssertionError("_embed_cognitive_context was not found")


def test_finiteness_is_no_longer_rechecked_where_it_is_guaranteed():
    """A dead condition reads as a live guard to the next person."""
    source = inspect.getsource(engine_mod)

    assert "elif math.isfinite(probe_score):" not in source
    assert math.isfinite(1.0)
