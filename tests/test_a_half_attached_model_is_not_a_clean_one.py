"""A partial fast-weight attach left the model dirty and the receipt silent.

``fast_weights.attach()`` walks the resident layers and wraps them one at a
time. It ran outside the block whose handler detaches and proves erase, and
``fast_weights_applied`` was set only after it returned. So an attach that
mutated three of eight layers and then raised produced a receipt saying no
fast weights were applied, over a model that had them. The outer handler read
that receipt, decided the model was clean, and served a vanilla decode against
weights nobody could account for.

The same boundary was wrong two layers down. The runtime-integrity proof made
the erase evidence "required" only when the attach had completed, and the
cache-invalidation proof did the same — both went quiet in exactly the state
that needs them.

Separately: the engine mutates one shared resident model in place and had no
single-flight guard. Two overlapping callers interleaved fast-weight attaches
and checkpoint probes, and each one's proof then described a model the other
was also editing.
"""
from __future__ import annotations

import ast
import inspect
import threading

import pytest

import core.brain.llm.latent_cortex.engine as engine_mod
from core.brain.llm.latent_cortex.engine import (
    LatentCortexEngine,
    LatentEngineBusyError,
)
from tests.source_contract import family_text


# ─────────────────────────── the attach is inside the cleanup block


def _latent_episode_tree() -> tuple[ast.FunctionDef, str]:
    source = family_text(engine_mod)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_latent_episode":
            return node, source
    raise AssertionError("_latent_episode was not found")


def _attach_call(node: ast.AST) -> ast.Call:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "attach"
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "fast_weights"
        ):
            return child
    raise AssertionError("the fast-weight attach call was not found")


def test_the_attach_runs_inside_a_block_that_finalizes_on_failure():
    episode, _source = _latent_episode_tree()
    attach = _attach_call(episode)

    guarded = []
    for node in ast.walk(episode):
        if not isinstance(node, ast.Try):
            continue
        if not (node.lineno <= attach.lineno <= node.end_lineno):
            continue
        body_span = any(
            statement.lineno <= attach.lineno <= statement.end_lineno
            for statement in node.body
        )
        if not body_span:
            continue
        finalizes = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_finalize_fast_weights"
            for handler in node.handlers
            for call in ast.walk(handler)
        ) or any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_finalize_fast_weights"
            for statement in node.finalbody
            for call in ast.walk(statement)
        )
        if finalizes:
            guarded.append(node)

    assert guarded, (
        "fast_weights.attach() is not inside a block that detaches and proves "
        "erase when it raises"
    )


def test_the_model_is_marked_dirty_before_the_first_mutation():
    episode, _source = _latent_episode_tree()
    attach = _attach_call(episode)

    marks = [
        node.lineno
        for node in ast.walk(episode)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "fast_weights_attach_attempted"
            for target in node.targets
        )
    ]

    assert marks, "nothing records that the resident model was touched"
    assert min(marks) < attach.lineno, (
        "the dirty mark lands after the attach, so a partial attach still "
        "reports a clean model"
    )


def test_a_zero_layer_attach_is_not_reported_as_applied():
    episode, source = _latent_episode_tree()
    rendered = ast.get_source_segment(source, episode) or ""

    assert "fast-weight attach wrapped zero layers" in rendered


def test_the_vanilla_fallback_refuses_a_model_that_was_only_touched():
    source = family_text(engine_mod)

    assert "receipt.fast_weights_attach_attempted\n" in source
    assert "fallback_refused_unproven_model_state" in source

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node.test) or ""
        if "fast_weights_erased" not in rendered:
            continue
        if "fast_weights_applied" not in rendered:
            continue
        assert "fast_weights_attach_attempted" in rendered, (
            "the fallback guard still reads only the completed attach"
        )
        return
    raise AssertionError("the vanilla-fallback guard was not found")


# ─────────────────────────── the proof required in the dirty state


def _erase_proof(*, applied: bool, attempted: bool) -> dict:
    from core.brain.llm.latent_cortex.runtime_integrity import _fast_weight_erase

    return _fast_weight_erase(
        episode_id="episode-partial-attach",
        input_tokens_sha256="7" * 64,
        fast_weights_applied=applied,
        fast_weight_learning=None,
        fast_weight_cleanup=None,
        fast_weights_attach_attempted=attempted,
    )


def test_cleanup_evidence_is_required_after_a_partial_attach():
    proof = _erase_proof(applied=False, attempted=True)

    assert proof["required"] is True
    assert proof["exact"] is False


def test_cleanup_evidence_is_required_after_a_complete_attach():
    assert _erase_proof(applied=True, attempted=True)["required"] is True


def test_an_episode_that_never_touched_the_model_needs_no_erase_proof():
    assert _erase_proof(applied=False, attempted=False)["required"] is False


def _cache_proof(*, applied: bool, attempted: bool, invalidations: list[str]):
    from core.brain.llm.latent_cortex.runtime_integrity import _cache_proof as proof

    return proof(
        fast_weights_applied=applied,
        probe_cache={"entries": 0, "invalidations": invalidations},
        fast_weights_attach_attempted=attempted,
    )


def test_a_partial_attach_still_owes_a_cache_invalidation():
    assert _cache_proof(applied=False, attempted=True, invalidations=[])["safe"] is False


def test_a_partial_attach_that_invalidated_both_ends_is_safe():
    proof = _cache_proof(
        applied=False,
        attempted=True,
        invalidations=["fast_weights_attached", "fast_weights_detached"],
    )

    assert proof["safe"] is True


def test_an_untouched_model_owes_no_invalidation():
    assert _cache_proof(applied=False, attempted=False, invalidations=[])["safe"] is True


# ─────────────────────────── the expectation names the same state


def _bound_proof(*, applied: bool, attempted: bool):
    from tests.fixtures.rlc_runtime_integrity import bound_runtime_integrity

    return bound_runtime_integrity(
        episode_id="episode-partial-attach",
        input_tokens_sha256="7" * 64,
        fast_weights_applied=applied,
        fast_weights_attach_attempted=attempted,
    )


def test_a_caller_expecting_a_clean_model_is_told_the_scope_differs():
    from core.brain.llm.latent_cortex.runtime_integrity import (
        validate_runtime_integrity_receipt,
    )

    proof = _bound_proof(applied=False, attempted=True)

    with pytest.raises(ValueError, match="fast-weight scope mismatch"):
        validate_runtime_integrity_receipt(
            proof,
            require_worker=True,
            expected_fast_weights_applied=False,
            expected_fast_weights_attach_attempted=False,
        )


def test_the_matching_expectation_reconstructs():
    from core.brain.llm.latent_cortex.runtime_integrity import (
        validate_runtime_integrity_receipt,
    )

    proof = _bound_proof(applied=False, attempted=True)

    parsed = validate_runtime_integrity_receipt(
        proof,
        require_worker=True,
        expected_fast_weights_applied=False,
        expected_fast_weights_attach_attempted=True,
    )

    assert parsed["fast_weight_erase"]["required"] is True


def test_the_receipt_carries_the_mutation_fact():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")

    assert receipt.fast_weights_attach_attempted is False
    assert "fast_weights_attach_attempted" in receipt.to_dict()


def test_the_worker_recycles_the_cache_after_a_partial_attach():
    handler = inspect.getsource(
        __import__(
            "core.brain.llm.latent_cortex.worker_handler",
            fromlist=["worker_handler"],
        )
    )

    tree = ast.parse(handler)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [
            target
            for target in node.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "requires_cache_clear"
        ]
        if not targets:
            continue
        rendered = ast.get_source_segment(handler, node.value) or ""
        assert "fast_weights_attach_attempted" in rendered, (
            "a partial attach leaves the probe cache holding stale entries"
        )
        return
    raise AssertionError("requires_cache_clear was not found")


# ─────────────────────────── one episode at a time


def _guard_only_engine() -> LatentCortexEngine:
    """The single-flight guard needs no model — that is the point of it."""
    from core.runtime.lockdep import LockRank, checked_lock

    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    engine._episode_lock = checked_lock(
        "latent_cortex.engine.episode.test",
        rank=LockRank.UNRANKED,
    )
    engine._episode_holder = ""
    return engine


def test_a_second_episode_is_refused_while_one_holds_the_model():
    engine = _guard_only_engine()

    with engine._single_flight_episode():
        with pytest.raises(LatentEngineBusyError):
            with engine._single_flight_episode():
                raise AssertionError("two episodes held the model at once")


def test_the_model_is_released_when_an_episode_fails():
    engine = _guard_only_engine()

    with pytest.raises(RuntimeError, match="episode blew up"):
        with engine._single_flight_episode():
            raise RuntimeError("episode blew up")

    with engine._single_flight_episode():
        pass


def test_the_holder_is_named_while_an_episode_runs():
    engine = _guard_only_engine()

    assert engine.episode_in_flight() is False
    with engine._single_flight_episode():
        assert engine.episode_in_flight() is True
        assert str(threading.get_ident()) in engine._episode_holder
    assert engine.episode_in_flight() is False


def test_a_concurrent_caller_from_another_thread_is_refused():
    engine = _guard_only_engine()
    started = threading.Event()
    release = threading.Event()
    refused: list[BaseException] = []

    def hold():
        with engine._single_flight_episode():
            started.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert started.wait(5.0)

    try:
        with engine._single_flight_episode():
            pass
    except LatentEngineBusyError as exc:
        refused.append(exc)
    finally:
        release.set()
        holder.join(5.0)

    assert refused, "a second thread ran an episode against the same model"


def test_reason_runs_the_episode_under_the_guard():
    engine = _guard_only_engine()
    observed: list[bool] = []

    def fake_episode(*args, **kwargs):
        observed.append(engine.episode_in_flight())
        return "answer"

    engine._reason_episode = fake_episode

    assert engine.reason(prompt="hello") == "answer"
    assert observed == [True]
    assert engine.episode_in_flight() is False


def test_reason_still_advertises_its_real_parameters():
    """The guard forwards *args/**kwargs; callers and tests must still be able
    to read the signature."""
    parameters = inspect.signature(LatentCortexEngine.reason).parameters

    assert "prompt" in parameters
    assert "budget" in parameters
    assert "cognitive_context" in parameters
    assert "args" not in parameters


def test_a_busy_engine_does_not_ask_for_a_worker_recycle():
    handler = inspect.getsource(
        __import__(
            "core.brain.llm.latent_cortex.worker_handler",
            fromlist=["worker_handler"],
        )
    )
    tree = ast.parse(handler)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        rendered_type = ast.unparse(node.type) if node.type else ""
        if "LatentEngineBusyError" not in rendered_type:
            continue
        rendered = ast.get_source_segment(handler, node) or ""
        assert '"requires_worker_recycle": False' in rendered
        assert '"retryable": True' in rendered
        return
    raise AssertionError("the busy refusal was not found in the worker handler")
