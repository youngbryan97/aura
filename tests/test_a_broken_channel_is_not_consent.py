"""Callbacks that could not answer were read as answers.

`cancel_check` is how the owner of an episode says stop. Any exception from it
was swallowed and converted to False, with nothing on the receipt — so a dead
IPC channel read as "keep going", and expensive work continued after the owner
had lost the ability to stop it. `_emit_progress` had the mirror problem in a
quieter register: a monitoring consumer that lost stage updates could not tell
the gap from an episode that simply had no stages.

The probe decodes never received the cancellation channel at all. Fifteen call
sites, and threading three controls through each of them is how they went
missing; the single-flight guard makes one place the honest place to keep them.

Around the same boundary:

**Admission failures escaped the one-call contract.** Token encoding, the input
commitment and the pre-episode invariant all run before the episode's own
exception handling, so their failures came back as bare exceptions while every
later failure came back as a receipted result. A caller had to handle two
shapes for one class of problem — and once `pre_episode` had armed the
invariant, nothing closed it.

**The final text conversion ran after every handler.** A tokenizer exception
there threw away a complete structured result and the progress stream's
closing event with it.

**Raw exception text was published.** Reasons concatenated the message, which
for backend, tokenizer and filesystem errors carries local paths and model
directories, and sent it to the caller and the progress callback.

**Memory exhaustion allocated a fresh cache.** MemoryError was an ordinary
recoverable phase error, so the vanilla fallback answered device or host OOM by
asking for more of what had just run out.

**Stochastic decodes recorded no seed.** Temperature sampling used global MLX
randomness while the receipt recorded temperature and top-p, so the answer
could not be reproduced or tied to the state it came from.
"""
from __future__ import annotations

import ast
import inspect

import pytest

import core.brain.llm.latent_cortex.engine as engine_mod
from core.brain.llm.latent_cortex.engine import LatentCortexEngine
from tests.source_contract import family_text


def _engine():
    engine = LatentCortexEngine.__new__(LatentCortexEngine)
    engine._callback_faults = {}
    return engine


# ─────────────────────────── a channel that cannot answer


def test_no_callback_means_no_cancellation():
    assert _engine()._cancel_requested(None) is False


def test_a_working_channel_is_believed_either_way():
    engine = _engine()

    assert engine._cancel_requested(lambda: False) is False
    assert engine._cancel_requested(lambda: True) is True
    assert engine._callback_faults == {}


@pytest.mark.parametrize(
    "error", [OSError("broken pipe"), RuntimeError("closed"), AttributeError("gone")]
)
def test_a_broken_channel_is_treated_as_cancelled(error):
    """Returning False meant a dead channel read as permission to continue."""
    engine = _engine()

    def broken():
        raise error

    assert engine._cancel_requested(broken) is True


def test_a_channel_fault_is_receipted_as_its_own_kind():
    """A fault must never be mistaken for a person changing their mind."""
    engine = _engine()

    def broken():
        raise OSError("broken pipe")

    engine._cancel_requested(broken)

    assert engine._callback_faults["cancel_check"] == "OSError"


def test_an_ordinary_cancel_records_no_fault():
    engine = _engine()

    engine._cancel_requested(lambda: True)

    assert "cancel_check" not in engine._callback_faults


def test_a_failing_progress_callback_is_recorded_once():
    engine = _engine()
    calls: list[int] = []

    def broken(_payload):
        calls.append(1)
        raise RuntimeError("consumer went away")

    engine._emit_progress(broken, {"stage": "a"})
    engine._emit_progress(broken, {"stage": "b"})

    assert len(calls) == 2
    assert engine._callback_faults["progress"] == "RuntimeError"


def test_a_working_progress_callback_records_nothing():
    engine = _engine()
    seen: list[dict] = []

    engine._emit_progress(seen.append, {"stage": "a"})

    assert seen == [{"stage": "a"}]
    assert engine._callback_faults == {}


def test_the_faults_reach_the_receipt():
    source = family_text(engine_mod)

    assert 'receipt.flag(f"{channel}_callback_failed:{kind}")' in source


def test_the_episode_resets_the_fault_record():
    """A fault from the previous episode is not this episode's evidence."""
    source = family_text(engine_mod)

    assert "self._callback_faults = {}" in source


# ─────────────────────────── the probes inherit the episode's controls


def test_a_probe_refuses_to_start_after_cancellation():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_decode_probe"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "self._cancel_requested(self._episode_cancel_check)" in rendered
        assert '_LatentEpisodeCancelledError("verifier_probe")' in rendered
        return
    raise AssertionError("_decode_probe was not found")


def test_the_cancellation_check_precedes_the_memo_lookup():
    """Otherwise a cancelled caller gets an answer or an error depending on
    whether this probe happened to be cached."""
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_decode_probe"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        check = rendered.index("self._cancel_requested(self._episode_cancel_check)")
        memo = rendered.index("probe_cache.get(cache_key)")
        assert check < memo
        return
    raise AssertionError("_decode_probe was not found")


def test_a_probe_honours_the_cleanup_reserve():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_decode_probe"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "wall_reserve_forwards=self._episode_wall_reserve_forwards" in rendered
        return
    raise AssertionError("_decode_probe was not found")


def test_the_reserve_is_live_from_the_first_mutation_to_the_end_of_cleanup():
    source = family_text(engine_mod)

    assert (
        "self._episode_wall_reserve_forwards = _FW_ERASE_PROBE_TOKENS + 1" in source
    )
    assert source.count("self._episode_wall_reserve_forwards = 0") >= 2


# ─────────────────────────── one contract, one shape


def test_the_boundary_returns_a_receipted_result_for_runtime_failures():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "reason"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "except _LATENT_PHASE_ERRORS as exc:" in rendered
        assert 'receipt.flag(f"admission_failed:{type(exc).__name__}")' in rendered
        assert "ok=False," in rendered
        return
    raise AssertionError("the reason() boundary was not found")


def test_a_callers_contract_violation_still_raises():
    """Turning a malformed argument into ok=False hides the caller's bug."""
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "reason"):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "if receipt is None:" in rendered
        assert "raise" in rendered
        return
    raise AssertionError("the reason() boundary was not found")


def test_the_receipt_is_published_only_once_payload_validation_passed():
    """Payload refusals — a tampered memory authority above all — must stay
    loud rather than becoming a quiet ok=False."""
    source = family_text(engine_mod)
    tree = ast.parse(source)

    publish = None
    validate = None
    encode = None
    for node in ast.walk(tree):
        rendered = ast.get_source_segment(source, node) or ""
        if isinstance(node, ast.Assign) and rendered.strip() == "self._episode_receipt = receipt":
            publish = node.lineno
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_validate_cognitive_context":
            validate = node.lineno
        if (
            isinstance(node, ast.Assign)
            and rendered.strip() == "tokens = self._encode(prompt, messages, token_ids)"
        ):
            encode = node.lineno

    assert publish and validate and encode
    assert validate < publish < encode


def test_an_armed_invariant_is_closed_on_an_admission_failure():
    source = family_text(engine_mod)

    assert "def _close_armed_invariant(" in source
    assert "self._episode_invariant_armed = True" in source
    assert "self._close_armed_invariant(receipt)" in source


def test_the_invariant_is_not_closed_twice():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_close_armed_invariant"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "if not self._episode_invariant_armed:" in rendered
        assert "self._episode_invariant_armed = False" in rendered
        return
    raise AssertionError("_close_armed_invariant was not found")


# ─────────────────────────── the last step is guarded too


def test_a_tokenizer_failure_at_the_end_still_returns_a_result():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_public_text_or_receipt"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "except _LATENT_PHASE_ERRORS as exc:" in rendered
        assert 'public_text_conversion_failed:' in rendered
        return
    raise AssertionError("_public_text_or_receipt was not found")


def test_an_empty_answer_is_not_the_failure_signal():
    """An empty string is a real answer shape — no tokens, or a substrate
    engine with no tokenizer — so it cannot double as "conversion failed"."""
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_public_text_or_receipt"
        ):
            continue
        assert "tuple[str, bool]" in (ast.get_source_segment(source, node) or "")
        return
    raise AssertionError("_public_text_or_receipt was not found")


def test_the_success_path_checks_the_conversion_verdict():
    source = family_text(engine_mod)

    assert "text, converted = self._public_text_or_receipt(out_tokens, receipt)" in source
    assert "if not converted:" in source


# ─────────────────────────── reasons are codes, not messages


def test_a_reason_carries_the_class_and_not_the_message():
    secret_path = "/".join(["", "home", "someone", "models", "private-32b"])
    error = OSError(f"{secret_path}/weights.safetensors missing")

    reason = engine_mod._public_reason("latent_phase_failed", error)

    assert reason == "latent_phase_failed:OSError"
    assert "someone" not in reason
    assert "private-32b" not in reason


def test_no_reason_interpolates_the_exception_itself():
    source = family_text(engine_mod)

    assert '{type(exc).__name__}:{exc}' not in source
    assert '{type(exc).__name__}: {exc}' not in source


def test_the_public_codes_are_still_distinguishable():
    from core.brain.llm.latent_cortex.loop_core import ComputeBudgetUnaffordable

    declined = engine_mod._public_reason(
        "latent_budget_declined", ComputeBudgetUnaffordable("x")
    )
    failed = engine_mod._public_reason("latent_phase_failed", ValueError("x"))

    assert declined.startswith("latent_budget_declined:")
    assert failed.startswith("latent_phase_failed:")


# ─────────────────────────── exhaustion is not a retry


def test_memory_exhaustion_does_not_fall_back():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "fallback_refused_memory_exhaustion" not in rendered:
            continue
        assert "isinstance(exc, MemoryError)" in rendered
        assert "self._release_episode_memory()" in rendered
        assert 'failure_reason = "latent_memory_exhausted"' in rendered
        return
    raise AssertionError("the memory-exhaustion path was not found")


def test_the_exhaustion_path_releases_what_the_episode_held():
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef) and node.name == "_release_episode_memory"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        assert "self._episode_probe_cache = None" in rendered
        assert "self._episode_kv_state_tree = None" in rendered
        assert "clear_cache" in rendered
        return
    raise AssertionError("_release_episode_memory was not found")


def test_an_exhausted_episode_asks_for_a_worker_recycle():
    import core.brain.llm.latent_cortex.worker_handler as handler_mod

    source = inspect.getsource(handler_mod)

    assert "fallback_refused_memory_exhaustion" in source
    assert "not integrity_safe or memory_exhausted" in source


# ─────────────────────────── a stochastic answer names its randomness


def test_the_receipt_carries_a_seed_and_a_trace():
    from core.brain.llm.latent_cortex.types import EpisodeReceipt

    receipt = EpisodeReceipt(episode_id="e")

    assert receipt.decode_sample_seed == -1
    assert receipt.decode_sample_trace_sha256 == ""
    published = receipt.to_dict()
    assert "decode_sample_seed" in published
    assert "decode_sample_trace_sha256" in published


def test_a_deterministic_decode_says_so_rather_than_inventing_a_seed():
    source = family_text(engine_mod)

    assert (
        "self._last_decode_sample_seed = -1 if sample_seed is None else int(sample_seed)"
        in source
    )


def test_the_episode_derives_its_seed_from_its_own_commitment():
    """A fresh random number would be neither stable for the same inputs nor
    tied to them."""
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "sample_seed = int.from_bytes(" not in rendered:
            continue
        assert "receipt.input_tokens_sha256" in rendered
        assert "receipt.episode_id" in rendered
        assert "self.config.decode_temperature > 0.0" in (
            ast.get_source_segment(source, node.test) or ""
        )
        return
    raise AssertionError("the derived sample seed was not found")


def test_every_sampled_decision_enters_the_trace():
    source = family_text(engine_mod)

    assert (
        source.count("sample_trace.update(_sample_decision_bytes(token, token_logprob))")
        == 2
    )


def test_the_trace_commits_to_the_token_and_its_logprob():
    packed = engine_mod._sample_decision_bytes(42, -1.5)
    other = engine_mod._sample_decision_bytes(42, -1.6)

    assert packed != other
    assert engine_mod._sample_decision_bytes(43, -1.5) != packed


def test_both_decode_paths_publish_the_same_discipline_fields():
    """The fallback recorded four of the six and dropped exactly the two that
    say sampling was intervened in."""
    source = family_text(engine_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name == "_record_decode_discipline"
        ):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        for field in (
            "decode_requested_tokens",
            "decode_generated_tokens",
            "decode_termination",
            "decode_contract_satisfied",
            "decode_contract_grace_used_tokens",
            "decode_newline_suppressions",
            "decode_repetition_penalty_applied",
            "decode_sample_seed",
            "decode_sample_trace_sha256",
        ):
            assert f"receipt.{field}" in rendered, field
        break
    else:
        raise AssertionError("_record_decode_discipline was not found")

    assert source.count("self._record_decode_discipline(") == 2
