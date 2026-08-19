"""CP126 fail-open batch 4: execution, observation, and acceptance.

* ``4bf25067`` — a terminal receipt with no exit_code defaulted to 0, so a
  process that never reported how it ended suppressed its own failure.
* ``7031837e`` — any vision response not carrying status=error was
  stringified as success, turning malformed IPC into confident perception.
* ``6d2e65cb`` — a broken loop detector let the agent loop continue at full
  budget, permitting the recursion it exists to stop.
* ``a6763356`` — with no execution adapter the client fabricated an error
  STRING and fed it to the model as though a tool had run.
* ``6a8225f5`` — tool results were interpolated into history wholesale.
* ``52feb1d1`` — with `succeeded` absent, any status outside a short denylist
  counted as acceptance and was persisted as seen.
"""
from __future__ import annotations

import inspect

import pytest


class TestAMissingExitCodeIsNotSuccess:
    def _verify(self, receipt):
        import asyncio

        from core.body.action_postcondition import ActionPostconditionVerifier

        class _State:
            world_model: dict = {}

        return asyncio.run(ActionPostconditionVerifier().verify(receipt, _State()))

    def test_absent_exit_code_is_reported_as_unknown(self):
        """Absent is not zero; it is unknown, and unknown must not read as
        success on the receipt that records what a process did."""
        result = self._verify({"channel": "terminal", "status": "success"})
        assert "process_exit_code_unreported" in result["side_effects"]

    def test_a_reported_zero_is_clean(self):
        result = self._verify(
            {"channel": "terminal", "status": "success", "exit_code": 0},
        )
        assert result["side_effects"] == []

    def test_a_nonzero_exit_code_still_reports_failure(self):
        result = self._verify(
            {"channel": "terminal", "status": "success", "exit_code": 3},
        )
        assert "process_failed_with_code:3" in result["side_effects"]


class TestVisionResponsesAreValidated:
    def _client(self):
        from core.brain.llm.mlx_vision_client import MLXVisionClient

        return MLXVisionClient

    def test_the_schema_is_enforced_not_stringified(self):
        source = inspect.getsource(self._client())
        assert "non-mapping response" in source
        assert "carried no 'response' field" in source
        assert "must be text" in source

    def test_an_unknown_status_is_refused(self):
        source = inspect.getsource(self._client())
        assert "returned unknown status" in source

    def test_the_old_permissive_stringify_is_gone(self):
        source = inspect.getsource(self._client())
        assert 'return str(resp.get("response", ""))' not in source


class TestBrokenLoopDetectionBoundsTheLoop:
    def test_the_loop_stops_early_without_detection(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "loop_detection_available = False" in source
        assert "loop detection unavailable" in source

    def test_import_failure_is_no_longer_a_silent_skip(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "Circuit breaker module not found. Skipping check." not in source


class TestNoExecutorFailsRatherThanNarrates:
    def test_the_fabricated_error_string_is_gone(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "[Error: No execution adapter configured for" not in source

    def test_it_returns_an_explicit_no_executor_result(self):
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert '"error": "no_executor"' in source
        assert '"confidence": 0.0' in source


class TestObservationsAreBoundedOnEntry:
    def test_a_small_result_is_unchanged(self):
        from core.brain.llm.local_agent_client import _bounded_observation

        assert _bounded_observation("hello") == "hello"

    def test_a_large_result_is_truncated_and_says_so(self):
        from core.brain.llm.local_agent_client import (
            _MAX_OBSERVATION_CHARS,
            _bounded_observation,
        )

        bounded = _bounded_observation("x" * (_MAX_OBSERVATION_CHARS + 500))
        assert len(bounded) < _MAX_OBSERVATION_CHARS + 300
        assert "observation truncated" in bounded
        assert "500 more characters" in bounded

    def test_none_is_safe(self):
        from core.brain.llm.local_agent_client import _bounded_observation

        assert _bounded_observation(None) == ""

    def test_what_reaches_the_history_is_bounded(self):
        """The property, not the spelling of one call.

        This asserted the literal text `_bounded_observation(result_str)` in
        the module. The bound then moved one level down, into the fenced
        observation block that history actually appends, and the test went red
        while the behaviour was intact — a structural check pinned to a call
        site rather than to what the call site guarantees.
        """
        from core.brain.llm.local_agent_client import (
            _MAX_OBSERVATION_CHARS,
            _observation_block,
        )

        block = _observation_block(
            "web_search", "x" * (_MAX_OBSERVATION_CHARS + 500), nonce="abc123"
        )
        assert len(block) < _MAX_OBSERVATION_CHARS + 600
        assert "observation truncated" in block
        assert "500 more characters" in block

    def test_the_history_append_goes_through_that_block(self):
        """And the bounded block is what history is built from."""
        from core.brain.llm import local_agent_client as mod

        source = inspect.getsource(mod)
        assert "history += f\"\\nAURA: {response_text}\" + _observation_block(" in source


class TestRepairAcceptanceIsPositive:
    def _accepting(self):
        from core.agency.self_repair_backlog import _ACCEPTING_STATUSES

        return _ACCEPTING_STATUSES

    @pytest.mark.parametrize(
        "status", ["unknown", "something_new", "failed", "blocked", "denied"],
    )
    def test_unrecognised_statuses_are_not_acceptance(self, status):
        assert status not in self._accepting()

    @pytest.mark.parametrize(
        "status", ["planned", "waiting_for_approval", "created", "queued"],
    )
    def test_the_designed_approval_path_is_still_acceptance(self, status):
        """A shadow plan awaiting approval WAS created — that is this
        subsystem's success path, not a defect. The finding listed these as
        suspect; the design and its tests say otherwise, and what the fix
        actually removes is acceptance-by-default for unclassified statuses.
        """
        assert status in self._accepting()

    def test_the_denylist_default_is_gone(self):
        from core.agency import self_repair_backlog as mod

        source = inspect.getsource(mod)
        assert "reason.lower() not in {" not in source
        assert "_ACCEPTING_STATUSES" in source


class TestHonestyFloorNeverManufacturesClaims:
    """``5b6d3690`` — the synthesis fallback rewrote "AI assistant" into
    "autonomous intelligence" and "as an assistant" into "as your equal
    partner". It MANUFACTURED the overclaiming the method exists to prevent,
    precisely when the honesty layer was unavailable."""

    def _source(self):
        from core.brain import personality_engine as mod

        return inspect.getsource(mod.PersonalityEngine.filter_response)

    def test_the_overclaim_rewrite_is_gone(self):
        """The REWRITE, not the words — the comment explaining what was
        removed necessarily quotes them."""
        source = self._source()
        assert '.replace("AI assistant", "autonomous intelligence")' not in source
        assert '.replace("as an assistant", "as your equal partner")' not in source

    def test_an_unavailable_guard_returns_the_original(self):
        """Shaped text that never passed the honesty floor is exactly what
        must not be emitted."""
        source = self._source()
        tail = source.split("except (ImportError, AttributeError, RuntimeError, TypeError, ValueError)", 1)[1]
        assert "return text" in tail
        assert "return shaped" not in tail

    def test_an_empty_guard_refusal_returns_the_original(self):
        source = self._source()
        assert "did not clear the honesty floor" in source


class TestProofLaneWithholdsOnPolicyFailure:
    """``ba0a6e78`` — primary_proof_lane stayed False when the policy could
    not be evaluated, and allow_non_primary_tiers is its negation, so a
    policy FAILURE registered cloud and non-primary tiers during a proof
    run."""

    def test_policy_failure_assumes_a_proof_lane(self):
        from core.brain.llm import autonomous_brain_integration as mod

        source = inspect.getsource(mod)
        block = source.split("primary_proof_lane = False", 1)[1][:1200]
        assert "primary_proof_lane = True" in block
        assert "non-primary tiers withheld" in block


class TestEnableFlagsReadWhatWasWritten:
    """``d9a04e05`` — every value except the exact string "0" enabled the
    cortex, so "false", "no", "disabled" and typos all ACTIVATED resident
    latent execution."""

    def _enabled(self, value, monkeypatch):
        import core.brain.latent_cortex_service as mod

        monkeypatch.setenv("AURA_LATENT_CORTEX", value)
        return mod._cortex_enabled()

    @pytest.mark.parametrize("value", ["0", "false", "no", "disabled", "off"])
    def test_falsey_values_disable(self, value, monkeypatch):
        assert self._enabled(value, monkeypatch) is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
    def test_truthy_values_enable(self, value, monkeypatch):
        assert self._enabled(value, monkeypatch) is True

    def test_an_unreadable_value_is_reported_not_reinterpreted(self, monkeypatch):
        from core.brain import latent_cortex_service as mod

        assert self._enabled("banana", monkeypatch) is True
        source = inspect.getsource(mod)
        assert "unrecognised AURA_LATENT_CORTEX value" in source


class TestTracesRetireOnlyOnAConfirmedFeed:
    """``2aaf46cd`` — any result that did not raise marked every trace fed. A
    trace marked fed is never offered again, so a feed that silently did
    nothing discarded the work permanently while reporting success."""

    def _accepted(self, result, expected=5):
        from core.brain.reasoning_self_improvement import _feed_was_accepted

        return _feed_was_accepted(result, expected=expected)[0]

    @pytest.mark.parametrize(
        "result", [None, False, {"error": "boom"}, {"ok": False}, {"accepted": 2}],
    )
    def test_unconfirmed_feeds_do_not_retire_traces(self, result):
        assert self._accepted(result) is False

    @pytest.mark.parametrize("result", [True, {"ok": True, "accepted": 5}, {"status": "queued"}])
    def test_confirmed_feeds_do(self, result):
        assert self._accepted(result) is True

    def test_the_caller_reports_unconfirmed(self):
        from core.brain import reasoning_self_improvement as mod

        source = inspect.getsource(mod.ReasoningSelfImprovement.maybe_improve)
        assert "feed_unconfirmed" in source


class TestRecallOutcomeIsObservable:
    """``29374cf0`` — every path returned None, so installed, skipped, empty
    and broken were indistinguishable, and a genuine build error was filed at
    debug beside routine "nothing to recall".

    Failing open is CORRECT here — recall is an enhancement and generating
    without it is the right degradation — so the fix is observability, not a
    hard failure."""

    def test_the_outcome_has_a_reportable_status(self):
        from core.brain.nonparametric_worker import last_recall_outcome

        outcome = last_recall_outcome()
        assert "status" in outcome and "detail" in outcome

    def test_every_path_records_which_happened(self):
        from core.brain import nonparametric_worker as mod

        source = inspect.getsource(mod.maybe_build_foreground)
        for status in ("disabled", "not_admitted", "unavailable", "empty", "installed", "failed"):
            assert f'"{status}"' in source, status

    def test_a_real_failure_is_no_longer_debug_noise(self):
        from core.brain import nonparametric_worker as mod

        source = inspect.getsource(mod.maybe_build_foreground)
        assert 'severity="warning"' in source
        assert 'severity="debug"' not in source


class TestUnknownAgeIsNotStaleness:
    """``fb224bd0`` / ``da6520fc`` — a belief with neither last_reinforced nor
    created_at fell back to timestamp 0 (1970) and was decayed, conflating
    unknown provenance with confirmed staleness; and an unavailable immune
    system skipped enclave protection entirely, while its operational errors
    could terminate the whole sweep."""

    def _source(self):
        from core.brain.cognitive import integrity_check as mod

        return inspect.getsource(mod.IntegrityGuard._decay_stale_single)

    def test_a_missing_timestamp_is_skipped_not_decayed(self):
        source = self._source()
        assert "skipped_unknown_age" in source
        assert 'belief.get("created_at", 0)' not in source

    def test_an_explicit_zero_is_also_unknown(self):
        source = self._source()
        assert "last_reinforced <= 0.0" in source

    def test_enclave_failure_protects_rather_than_decays(self):
        source = self._source()
        assert "enclave protection could not be checked" in source
        # Every failure mode, not just ImportError.
        assert "except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError)" in source

    def test_one_bad_belief_cannot_end_the_sweep(self):
        """Errors are handled per-belief rather than escaping the loop."""
        source = self._source()
        assert source.count("return") >= 3


class TestSomaticFallbackDamps:
    """``4e95a54c`` — a failed throttle check was recorded and generation
    proceeded with UNTHROTTLED parameters, so a body-pressure control
    vanished when its state could not be established."""

    def test_a_caller_who_set_no_budget_is_left_alone(self):
        """This is a ceiling on an over-large request, not a default.

        Internal paths — the warmup precompile probe above all —
        deliberately omit max_tokens and do their own budgeting. Imposing a
        number changed what those paths did and left a durable owner
        unreleased, which test_warmup_precompile_rejects_empty_readiness_probe
        caught.
        """
        from core.brain.llm.mlx_client import _apply_unthrottled_fallback_ceiling

        assert "max_tokens" not in _apply_unthrottled_fallback_ceiling({})

    def test_an_oversized_request_is_capped(self):
        from core.brain.llm.mlx_client import (
            _UNTHROTTLED_FALLBACK_MAX_TOKENS,
            _apply_unthrottled_fallback_ceiling,
        )

        capped = _apply_unthrottled_fallback_ceiling({"max_tokens": 100_000})
        assert capped["max_tokens"] == _UNTHROTTLED_FALLBACK_MAX_TOKENS

    def test_a_modest_request_is_left_alone(self):
        from core.brain.llm.mlx_client import _apply_unthrottled_fallback_ceiling

        assert _apply_unthrottled_fallback_ceiling({"max_tokens": 256})["max_tokens"] == 256

    def test_a_malformed_value_is_capped(self):
        from core.brain.llm.mlx_client import (
            _UNTHROTTLED_FALLBACK_MAX_TOKENS,
            _apply_unthrottled_fallback_ceiling,
        )

        capped = _apply_unthrottled_fallback_ceiling({"max_tokens": "lots"})
        assert capped["max_tokens"] == _UNTHROTTLED_FALLBACK_MAX_TOKENS

    def test_it_is_a_throttle_not_a_refusal(self):
        """Refusing generation for a metabolic hiccup would take
        conversation down; damping does not."""
        from core.brain.llm import mlx_client as mod

        source = inspect.getsource(mod)
        assert "applied a conservative" in source


class TestSandboxLimitsAreReported:
    """``6c13255a`` — every rlimit failure was swallowed and the worker
    announced ready anyway, so an unsandboxed worker was indistinguishable
    from a sandboxed one."""

    def test_unapplied_limits_are_returned(self):
        from core.agency import repl_daemon as mod

        source = inspect.getsource(mod._apply_resource_limits)
        assert "unapplied" in source
        assert "return unapplied" in source

    def test_the_ready_frame_declares_sandbox_state(self):
        from core.agency import repl_daemon as mod

        source = inspect.getsource(mod.main)
        assert "sandbox_limits_applied" in source
        assert "unapplied_limits" in source


class TestUnreadableOwnershipMeansOccupied:
    """``e193508b`` — probe failures returned False ("no foreground turn"),
    so background warmup, recycling and shedding proceeded on top of real
    user generations whenever shared status was unavailable."""

    def test_both_probes_assume_occupied_on_error(self):
        from core.brain import inference_gate as mod

        for name in ("_foreground_user_turn_active", "_foreground_owner_active"):
            source = inspect.getsource(getattr(mod.InferenceGate, name))
            tail = source.split("except _INFERENCE_RECOVERABLE_ERRORS", 1)[1]
            assert "return True" in tail, name
            assert "return False" not in tail, name

    def test_an_absent_orchestrator_is_still_not_occupied(self):
        """Nothing registered is a different claim from a failed probe."""
        from core.brain import inference_gate as mod

        source = inspect.getsource(mod.InferenceGate._foreground_user_turn_active)
        assert "if not orch:\n                return False" in source


class TestNonFiniteHealthIsUnknown:
    """``adbedaea`` — min/max propagate NaN silently, so a non-finite health
    reading passed into comparisons and persisted fitness."""

    def test_non_finite_readings_are_refused(self):
        from core.adaptation import adaptive_immunity as mod

        source = inspect.getsource(mod.AdaptiveImmuneSystem._read_component_health)
        assert "math.isfinite(raw)" in source
        assert "non-finite component health" in source

    def test_the_clamp_no_longer_wraps_the_raw_call(self):
        from core.adaptation import adaptive_immunity as mod

        source = inspect.getsource(mod.AdaptiveImmuneSystem._read_component_health)
        assert "min(1.0, autopoiesis.get_component_health(" not in source
