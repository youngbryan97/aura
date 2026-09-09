"""CP126 mlx_client batch 3 — correlation, readiness, admission and tool-intent.

Each test pins one closed finding from artifacts/closeout/semantic_review/cp126/.
"""
from __future__ import annotations

import json

from core.brain.llm import mlx_client as mc
from core.brain.llm.latent_cortex.runtime_identity import latent_request_payload_sha256


class TestReadinessProof:
    """cdd743de + b6439433: readiness must be earned by an answered probe."""

    def test_expected_answer_accepted(self):
        assert mc._readiness_answer_accepted("ready") is True
        assert mc._readiness_answer_accepted("  Ready!  ") is True
        # A short latent/reasoning prefix must not fail a healthy lane.
        assert mc._readiness_answer_accepted("<think>ok</think> ready") is True

    def test_hallucinated_or_garbled_output_rejected(self):
        for bad in ("", "   ", "I am a large language model.", "!!!", "42"):
            assert mc._readiness_answer_accepted(bad) is False, bad

    def test_prompt_echo_is_not_an_answer(self):
        assert mc._readiness_answer_accepted("Reply exactly: ready") is False

    def test_unbounded_ramble_rejected(self):
        assert mc._readiness_answer_accepted("ready " * 200) is False

    def test_probe_runs_unconditionally_in_precompile(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient._run_warmup_precompile)
        # The visible probe must NOT be gated behind "precompile produced no
        # text" — a single token from max_tokens=1 is not conversation proof.
        assert "if not warmup_text or not str(warmup_text).strip():" not in source
        # And the probe's answer is judged, wherever the judging lives. It used
        # to be inline here and moved into `prove_visible_readiness` — "the one
        # place that means the lane has been seen to answer" — so this asserted
        # a call site rather than the property, and was red at HEAD.
        assert "prove_visible_readiness" in source
        judged = inspect.getsource(mc.MLXLocalClient.prove_visible_readiness)
        assert "_readiness_answer_accepted(" in judged


class TestResponseCorrelation:
    """49d694a1: an id-less terminal frame must never complete a request."""

    def test_ok_route_requires_matching_id(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient._response_listener_loop)
        assert "and (not req_id or req_id == self._current_request_id)" not in source
        assert "uncorrelated_worker_response" in source


class TestWarmupSingleflight:
    """4d8a7d6b: concurrent callers join; staleness uses the warmup's own clock."""

    def test_client_tracks_warmup_task_and_start(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient.warmup)
        assert "self._warmup_inflight" in source
        assert "asyncio.shield(inflight)" in source
        # Staleness must NOT be measured from the unrelated lane transition.
        assert "time.time() - self._lane_transition_at" not in source
        assert "_WARMUP_STALE_AFTER_S" in source

    def test_impl_no_longer_force_clears_on_lane_transition(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient._warmup_impl)
        assert "elapsed_since_transition" not in source


class TestBackgroundWarmupOrdering:
    """811cde6f: yield to foreground BEFORE touching worker lifecycle."""

    def test_defer_check_precedes_ensure_worker_alive(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient._warmup_impl)
        marker = source.index("background warmup")
        defer_idx = source.index("_foreground_owner_active()", marker)
        spawn_idx = source.index("_ensure_worker_alive(", marker)
        assert defer_idx < spawn_idx, "worker spawn happens before the yield decision"


class TestRebootAndCloseLockDiscipline:
    """ec341dfa + 97aa64fc: never mutate lifecycle state after losing the lock."""

    def test_reboot_defers_before_forcing(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient.reboot_worker)
        assert "Forcing reboot anyway to break deadlock" not in source
        assert "_REBOOT_LOCK_ESCALATED_WAIT_S" in source
        assert "reboot_deferred_lock" in source
        assert "_REBOOT_LOCK_FORCE_AFTER" in source

    def test_close_waits_and_receipts(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient.close)
        assert "_CLOSE_LOCK_WAIT_S" in source
        assert "close_lock_unavailable" in source

    def test_close_wait_is_longer_than_the_original_one_second(self):
        assert mc._CLOSE_LOCK_WAIT_S >= 5.0


class TestToolIntentEnvelope:
    """5a924075 + 0da5db2e: prose examples are not effect requests."""

    allowed = {"web_search", "read_file"}

    def _extract(self, text):
        return mc.MLXLocalClient._extract_tool_call_payload(text, allowed_tools=self.allowed)

    def test_whole_response_envelope_is_a_call(self):
        call = self._extract('{"tool": "web_search", "args": {"q": "aura"}}')
        assert call == {"tool": "web_search", "args": {"q": "aura"}}

    def test_fenced_whole_response_is_a_call(self):
        call = self._extract('```json\n{"tool": "web_search", "args": {}}\n```')
        assert call == {"tool": "web_search", "args": {}}

    def test_native_channel_is_trusted_anywhere(self):
        call = self._extract(
            'Let me look that up.\n<tool_call>{"name": "web_search", '
            '"arguments": {"q": "x"}}</tool_call>'
        )
        assert call == {"tool": "web_search", "args": {"q": "x"}}

    def test_example_inside_prose_is_not_a_call(self):
        text = (
            "To search the web you would emit something like this:\n"
            '```json\n{"tool": "web_search", "args": {"q": "example"}}\n```\n'
            "But I already know the answer, so I will not call it."
        )
        assert self._extract(text) is None

    def test_unadvertised_tool_rejected(self):
        assert self._extract('{"tool": "rm_rf", "args": {}}') is None

    def test_unparseable_string_arguments_are_not_invented(self):
        # The old parser wrapped this as {"value": "not json"}.
        assert self._extract('{"name": "web_search", "arguments": "not json"}') is None

    def test_non_dict_arguments_rejected(self):
        assert self._extract('{"tool": "web_search", "args": [1, 2]}') is None


class TestToolArgumentSchema:
    """0da5db2e: arguments are bound to the tool's advertised schema."""

    schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "limit": {"type": "integer"},
                "mode": {"type": "string", "enum": ["fast", "deep"]},
            },
            "required": ["q"],
            "additionalProperties": False,
        }
    }

    def test_valid_arguments_pass(self):
        assert mc._tool_arguments_schema_error(self.schema, {"q": "hi", "limit": 3}) == ""

    def test_missing_required_rejected(self):
        assert "missing required" in mc._tool_arguments_schema_error(self.schema, {"limit": 1})

    def test_wrong_type_rejected(self):
        assert "must be integer" in mc._tool_arguments_schema_error(self.schema, {"q": "a", "limit": "x"})

    def test_bool_is_not_an_integer(self):
        assert "must be integer" in mc._tool_arguments_schema_error(self.schema, {"q": "a", "limit": True})

    def test_enum_enforced(self):
        assert "allowed values" in mc._tool_arguments_schema_error(
            self.schema, {"q": "a", "mode": "sideways"}
        )

    def test_unexpected_property_rejected(self):
        assert "unexpected" in mc._tool_arguments_schema_error(self.schema, {"q": "a", "evil": 1})

    def test_non_dict_rejected(self):
        assert "JSON object" in mc._tool_arguments_schema_error(self.schema, ["q"])

    def test_depth_and_size_bounded(self):
        deep: dict = {"q": "a"}
        node = deep
        for _ in range(12):
            node["n"] = {}
            node = node["n"]
        assert mc._tool_arguments_schema_error({}, deep) != ""
        assert "too large" in mc._tool_arguments_schema_error({}, {"q": "x" * 30_000})

    def test_no_advertised_schema_still_bounds_shape(self):
        assert mc._tool_arguments_schema_error(None, {"anything": 1}) == ""
        assert mc._tool_arguments_schema_error(None, "nope") != ""


class TestToolResultTruncation:
    """abd93abf: truncation must not hand the model broken JSON."""

    def test_short_result_untouched(self):
        assert mc._truncate_tool_result("small") == "small"

    def test_json_result_stays_parseable(self):
        payload = json.dumps({"rows": [{"i": i, "text": "x" * 40} for i in range(400)]})
        out = mc._truncate_tool_result(payload, limit=2000)
        parsed = json.loads(out)  # must not raise
        assert parsed["truncated"] is True
        assert parsed["original_chars"] == len(payload)
        assert len(out) <= 2000

    def test_json_result_budget_accounts_for_escaping(self):
        payload = json.dumps({"value": "\0" * 5000})
        out = mc._truncate_tool_result(payload, limit=1000)

        assert json.loads(out)["truncated"] is True
        assert len(out) <= 1000

    def test_plain_text_marked_truncated(self):
        out = mc._truncate_tool_result("y" * 9000, limit=1000)
        assert "TRUNCATED" in out
        assert len(out) <= 1000


class TestLatentRequestIdentity:
    """9721b1be: verifier controls are part of the request identity."""

    def _digest(self, **extra):
        return latent_request_payload_sha256(
            prompt="p",
            messages=None,
            domain="general",
            config=None,
            budget=None,
            runtime_controls=None,
            **extra,
        )

    def test_verifier_guidance_changes_identity(self):
        assert self._digest() != self._digest(verifier_guidance=True)

    def test_facet_reliability_changes_identity(self):
        a = self._digest(verifier_guidance=True, facet_reliability={"f": 0.9})
        b = self._digest(verifier_guidance=True, facet_reliability={"f": 0.1})
        assert a != b

    def test_absent_controls_are_backward_compatible(self):
        # Episodes that pass neither must hash exactly as they always did.
        assert self._digest() == self._digest(verifier_guidance=None, facet_reliability=None)


class TestProofPolicyFailsClosed:
    """84a18b06: enforcement infrastructure missing must not disable enforcement."""

    def test_proof_origin_detected(self):
        assert mc._proof_run_requested("proof_lane") is True
        assert mc._proof_run_requested("chat") is False

    def test_env_signals_detected(self, monkeypatch):
        monkeypatch.setenv("AURA_PROOF_RUN", "1")
        assert mc._proof_run_requested("chat") is True

    def test_disabled_env_values_are_not_proof_runs(self, monkeypatch):
        monkeypatch.setenv("AURA_PROOF_RUN", "0")
        monkeypatch.delenv("AURA_PROOF_MODEL_TIER", raising=False)
        monkeypatch.delenv("AURA_PROOF_HEADLESS", raising=False)
        assert mc._proof_run_requested("chat") is False


class TestEnqueueDeadline:
    """a838a49b: never queue work whose deadline already expired."""

    def test_expired_budget_is_refused_not_floored(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient._generate_inner)
        # Assert the EXECUTABLE assignment is gone (the fix comment quotes
        # the old expression deliberately, as the record of what broke).
        assert "enqueue_timeout = max(0.5" not in source
        assert "request_deadline_expired_before_enqueue" in source


class TestDurableOwnerRelease:
    """158ed09e: an unconfirmed release is not an absent owner."""

    def test_unconfirmed_release_reports_failure(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient._release_durable_model_lane_owner_sync)
        # The unregister/clear path must be reachable only after confirmation.
        confirm_idx = source.index("if not released:")
        unregister_idx = source.index("unregister_model_lane_owner_adapter(owner_id)")
        assert confirm_idx < unregister_idx
        assert "return False" in source
        assert "_note_lane_release_failure" in source


class TestLatentAnswerContract:
    """d78cbfa4: ok requires a real nonempty string answer."""

    def test_source_rejects_non_string_answers(self):
        import inspect

        source = inspect.getsource(mc.MLXLocalClient.latent_reason_async)
        assert 'str(res.get("text") or "")' not in source
        assert "latent_answer_invalid" in source


class TestLatentLoopClosures:
    """Progress and cancellation must retain their request iteration."""

    def test_nested_callbacks_bind_loop_values(self):
        import inspect

        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker._mlx_worker_loop)
        assert "_request_id: str = request_id" in source
        assert '"id": _request_id' in source
        assert "lambda _job_seq=job_seq: soft_cancel_requested(" in source
        assert "_job_seq," in source


class TestStrictValueIsNotLaundered:
    """7f86d404 (mlx_worker): an incorrect draft must not be replaced with
    the expected answer."""

    def test_exact_value_matches(self):
        from core.brain.llm.mlx_worker import _matches_expected_strict_value_prefix

        assert _matches_expected_strict_value_prefix("ok", "ok") is True

    def test_separator_suffix_still_matches(self):
        from core.brain.llm.mlx_worker import _matches_expected_strict_value_prefix

        for text in ("ok.", "ok ", "ok\n", "ok, then", "ok)"):
            assert _matches_expected_strict_value_prefix(text, "ok") is True, text

    def test_abutting_uppercase_is_not_a_match(self):
        from core.brain.llm.mlx_worker import _matches_expected_strict_value_prefix

        # "okInjected" is a DIFFERENT token from "ok" — crediting it graded
        # answer seeding rather than model merit.
        assert _matches_expected_strict_value_prefix("okInjected", "ok") is False
        assert _matches_expected_strict_value_prefix("okI output the value", "ok") is False

    def test_normalization_no_longer_returns_the_expected_answer(self):
        from core.brain.llm.mlx_worker import _normalize_strict_value_response

        out = _normalize_strict_value_response("okInjected", expected_value="ok")
        assert out != "ok", "an incorrect draft was laundered into the expected value"

    def test_genuine_answer_still_normalizes(self):
        from core.brain.llm.mlx_worker import _normalize_strict_value_response

        assert _normalize_strict_value_response("ok", expected_value="ok") == "ok"
        assert _normalize_strict_value_response("  ok.  ", expected_value="ok") == "ok"
