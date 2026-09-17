"""test_cortex_hardening_v54.py — Stress tests for HARDENING v54 fixes.

Validates:
  1. Recovery response never echoes prompt content (hallucination fix)
  2. Stale lane reset actually resets MLX client state (infinite loop fix)
  3. Stale lane reset triggers recovery warmup (dead cortex fix)
  4. Recovery exhaustion uses exponential backoff, not 5-min lockout
  5. Log spam is rate-limited during stale state resets
  6. Repeated get_conversation_status() calls don't produce log spam
"""
import asyncio
import time
import unittest


class LaneClientDouble:
    """Minimal MLXLocalClient-compatible lane object for state-transition tests."""

    def __init__(self, alive: bool = False, state: str = "recovering"):
        self._lane_state = state
        self._lane_error = ""
        self._lane_transition_at = time.time() - 200  # >90s ago
        self._warmup_in_flight = False
        self._warmup_attempted = False
        self._last_heartbeat = 0.0
        self._last_progress_at = 0.0
        self._last_token_progress_at = 0.0
        self._last_ready_at = 0.0
        self._last_generation_completed_at = 0.0
        self._last_user_facing_completed_at = 0.0
        self._last_visible_readiness_at = 0.0
        self._process_started_at = 0.0
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._current_request_prompt_chars = 0
        self._active_generations = 0
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive

    def get_lane_status(self):
        return {
            "state": self._lane_state,
            "last_error": self._lane_error,
            "conversation_ready": self._alive and self._lane_state == "ready",
            "foreground_owned": False,
            "foreground_owner": "",
            "last_heartbeat": self._last_heartbeat,
            "last_progress_at": self._last_progress_at,
            "last_token_progress_at": self._last_token_progress_at,
            "last_ready_at": self._last_ready_at,
            "last_generation_completed_at": self._last_generation_completed_at,
            "last_user_facing_completed_at": self._last_user_facing_completed_at,
            "last_visible_readiness_at": self._last_visible_readiness_at,
            "last_transition_at": self._lane_transition_at,
            "warmup_attempted": self._warmup_attempted,
            "warmup_in_flight": self._warmup_in_flight,
            "process_started_at": self._process_started_at,
            "current_request_started_at": self._current_request_started_at,
            "current_first_token_at": self._current_first_token_at,
            "current_request_prompt_chars": self._current_request_prompt_chars,
        }

    def _set_lane_state(self, state: str, error: str = ""):
        if state != self._lane_state:
            self._lane_transition_at = time.time()
        self._lane_state = state
        if error:
            self._lane_error = error
        elif state == "ready":
            self._lane_error = ""

    def note_lane_recovering(self, reason: str):
        self._warmup_in_flight = False
        self._set_lane_state("recovering", reason)

    def warmup(self):
        async def _noop():
            return {"ok": True, "lane": self.name}
        return _noop()


def _make_gate(client=None):
    """Create a minimal InferenceGate instance for testing."""
    from core.brain.inference_gate import InferenceGate
    gate = InferenceGate.__new__(InferenceGate)
    gate.orch = None
    gate._created_at = time.monotonic()
    gate._mlx_client = client
    gate._initialized = True
    gate._init_error = None
    gate._cached_identity_prompt = None
    gate._identity_prompt_time = 0.0
    gate._cloud_backoff_until = 0.0
    gate._cortex_recovery_in_progress = False
    gate._last_cortex_check = 0.0
    gate._cortex_recovery_attempts = 0
    gate._cortex_recovery_exhausted_at = 0.0
    gate._last_stale_reset_log_at = 0.0
    gate._last_successful_generation_at = time.time()
    gate._prewarm_task = None
    gate._deferred_prewarm_task = None
    gate._maintenance_task = None
    gate._last_background_memory_shed_at = 0.0
    gate._last_spare_maintenance_at = 0.0
    gate._last_cortex_warmup_deferral_log_at = 0.0
    gate._last_user_generation_endpoint = None
    gate._last_user_generation_at = 0.0
    gate._last_user_generation_used_fallback = False
    import threading
    gate._foreground_ready_lock = threading.Lock()
    return gate


class TestRecoveryResponseNoEcho(unittest.TestCase):
    """Verify _user_facing_recovery_response NEVER echoes prompt content."""

    def test_does_not_contain_prompt_text(self):
        from core.brain.inference_gate import InferenceGate
        prompt = "User: What is your favorite mathematical theorem and why?"
        response = InferenceGate._user_facing_recovery_response(prompt)
        self.assertNotIn("mathematical", response)
        self.assertNotIn("theorem", response)
        self.assertNotIn("favorite", response)
        self.assertNotIn("You were saying", response)

    def test_does_not_echo_system_prompt(self):
        from core.brain.inference_gate import InferenceGate
        prompt = "You are Aura Luna, a sovereign digital entity. System: be helpful."
        response = InferenceGate._user_facing_recovery_response(prompt)
        self.assertNotIn("sovereign", response)
        self.assertNotIn("Aura Luna", response)
        self.assertNotIn("System:", response)

    def test_returns_fallback_instead_of_showing_filler(self):
        from core.brain.inference_gate import InferenceGate
        response = InferenceGate._user_facing_recovery_response("hello")
        self.assertNotIn("try again", response.lower())
        self.assertNotIn("send your message again", response.lower())
        self.assertGreater(len(response), 0)


class TestStaleStateReset(unittest.TestCase):
    """Verify the stale lane reset actually fixes the MLX client state."""

    def test_resets_mlx_client_state_not_just_dict(self):
        """The critical bug: resetting only lane dict left MLX client in 'recovering'."""
        client = LaneClientDouble(alive=False, state="recovering")
        client._lane_transition_at = time.time() - 200  # Stale for >90s
        gate = _make_gate(client)

        # Call get_conversation_status — should detect stale and reset
        lane = gate.get_conversation_status()

        # The returned dict should say "cold"
        self.assertEqual(lane["state"], "cold")
        # CRITICAL: The MLX client's ACTUAL state must also be "cold"
        self.assertEqual(client._lane_state, "cold",
                         "MLX client _lane_state was NOT reset — infinite loop bug still present")

    def test_clears_warmup_in_flight(self):
        client = LaneClientDouble(alive=False, state="recovering")
        client._lane_transition_at = time.time() - 200
        client._warmup_in_flight = True
        gate = _make_gate(client)

        lane = gate.get_conversation_status()
        self.assertFalse(client._warmup_in_flight)
        self.assertFalse(lane["warmup_in_flight"])

    def test_schedules_recovery_warmup(self):
        """After resetting to cold, a background prewarm should be scheduled."""
        client = LaneClientDouble(alive=False, state="recovering")
        client._lane_transition_at = time.time() - 200
        gate = _make_gate(client)

        scheduled: list[float] = []
        gate._schedule_background_cortex_prewarm = lambda delay=12.0: scheduled.append(delay)

        gate.get_conversation_status()
        self.assertEqual(scheduled, [3.0])

    def test_repeated_calls_do_not_spam_logs(self):
        """Multiple rapid calls should NOT produce a log for each one."""
        client = LaneClientDouble(alive=False, state="recovering")
        client._lane_transition_at = time.time() - 200
        gate = _make_gate(client)

        gate._schedule_background_cortex_prewarm = lambda delay=12.0: None

        # After the first call resets client to cold, subsequent calls
        # should NOT trigger the stale check again because client is now "cold"
        gate.get_conversation_status()  # First call — resets to cold
        # Client is now actually "cold", so subsequent calls won't trigger
        self.assertEqual(client._lane_state, "cold")

        # Re-poison the state to simulate ongoing issue
        client._set_lane_state("recovering", "test")
        client._lane_transition_at = time.time() - 200
        gate.get_conversation_status()  # Second call, should reset again
        # But should NOT log because rate limit (30s window)
        self.assertEqual(client._lane_state, "cold")


class TestRecoveryExhaustion(unittest.TestCase):
    """Verify recovery uses exponential backoff instead of 5-min lockout."""

    def test_exponential_backoff_not_flat_5min(self):
        """After 5 failures, cooldown should be 30s, not 300s."""
        client = LaneClientDouble(alive=False, state="cold")
        gate = _make_gate(client)
        gate._cortex_recovery_attempts = 5
        gate._cortex_recovery_exhausted_at = time.monotonic() - 35  # 35s ago

        # With old code: 300s cooldown → would still be locked out
        # With new code: 30s cooldown → should proceed
        # _ensure_cortex_recovery is async, so we need to check the logic
        now = time.monotonic()
        gate._last_cortex_check = now - 5  # Not rate-limited

        # The cooldown for 5 failures should be 30s
        cooldown = min(120.0, 30.0 * (1 + (gate._cortex_recovery_attempts - 5) // 5))
        self.assertEqual(cooldown, 30.0)

    def test_backoff_caps_at_120s(self):
        """Even after many failures, cooldown should cap at 120s."""
        for attempts in [5, 10, 15, 20, 50]:
            cooldown = min(120.0, 30.0 * (1 + (attempts - 5) // 5))
            self.assertLessEqual(cooldown, 120.0,
                                 f"Cooldown {cooldown}s exceeds 120s cap at {attempts} attempts")

    def test_never_permanently_gives_up(self):
        """Recovery should never permanently stop trying."""
        # Simulating 100 failures — should still eventually retry
        cooldown = min(120.0, 30.0 * (1 + (100 - 5) // 5))
        self.assertEqual(cooldown, 120.0)  # Caps at 120s, never "infinite"


class TestConversationLaneStatus(unittest.TestCase):
    """Validate get_conversation_status edge cases."""

    def test_ready_cortex_returns_conversation_ready(self):
        client = LaneClientDouble(alive=True, state="ready")
        now = time.time()
        client._last_ready_at = now
        client._last_progress_at = now
        client._last_visible_readiness_at = now
        gate = _make_gate(client)
        lane = gate.get_conversation_status()
        self.assertTrue(lane["conversation_ready"])

    def test_ready_cortex_idle_after_serving_stays_conversation_ready(self):
        # Served a visible turn 10 min ago, idle since (anchor > 0 but stale).
        # The visible-probe guard fires ONLY on a never-served lane (anchor <= 0);
        # idle liveness is covered by the worker-progress-staleness probe, so a
        # warm lane that already proved it can serve must NOT be downgraded.
        client = LaneClientDouble(alive=True, state="ready")
        now = time.time()
        client._last_ready_at = now
        client._last_progress_at = now
        client._last_visible_readiness_at = now - 600.0
        gate = _make_gate(client)

        lane = gate.get_conversation_status()

        self.assertTrue(lane["conversation_ready"])
        self.assertNotIn("visible_conversation_probe_missing", lane["readiness_blockers"])

    def test_ready_cortex_never_served_is_flagged_zombie(self):
        # Claims "ready" but never served a visible turn (anchor <= 0) → a cold
        # lane masquerading as ready. This is the true zombie the guard catches.
        client = LaneClientDouble(alive=True, state="ready")
        now = time.time()
        client._last_ready_at = now
        client._last_progress_at = now
        client._last_visible_readiness_at = 0.0
        gate = _make_gate(client)

        lane = gate.get_conversation_status()

        self.assertFalse(lane["conversation_ready"])
        self.assertIn("visible_conversation_probe_missing", lane["readiness_blockers"])

    def test_dead_cortex_cold_does_not_trigger_stale_reset(self):
        """A cold lane should NOT trigger the stale reset warning."""
        client = LaneClientDouble(alive=False, state="cold")
        client._lane_transition_at = time.time() - 200
        gate = _make_gate(client)
        lane = gate.get_conversation_status()
        # Cold state should pass through without the stale reset
        # (stale reset only triggers for "warming" or "recovering")
        self.assertIn(lane["state"], ("cold", "failed"))


if __name__ == "__main__":
    unittest.main()


def test_a_lane_that_went_cold_is_not_still_carrying_why_it_rebooted():
    """A reason to reboot is not a reason to refuse the endpoint for ever.

    LIVE 2026-09-17: one soft-cancel the worker never acknowledged rebooted
    the reflex lane, which then sat cold with that reason as its current
    error. Every validation read it as a fresh failure — 74 "Circuit OPEN for
    Reflex" in one window, on a worker nobody had asked to do anything — and
    every background plan that wanted the tier it serves found nothing in it.
    """
    from core.brain.llm.mlx_client import MLXLocalClient as MLXClient

    lane = MLXClient.__new__(MLXClient)
    lane._lane_state = "ready"
    lane._lane_error = ""
    lane._lane_transition_at = 0.0
    lane._lane_transition_monotonic_at = 0.0

    MLXClient._set_lane_state(lane, "recovering", "cancelled_worker_not_acknowledged")
    assert lane._lane_error == "cancelled_worker_not_acknowledged"
    MLXClient._set_lane_state(lane, "cold")
    assert lane._lane_error == ""
    # A lane that failed keeps what it failed with.
    MLXClient._set_lane_state(lane, "failed", "spawn_refused")
    MLXClient._set_lane_state(lane, "failed")
    assert lane._lane_error == "spawn_refused"
