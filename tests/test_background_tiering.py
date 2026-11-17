import logging
import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
)
from core.container import ServiceContainer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Aura.Test")


@contextmanager
def _safe_boot_disabled():
    """Scope AURA_SAFE_BOOT_DESKTOP to this block.

    The restore is in a finally, so it is correct — but the flag it touches
    steers sensory lane selection process-wide, and a neighbouring test that
    reads it inherits whatever is set at that moment. Documented here because
    the coupling is invisible from either side: the leak this guards against
    surfaced as a boot-sensory test asserting an empty degraded map.
    """
    previous = os.environ.get("AURA_SAFE_BOOT_DESKTOP")
    os.environ["AURA_SAFE_BOOT_DESKTOP"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AURA_SAFE_BOOT_DESKTOP", None)
        else:
            os.environ["AURA_SAFE_BOOT_DESKTOP"] = previous


@contextmanager
def _service_get_overrides(**services):
    original_get = ServiceContainer.__dict__["get"]
    original_peek = ServiceContainer.__dict__["peek"]
    ServiceContainer.get = classmethod(
        lambda cls, name, default=None: services.get(name, default)
    )
    ServiceContainer.peek = classmethod(
        lambda cls, name, default=None: services.get(name, default)
    )
    try:
        yield
    finally:
        ServiceContainer.get = original_get
        ServiceContainer.peek = original_peek


class EndpointCallRecorder:
    def __init__(self, text: str = "deterministic response"):
        self.text = text
        self.calls = []

    async def __call__(self, endpoint, *args, **kwargs):
        self.calls.append((endpoint, args, kwargs))
        return {"ok": True, "text": self.text}

    @property
    def last_endpoint(self):
        assert self.calls, "endpoint recorder was not called"
        return self.calls[-1][0]


class TestBackgroundTiering(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from core.brain.llm_health_router import HealthAwareLLMRouter
        self.router = HealthAwareLLMRouter()
        self.router.logger = logger
        self.router.register(PRIMARY_ENDPOINT, "internal", "32B", is_local=True, tier="local")
        self.router.register(DEEP_ENDPOINT, "internal", "72B", is_local=True, tier="local_deep")
        # These used to register two cloud endpoints. Remote model providers
        # are gone, and the router refuses them at registration — so the
        # fixture was building a router that could not exist and every test
        # below it errored before reaching its own subject. Background
        # tiering is about which LOCAL tier a background turn lands on, and
        # nothing here needed a remote endpoint to say so.
        self.router.register(BRAINSTEM_ENDPOINT, "internal", "7B-Local", is_local=True, tier="local_fast")
        self.router.register(FALLBACK_ENDPOINT, "internal", "7B-Fallback", is_local=True, tier="local_fast")
        self.endpoint_calls = EndpointCallRecorder()
        self.router._call_endpoint = self.endpoint_calls

    async def test_automatic_background_tiering_by_flag(self):
        """Verify that is_background=True forces the tertiary tier."""
        with _safe_boot_disabled():
            await self.router.think("Hello", is_background=True)

        # Background routing should use the local 7B brainstem first.
        called_ep = self.endpoint_calls.last_endpoint
        self.assertEqual(called_ep.name, BRAINSTEM_ENDPOINT)
        self.assertEqual(called_ep.tier, "local_fast")

    async def test_automatic_background_tiering_by_origin(self):
        """Verify that origin='metabolic' forces the tertiary tier."""
        with _safe_boot_disabled():
            await self.router.think("Hello", origin="metabolic_cycle")

        called_ep = self.endpoint_calls.last_endpoint
        self.assertEqual(called_ep.name, BRAINSTEM_ENDPOINT)

    async def test_background_override_is_demoted(self):
        """Background tasks must stay on the 7B path even if they request primary."""
        with _safe_boot_disabled():
            await self.router.think("Hello", prefer_tier="primary", is_background=True)

        called_ep = self.endpoint_calls.last_endpoint
        self.assertEqual(called_ep.name, BRAINSTEM_ENDPOINT)

    async def test_originless_primary_request_is_background_unless_purpose_is_user_facing(self):
        """Internal callers must not become foreground just by requesting primary."""
        with _safe_boot_disabled():
            await self.router.think("quiet internal reflection", prefer_tier="primary")

        called_ep = self.endpoint_calls.last_endpoint
        self.assertEqual(called_ep.name, BRAINSTEM_ENDPOINT)

    async def test_background_inference_is_suppressed_while_foreground_user_turn_is_active(self):
        """Background jobs should back off instead of contending with an active user reply."""
        orch = SimpleNamespace(
            status=SimpleNamespace(is_processing=True),
            _current_origin="api",
            _current_task_is_autonomous=False,
            _foreground_user_quiet_until=0.0,
        )

        with _service_get_overrides(orchestrator=orch):
            result = await self.router.think("Hello", origin="sovereign_pruner", is_background=True)

        self.assertIsNone(result)
        self.assertEqual(self.endpoint_calls.calls, [])

    async def test_background_inference_is_suppressed_during_quiet_window(self):
        """Background jobs should also back off immediately after a user-facing turn completes."""
        orch = SimpleNamespace(
            status=SimpleNamespace(is_processing=False),
            _current_origin="",
            _current_task_is_autonomous=False,
            _foreground_user_quiet_until=9999999999.0,
        )

        with _service_get_overrides(orchestrator=orch):
            result = await self.router.think("Hello", origin="sovereign_pruner", is_background=True)

        self.assertIsNone(result)
        self.assertEqual(self.endpoint_calls.calls, [])

    def test_background_policy_blocks_while_foreground_generation_is_active(self):
        from core.runtime.background_policy import background_activity_reason

        gate = SimpleNamespace(
            get_conversation_status=lambda: {
                "foreground_owned": True,
                "active_generations": 1,
                "kernel_lock_held": True,
            }
        )
        orch = SimpleNamespace(
            is_busy=False,
            _suppress_unsolicited_proactivity_until=0.0,
            _foreground_user_quiet_until=0.0,
            _last_user_interaction_time=0.0,
        )

        with _service_get_overrides(inference_gate=gate):
            reason = background_activity_reason(orch, allow_no_user_anchor=True)

        self.assertEqual(reason, "foreground_generation_active")

    def test_work_that_does_not_use_her_model_is_held_by_a_persons_turn_not_by_her_models_lane(self):
        """LIVE 17 November 2026: her repair loop, which works on her Python and tests, waited fourteen minutes as
        "foreground_generation_active" behind her curriculum loop's background generations, nobody talking to her."""
        from unittest import mock

        from core.runtime import foreground_guard
        from core.runtime.background_policy import background_activity_reason

        gate = SimpleNamespace(get_conversation_status=lambda: {"foreground_owned": False, "active_generations": 1})
        orch = SimpleNamespace(is_busy=False, _suppress_unsolicited_proactivity_until=0.0,
                               _foreground_user_quiet_until=0.0, _last_user_interaction_time=0.0)

        with _service_get_overrides(inference_gate=gate):
            self.assertEqual(background_activity_reason(orch, allow_no_user_anchor=True),
                             "foreground_generation_active")
            self.assertNotIn("foreground", background_activity_reason(orch, allow_no_user_anchor=True,
                                                                      needs_the_model=False))
            with mock.patch.object(foreground_guard, "foreground_activity_reason",
                                   lambda: "foreground_chat_active"):
                self.assertEqual(background_activity_reason(orch, allow_no_user_anchor=True, needs_the_model=False),
                                 "foreground_chat_active")

if __name__ == "__main__":
    unittest.main()
