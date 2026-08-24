"""Contracts for declarative model-lane memory admission (roadmap K3).

The over-commitment doom loop (stall → force-kill → cold reload) came from
lanes spawning against instantaneous free-RAM spot checks. These tests pin
the declarative model: declared footprints vs an explicit host budget, QoS
eviction order, the envelope-breach refusal, and the mlx_client spawn seam.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from core.brain.lane_admission import (
    ActiveLane,
    LaneAdmissionController,
    QoSClass,
    classify_lane,
    get_lane_admission_controller,
    lane_budget_gb,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def controller():
    return LaneAdmissionController()


@pytest.fixture
def budget_46(monkeypatch):
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "46")
    return 46.0


class TestLaneClassification:
    def test_primary_cortex_is_guaranteed_by_role(self):
        assert classify_lane("/models/renamed-artifact", role="cortex") == (
            "cortex",
            QoSClass.GUARANTEED,
        )

    @pytest.mark.parametrize("role", ["solver", "brainstem", "reflex"])
    def test_nonresident_serving_roles_are_burstable(self, role):
        assert classify_lane("/models/renamed-artifact", role=role) == (
            role,
            QoSClass.BURSTABLE,
        )

    def test_trainers_are_best_effort_regardless_of_size(self):
        assert classify_lane("/models/Aura-32B-4bit", purpose="train") == (
            "trainer",
            QoSClass.BEST_EFFORT,
        )

    def test_unknown_paths_are_best_effort_auxiliary(self):
        assert classify_lane("/models/whisper-large") == ("auxiliary", QoSClass.BEST_EFFORT)

    def test_size_and_role_words_cannot_self_assign_authority(self):
        assert classify_lane("/models/Deep-72B-Solver") == (
            "auxiliary",
            QoSClass.BEST_EFFORT,
        )


class TestBudget:
    def test_absolute_override_wins(self, monkeypatch):
        monkeypatch.setenv("AURA_LANE_BUDGET_GB", "40")
        assert lane_budget_gb() == 40.0

    def test_fraction_is_clamped(self, monkeypatch):
        monkeypatch.delenv("AURA_LANE_BUDGET_GB", raising=False)
        monkeypatch.setenv("AURA_LANE_BUDGET_FRACTION", "0.05")
        # clamped to 0.30 of host — never a sliver budget
        assert lane_budget_gb() > 0.0

    def test_desktop_lane_budget_matches_the_process_envelope(self, monkeypatch):
        from core.brain import lane_admission

        monkeypatch.delenv("AURA_LANE_BUDGET_GB", raising=False)
        monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
        monkeypatch.setenv("AURA_PROCESS_RSS_LIMIT_GB", "auto")
        monkeypatch.setattr(
            lane_admission,
            "_host_total_gb",
            lambda _observer=None: (64.0, True),
        )

        assert lane_budget_gb() == pytest.approx(51.84)


class TestAdmissionArithmetic:
    def test_fits_admits_cleanly(self, controller, budget_46):
        decision = controller.admit(
            model_path="/models/resident-renamed",
            request_gb=23.0,
            active=[],
            role="cortex",
        )
        assert decision.admitted and decision.reason == "fits"
        assert decision.evict_first == ()

    def test_committed_lanes_count_against_budget(self, controller, budget_46):
        active = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0),
            ActiveLane("brainstem", QoSClass.BURSTABLE, 5.0),
        ]
        decision = controller.admit(
            model_path="/models/emergency-renamed",
            request_gb=2.0,
            active=active,
            role="reflex",
        )
        assert decision.admitted
        assert decision.committed_gb == pytest.approx(25.0)

    def test_guaranteed_candidate_gets_yield_advisory(self, controller, budget_46):
        """Cortex coming up over a loaded solver: admit, advise the solver yields."""
        active = [ActiveLane("solver", QoSClass.BURSTABLE, 41.0, model_path="/m/deep-72b")]
        decision = controller.admit(
            model_path="/models/resident-renamed",
            request_gb=23.0,
            active=active,
            role="cortex",
        )
        assert decision.admitted and decision.reason == "fits_after_yield"
        assert decision.evict_first == ("/m/deep-72b",)

    def test_guaranteed_candidate_ignores_the_user_facing_shield(
        self, controller, budget_46
    ):
        """The cortex must ALWAYS be able to come up — even over a lane that
        served the user seconds ago."""
        active = [
            ActiveLane(
                "solver",
                QoSClass.BURSTABLE,
                41.0,
                model_path="/m/deep-72b",
                last_user_facing_age_s=10.0,
            )
        ]
        decision = controller.admit(
            model_path="/models/resident-renamed",
            request_gb=23.0,
            active=active,
            role="cortex",
        )
        assert decision.admitted and decision.evict_first == ("/m/deep-72b",)

    def test_burstable_candidate_respects_the_shield(self, controller, budget_46):
        """A background solver must NOT be told it may evict a reflex lane
        that just served the user."""
        active = [
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
            ActiveLane(
                "reflex",
                QoSClass.BURSTABLE,
                2.0,
                model_path="/m/reflex",
                last_user_facing_age_s=5.0,
            ),
        ]
        decision = controller.admit(
            model_path="/models/specialist-renamed",
            request_gb=41.0,
            active=active,
            role="solver",
        )
        assert not decision.admitted
        assert "lane_budget_exceeded" in decision.reason
        # cortex is GUARANTEED (higher QoS) and reflex is shielded: no advisories
        assert decision.evict_first == ()

    def test_explicit_disruptive_handoff_can_replace_guaranteed_lane(
        self, controller, budget_46
    ):
        active = [
            ActiveLane(
                "cortex",
                QoSClass.GUARANTEED,
                38.0,
                model_path="/m/cortex",
                last_user_facing_age_s=1.0,
            )
        ]

        decision = controller.admit(
            model_path="/models/specialist-renamed",
            request_gb=46.0,
            active=active,
            allow_disruptive_eviction=True,
            role="solver",
        )

        assert decision.admitted
        assert decision.reason == "fits_after_yield"
        assert decision.evict_first == ("/m/cortex",)

    def test_envelope_breach_names_the_arithmetic(self, controller, budget_46):
        """The 72B over a committed host: the refusal that replaces the
        OOM-SIGKILL-with-empty-stderr death."""
        active = [ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex")]
        decision = controller.admit(
            model_path="/models/specialist-renamed",
            request_gb=41.0,
            active=active,
            role="solver",
        )
        assert not decision.admitted
        assert "request 41.0GB" in decision.reason
        assert "budget 46.0GB" in decision.reason

    def test_best_effort_evicted_before_burstable(self, controller, budget_46):
        active = [
            ActiveLane("trainer", QoSClass.BEST_EFFORT, 8.0, model_path="/m/trainer"),
            ActiveLane("brainstem", QoSClass.BURSTABLE, 5.0, model_path="/m/brainstem"),
            ActiveLane("cortex", QoSClass.GUARANTEED, 20.0, model_path="/m/cortex"),
        ]
        # solver needs 20GB of room in a 46 budget: 33 committed + 20 = 53 > 46;
        # evicting the 8GB trainer alone brings it to 45 <= 46.
        decision = controller.admit(
            model_path="/models/specialist-renamed",
            request_gb=20.0,
            active=active,
            role="solver",
        )
        assert decision.admitted and decision.reason == "fits_after_yield"
        assert decision.evict_first == ("/m/trainer",)

    def test_advise_mode_never_enforces(self, controller, budget_46, monkeypatch):
        monkeypatch.setenv("AURA_LANE_ADMISSION", "advise")
        active = [ActiveLane("cortex", QoSClass.GUARANTEED, 20.0)]
        decision = controller.admit(
            model_path="/models/specialist-renamed",
            request_gb=41.0,
            active=active,
            role="solver",
        )
        assert not decision.admitted
        assert decision.enforced is False


class TestObservability:
    def test_snapshot_carries_recent_decisions(self, controller, budget_46):
        controller.admit(model_path="/m/qwen-7b", request_gb=5.0, active=[])
        snap = controller.snapshot()
        assert snap["alive"] is True
        assert snap["ready"] is True
        assert snap["budget_gb"] == 46.0
        assert snap["mode"] in {"enforce", "advise"}
        assert snap["recent_decisions"][-1]["admitted"] is True
        assert controller.is_alive() is True
        assert controller.is_ready() is True
        assert controller.get_status() == snap

    def test_singleton_accessor(self):
        assert get_lane_admission_controller() is get_lane_admission_controller()


class TestSpawnSeam:
    """The mlx_client integration: observed lanes + the spawn consult."""

    class _FakeClient:
        def __init__(self, model_path, alive=True, last_user_facing=0.0):
            self.model_path = model_path
            self._alive = alive
            self._last_user_facing_completed_at = last_user_facing

        def is_alive(self):
            return self._alive

    def test_model_load_admission_timeouts_use_typed_flags(self, monkeypatch):
        from core.brain.llm import mlx_client as mc

        monkeypatch.setenv("AURA_FOREGROUND_MODEL_LOAD_ADMISSION_TIMEOUT_S", "12.5")
        monkeypatch.setenv("AURA_BACKGROUND_MODEL_LOAD_ADMISSION_TIMEOUT_S", "invalid")

        assert mc._model_load_admission_timeout_s(foreground_request=True) == 12.5
        assert mc._model_load_admission_timeout_s(foreground_request=False) == 0.0

    def test_observed_lanes_exclude_self_and_dead(self, monkeypatch, budget_46):
        from core.brain.llm import mlx_client as mc
        from core.brain.llm import model_registry

        me = self._FakeClient("/m/Aura-32B-cortex")
        other = self._FakeClient("/m/qwen-7b")
        dead = self._FakeClient("/m/Deep-72B", alive=False)
        monkeypatch.setattr(
            mc, "_CLIENTS", {c.model_path: c for c in (me, other, dead)}
        )
        monkeypatch.setattr(
            model_registry,
            "get_model_lane_role",
            lambda path: "brainstem" if path == other.model_path else None,
        )
        lanes = mc._observed_active_lanes(exclude_client=me)
        assert [lane.lane for lane in lanes] == ["brainstem"]

    def test_runtime_overhead_excludes_observed_model_worker_memory(self, monkeypatch):
        from core.brain.llm import mlx_client as mc

        owner = SimpleNamespace(observed_gb=9.0)
        monkeypatch.setattr(
            mc,
            "get_memory_pressure_snapshot",
            lambda: SimpleNamespace(process_rss_gb=31.7),
        )

        assert mc._transient_runtime_footprint_gb([owner]) == pytest.approx(22.7)

    @pytest.mark.asyncio
    async def test_model_load_context_holds_and_releases_canonical_lease(
        self,
        monkeypatch,
        tmp_path,
    ):
        from core.brain.llm import mlx_client as mc
        from core.runtime import control_plane, model_lane_control
        from core.runtime.control_plane import (
            PressureSnapshot,
            ResourceAdmissionController,
            WorkClass,
        )
        from core.runtime.model_lane_control import ModelLaneController
        from core.runtime.receipts import ReceiptStore

        controller = ResourceAdmissionController(
            pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
            receipt_store=ReceiptStore(tmp_path / "receipts"),
        )
        monkeypatch.setattr(
            control_plane,
            "get_runtime_control_plane",
            lambda: SimpleNamespace(admission=controller),
        )
        lane_controller = ModelLaneController(
            state_path=tmp_path / "model_lanes.json",
            receipt_store=ReceiptStore(tmp_path / "lane_receipts"),
            process_discovery=None,
        )
        monkeypatch.setattr(
            model_lane_control,
            "get_model_lane_controller",
            lambda: lane_controller,
        )
        monkeypatch.setattr(mc, "_CLIENTS", {})

        candidate = self._FakeClient("/m/Aura-32B-cortex")
        candidate._warmup_timeout = lambda: 60.0
        candidate._handshake_timeout = lambda: 300.0
        candidate._process = SimpleNamespace(
            pid=os.getpid(),
            name="test-worker",
            is_alive=lambda: True,
        )
        candidate._init_done = True
        async with mc._model_load_admission_context(
            candidate,
            foreground_request=True,
        ) as decision:
            assert decision.admitted is True
            assert controller.active_lease_count(WorkClass.MODEL_LOAD) == 1
            lease = controller.status()["active_leases"][0]
            assert lease["ttl_remaining_s"] >= 419.0

        assert controller.active_lease_count(WorkClass.MODEL_LOAD) == 0
        history = controller.status()["history"]
        assert [entry["outcome"] for entry in history[-2:]] == [
            "admitted",
            "released",
        ]
        lane_snapshot = lane_controller.snapshot()
        assert lane_snapshot["reserved_gb"] == 0.0
        assert len(lane_snapshot["owners"]) == 1
        assert lane_snapshot["owners"][0]["process"]["pid"] == os.getpid()
        assert candidate._model_lane_fencing_token > 0
        assert candidate._model_lane_terminal_receipt_id
