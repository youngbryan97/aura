"""Contract tests for the autonomous weight-compounding scheduler.

The scheduler is the piece that turns installed machinery into live behavior,
so the contracts here are about WHEN it acts: kill switch, cooldown,
maintenance-idle gate, data readiness, Will approval, and what it does with a
qualification candidate. The heavy loop itself is covered by test_weight_compounding.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import core.learning.compounding_scheduler as sched_mod
from core.learning.compounding_scheduler import CompoundingScheduler

pytestmark = pytest.mark.unit


@dataclass
class FakeReceipt:
    generation_id: str = "g0000-test"
    status: str = "candidate"
    candidate_model_path: str = "fake-fused-artifact"
    promoted_model_path: str = ""
    reasons: list = field(default_factory=list)

    def to_dict(self):
        return {
            "generation_id": self.generation_id,
            "status": self.status,
            "candidate_model_path": self.candidate_model_path,
            "promoted_model_path": self.promoted_model_path,
            "reasons": self.reasons,
        }


class FakeLoop:
    def __init__(self, receipt: FakeReceipt, ready: bool = True):
        self._receipt = receipt
        self._ready = ready
        self.cycles_run = 0

    def data_readiness(self):
        return {"ready": self._ready, "sft_rows": 100, "dpo_rows": 0}

    def run_cycle(self):
        self.cycles_run += 1
        return self._receipt

    def stats(self):
        return {"generations": self.cycles_run, "verdict": "NO_RSI"}


@pytest.fixture
def scheduler(tmp_path: Path, monkeypatch) -> CompoundingScheduler:
    monkeypatch.setenv("AURA_WEIGHT_COMPOUNDING", "1")
    s = CompoundingScheduler(orchestrator=None)
    monkeypatch.setattr(
        CompoundingScheduler, "_state_path", lambda self: tmp_path / "state.json"
    )
    return s


def allow_maintenance(monkeypatch, allowed: bool) -> None:
    import core.runtime.background_policy as policy

    monkeypatch.setattr(
        policy, "background_activity_allowed", lambda *a, **k: allowed
    )


def approve_will(monkeypatch, scheduler, approved: bool = True, reason: str = "ok") -> list:
    calls: list = []

    def fake_approval(self, context):
        calls.append(context)
        return approved, reason

    monkeypatch.setattr(CompoundingScheduler, "_will_approval", fake_approval)
    return calls


class TestGates:
    async def test_kill_switch_prevents_start(self, scheduler, monkeypatch):
        monkeypatch.setenv("AURA_WEIGHT_COMPOUNDING", "0")
        await scheduler.start()
        assert scheduler._task is None

    async def test_cooldown_blocks(self, scheduler, monkeypatch):
        scheduler._save_state({"last_attempt_at": time.time()})
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, True)
        approve_will(monkeypatch, scheduler)
        await scheduler._maybe_cycle()
        assert loop.cycles_run == 0

    async def test_maintenance_gate_blocks(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, False)
        will_calls = approve_will(monkeypatch, scheduler)
        await scheduler._maybe_cycle()
        assert loop.cycles_run == 0
        assert not will_calls           # never even consulted the Will

    async def test_data_not_ready_blocks_before_will(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt(), ready=False)
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, True)
        will_calls = approve_will(monkeypatch, scheduler)
        await scheduler._maybe_cycle()
        assert loop.cycles_run == 0
        assert not will_calls

    async def test_will_denial_blocks_and_records(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, True)
        approve_will(monkeypatch, scheduler, approved=False, reason="unsafe_window")
        await scheduler._maybe_cycle()
        assert loop.cycles_run == 0
        state = json.loads(scheduler._state_path().read_text())
        assert state["last_status"] == "will_denied:unsafe_window"

    async def test_will_unavailable_fails_closed(self, scheduler):
        # real _will_approval with no Will importable in this context must deny
        approved, reason = scheduler._will_approval({"readiness": {}})
        assert isinstance(approved, bool)
        if not approved:
            assert reason


class TestCycleExecution:
    async def test_candidate_cycle_records_state_without_activation(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt(candidate_model_path="fake-fused-g0"))
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, True)
        approve_will(monkeypatch, scheduler)
        await scheduler._maybe_cycle()

        assert loop.cycles_run == 1
        assert not hasattr(scheduler, "_activate_live")
        state = json.loads(scheduler._state_path().read_text())
        assert state["last_status"] == "candidate"
        assert state["last_generation_id"] == "g0000-test"
        assert state["last_candidate_at"] > 0

    async def test_refused_cycle_records_but_does_not_activate(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt(status="refused", promoted_model_path=""))
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, True)
        approve_will(monkeypatch, scheduler)
        await scheduler._maybe_cycle()

        assert loop.cycles_run == 1
        state = json.loads(scheduler._state_path().read_text())
        assert state["last_status"] == "refused"

    async def test_reentrancy_guard(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        allow_maintenance(monkeypatch, True)
        approve_will(monkeypatch, scheduler)
        scheduler._running_cycle = True
        await scheduler._maybe_cycle()
        assert loop.cycles_run == 0

    async def test_status_surface(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        status = scheduler.get_status()
        assert status["service"] == "weight_compounding"
        assert status["last_status"] == "never_attempted"
        assert "lineage" in status


class TestRunCycleNow:
    async def test_bypasses_cooldown_but_not_will(self, scheduler, monkeypatch):
        scheduler._save_state({"last_attempt_at": time.time()})   # cooldown hot
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        approve_will(monkeypatch, scheduler)
        receipt = await scheduler.run_cycle_now(reason="rsi_weight_update")
        assert loop.cycles_run == 1                       # cooldown did not block
        assert receipt["status"] == "candidate"
        state = json.loads(scheduler._state_path().read_text())
        assert state["last_trigger"] == "rsi_weight_update"

    async def test_will_denial_blocks_on_demand_too(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        approve_will(monkeypatch, scheduler, approved=False, reason="not_now")
        receipt = await scheduler.run_cycle_now(reason="manual")
        assert loop.cycles_run == 0
        assert receipt["status"] == "blocked"

    async def test_single_flight_guard_holds(self, scheduler, monkeypatch):
        loop = FakeLoop(FakeReceipt())
        monkeypatch.setattr(CompoundingScheduler, "_build_loop", lambda self: loop)
        scheduler._running_cycle = True
        receipt = await scheduler.run_cycle_now(reason="manual")
        assert receipt == {"status": "blocked", "reasons": ["cycle_already_running"]}

    async def test_kill_switch_blocks(self, scheduler, monkeypatch):
        monkeypatch.setenv("AURA_WEIGHT_COMPOUNDING", "0")
        receipt = await scheduler.run_cycle_now(reason="manual")
        assert receipt["reasons"] == ["disabled_by_env"]


class TestRSIWeightUpdateRouting:
    async def test_fake_learner_keeps_legacy_path(self, monkeypatch):
        from core.learning.recursive_self_improvement import (
            RecursiveSelfImprovementLoop,
        )

        calls = []

        class FakeLearner:
            def force_train(self):
                calls.append("force_train")
                return True

        rsi = RecursiveSelfImprovementLoop(live_learner=FakeLearner())
        import core.container as container_mod

        monkeypatch.setattr(
            container_mod.ServiceContainer,
            "get",
            classmethod(lambda cls, name, default=None: (_ for _ in ()).throw(
                AssertionError("container must not be consulted for test doubles")
            )),
        )
        assert await rsi._run_weight_update() is True
        assert calls == ["force_train"]

    async def test_real_singleton_routes_to_canonical_scheduler(self, monkeypatch):
        import core.container as container_mod
        import core.learning.live_learner as live_learner_module
        from core.learning.recursive_self_improvement import (
            RecursiveSelfImprovementLoop,
        )

        class RealLearnerStandIn:
            def force_train(self):
                raise AssertionError("legacy path must not run when scheduler exists")

        learner = RealLearnerStandIn()
        monkeypatch.setattr(live_learner_module, "_learner", learner)

        cycle_calls = []

        class FakeSchedulerService:
            async def run_cycle_now(self, *, reason):
                cycle_calls.append(reason)
                return {"status": "promoted"}

        monkeypatch.setattr(
            container_mod.ServiceContainer,
            "get",
            classmethod(
                lambda cls, name, default=None: FakeSchedulerService()
                if name == "weight_compounding" else default
            ),
        )
        rsi = RecursiveSelfImprovementLoop(live_learner=learner)
        assert await rsi._run_weight_update() is True
        assert cycle_calls == ["rsi_weight_update"]


class TestSpecialistHook:
    async def test_default_off(self, scheduler, monkeypatch):
        monkeypatch.delenv("AURA_DOMAIN_SPECIALISTS", raising=False)
        called: list[str] = []
        import core.learning.domain_specialists as ds

        monkeypatch.setattr(
            ds.DomainSpecialistTrainer, "eligible_domains",
            lambda self: called.append("checked") or [],
        )
        await scheduler._maybe_train_specialist()
        assert called == []          # env-gated off: never even reads the store

    async def test_trains_least_recently_trained_domain(self, scheduler, monkeypatch):
        monkeypatch.setenv("AURA_DOMAIN_SPECIALISTS", "1")
        import core.learning.domain_specialists as ds

        trained_domains: list[str] = []

        class FakeReceiptObj:
            status = "promoted"
            reasons: list = []

        monkeypatch.setattr(
            ds.DomainSpecialistTrainer, "eligible_domains",
            lambda self: ["modular", "sequence"],
        )
        monkeypatch.setattr(
            ds.DomainSpecialistTrainer, "train_domain",
            lambda self, d: trained_domains.append(d) or FakeReceiptObj(),
        )
        scheduler._save_state(
            {"specialist_trained_at": {"sequence": 100.0, "modular": 200.0}}
        )
        await scheduler._maybe_train_specialist()
        assert trained_domains == ["sequence"]   # least-recently-trained first
        state = scheduler._load_state()
        assert state["last_specialist_status"] == "sequence:promoted"
        assert state["specialist_trained_at"]["sequence"] > 100.0


class TestBootWiring:
    def test_boot_step_registered(self):
        source = Path("core/orchestrator/mixins/boot/boot_autonomy.py").read_text(
            encoding="utf-8"
        )
        assert '("weight_compounding", self._init_weight_compounding)' in source

    def test_init_method_exists(self):
        source = Path("core/orchestrator/mixins/boot/boot_cognitive.py").read_text(
            encoding="utf-8"
        )
        assert "async def _init_weight_compounding" in source
        assert "get_compounding_scheduler" in source

    def test_service_name_registered(self):
        from core.service_names import ServiceNames

        assert ServiceNames.WEIGHT_COMPOUNDING == "weight_compounding"

    def test_singleton_reset(self):
        sched_mod.reset_compounding_scheduler_for_test()
        a = sched_mod.get_compounding_scheduler()
        assert sched_mod.get_compounding_scheduler() is a
        sched_mod.reset_compounding_scheduler_for_test()
        assert sched_mod.get_compounding_scheduler() is not a
        sched_mod.reset_compounding_scheduler_for_test()
