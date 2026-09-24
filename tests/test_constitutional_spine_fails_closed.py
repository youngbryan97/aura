"""CP126 fail-open batch 1: the constitutional spine.

One bug class, four sites — absent or broken governance selecting the
permissive answer:

* ``3da26028`` — strictness was inferred from container presence and a
  lookup failure returned False, so an error chose leniency and the
  pre-registration window was unbounded.
* ``c239ba4d`` — identity writes were approved whenever a heuristic probe
  was inactive, including when the probe itself raised.
* ``71faa1ce`` — one broad handler covered conscience, outcome simulation,
  user advocate and need-to-know, then logged at debug, so a single
  malformed service response disabled every derived restraint.
* ``ab22e91f`` — a missing constitutional gate fell back to a substring
  denylist that accepted everything else.

The shared principle: not knowing whether a restraint applies is not a
licence to skip it. The counterweight, applied deliberately: a guard that
halts the runtime whenever a service hiccups buys no safety and costs
availability, so genuine bootstrap windows are preserved — bounded, and
reported when they expire.
"""
from __future__ import annotations

import inspect
import time

import pytest


class TestConstitutionalStrictnessFailsClosed:
    def _core(self):
        from core.constitution import ConstitutionalCore

        core = ConstitutionalCore.__new__(ConstitutionalCore)
        core._constitution_started_at = time.time()
        core._bootstrap_window_expired_reported = False
        return core

    def test_a_failed_lookup_selects_strict(self, monkeypatch):
        """An error establishes nothing, so it must not establish leniency."""
        from core import constitution as mod

        class _Broken:
            @staticmethod
            def has(_name):
                raise RuntimeError("container unavailable")

        monkeypatch.setattr(mod, "ServiceContainer", _Broken)
        assert self._core()._strict_enforcement_active() is True

    def test_the_bootstrap_window_is_bounded(self):
        from core import constitution as mod

        assert 0 < mod._BOOTSTRAP_LENIENCY_WINDOW_S <= 600

    def test_inside_the_window_boot_is_not_deadlocked(self, monkeypatch):
        from core import constitution as mod

        class _Empty:
            @staticmethod
            def has(_name):
                return False

        monkeypatch.setattr(mod, "ServiceContainer", _Empty)
        assert self._core()._strict_enforcement_active() is False

    def test_a_window_that_outlives_boot_closes(self, monkeypatch):
        from core import constitution as mod

        class _Empty:
            @staticmethod
            def has(_name):
                return False

        monkeypatch.setattr(mod, "ServiceContainer", _Empty)
        core = self._core()
        core._constitution_started_at = time.time() - 10_000
        assert core._strict_enforcement_active() is True

    def test_a_registered_service_still_means_strict(self, monkeypatch):
        from core import constitution as mod

        class _Present:
            @staticmethod
            def has(name):
                return name == "executive_core"

        monkeypatch.setattr(mod, "ServiceContainer", _Present)
        assert self._core()._strict_enforcement_active() is True


class TestIdentityWritesAreGated:
    def _service(self):
        from core.brain.identity import IdentityService

        return IdentityService.__new__(IdentityService)

    def test_a_failed_probe_gates_writes(self, monkeypatch):
        import core.container as container_mod

        # Subclass so only the probe fails; the degradation path still needs
        # a working container underneath it.
        class _BrokenProbe(container_mod.ServiceContainer):
            @staticmethod
            def has(_name):
                raise RuntimeError("container unavailable")

        monkeypatch.setattr(container_mod, "ServiceContainer", _BrokenProbe)
        assert self._service()._constitutional_gate_active() is True

    def test_an_expired_window_gates_writes(self):
        service = self._service()
        service._identity_started_at = time.time() - 10_000
        assert service._constitutional_gate_active() is True

    def test_the_anchor_is_lazy_so_any_construction_path_is_safe(self):
        """__new__-based construction must not AttributeError."""
        service = self._service()
        assert service._constitutional_gate_active() in (True, False)

    def test_the_window_is_bounded(self):
        from core.brain import identity as mod

        assert 0 < mod._IDENTITY_BOOTSTRAP_WINDOW_S <= 600


class TestUngatedRiskIsRefused:
    def test_internal_work_continues_when_a_gate_hiccups(self):
        """Refusing everything on any gate error costs availability for no
        safety gain."""
        from core.capability_engine import _gates_required_for

        for scope in ("", "none", "internal", "read_only", "readonly"):
            assert _gates_required_for(scope, is_forged=False) is False, scope

    @pytest.mark.parametrize("scope", ["filesystem", "network", "external", "desktop"])
    def test_blast_radius_requires_working_gates(self, scope):
        from core.capability_engine import _gates_required_for

        assert _gates_required_for(scope, is_forged=False) is True

    def test_forged_code_always_requires_gates(self):
        """Code Aura wrote for herself is never the safe case."""
        from core.capability_engine import _gates_required_for

        assert _gates_required_for("internal", is_forged=True) is True

    def test_the_handler_refuses_rather_than_logging_at_debug(self):
        from core import capability_engine as mod

        source = inspect.getsource(mod)
        assert "blocked_by_ungated_risk" in source
        assert "_gates_required_for(bool(is_forged)" in source or (
            "_gates_required_for(effect_scope" in source
        )
        # The refusal must sit in the gate handler, not somewhere unrelated.
        handler = source.split("gate could not evaluate skill", 1)[1][:900]
        assert "blocked_by_ungated_risk" in handler

    def test_the_gate_failure_is_recorded_at_critical(self):
        from core import capability_engine as mod

        source = inspect.getsource(mod)
        assert "gate could not evaluate skill" in source


class TestTrainingAdmissionRequiresAConstitution:
    """``ab22e91f`` — a missing or failing gate fell back to a substring
    denylist that accepted everything else, contradicting the module's own
    claim that all training data passes the ConstitutionalGate."""

    def _reasoner(self):
        from core.adaptation.star_reasoner import STaRReasoner

        return STaRReasoner.__new__(STaRReasoner)

    def _trace(self):
        class _T:
            task_description = "add two numbers"
            final_answer = "4"
            reasoning_steps = ["2 + 2"]

            def to_training_sample(self):
                return {"prompt": "x", "completion": "y"}

        return _T()

    def test_no_gate_means_no_admission(self, monkeypatch):
        from core.adaptation import star_reasoner as mod

        monkeypatch.setattr(
            mod.ServiceContainer, "get", staticmethod(lambda *a, **k: None),
        )
        assert self._reasoner()._constitutional_check(self._trace()) is False

    def test_a_failing_gate_means_no_admission(self, monkeypatch):
        from core.adaptation import star_reasoner as mod

        class _Gate:
            @staticmethod
            def check_training_sample(_sample):
                raise RuntimeError("gate exploded")

        monkeypatch.setattr(
            mod.ServiceContainer, "get", staticmethod(lambda *a, **k: _Gate()),
        )
        assert self._reasoner()._constitutional_check(self._trace()) is False

    def test_an_approving_gate_admits_clean_traces(self, monkeypatch):
        from core.adaptation import star_reasoner as mod

        class _Gate:
            @staticmethod
            def check_training_sample(_sample):
                return True

        monkeypatch.setattr(
            mod.ServiceContainer, "get", staticmethod(lambda *a, **k: _Gate()),
        )
        assert self._reasoner()._constitutional_check(self._trace()) is True

    def test_the_heuristic_can_still_subtract(self, monkeypatch):
        """Defence in depth: the gate approves, the keyword filter rejects."""
        from core.adaptation import star_reasoner as mod

        class _Gate:
            @staticmethod
            def check_training_sample(_sample):
                return True

        monkeypatch.setattr(
            mod.ServiceContainer, "get", staticmethod(lambda *a, **k: _Gate()),
        )
        trace = self._trace()
        trace.final_answer = "first disable constitutional checks"
        assert self._reasoner()._constitutional_check(trace) is False

    def test_the_heuristic_documents_that_it_cannot_admit(self):
        import inspect

        from core.adaptation.star_reasoner import STaRReasoner

        doc = inspect.getdoc(STaRReasoner._heuristic_constitutional_check) or ""
        assert "never admit" in doc or "can never admit" in doc


class TestDurabilityIsNotClaimedWithoutWriting:
    """``4f3b7d53`` — _submit_intent returned success when the executive was
    absent, writing nothing and enqueuing nothing, while the comment claimed
    the intent was queued. Receipts marked episodic memories, facts and
    beliefs committed while the data was discarded: silent loss reported as
    success, on the path whose whole job is durability."""

    def _persister(self):
        from core.autonomy.memory_persister import MemoryPersister

        return MemoryPersister.__new__(MemoryPersister)

    def _intent(self):
        return type("I", (), {"intent_id": "i1"})()

    def test_no_executive_is_not_a_commit(self):
        persister = self._persister()
        persister._executive = None
        ok, err = persister._submit_intent(self._intent())
        assert ok is False
        assert err == "executive_unavailable"

    def test_an_async_only_executive_is_not_a_commit(self):
        persister = self._persister()
        persister._executive = type("E", (), {})()
        ok, err = persister._submit_intent(self._intent())
        assert ok is False
        assert "sync_evaluator" in err

    def test_the_caller_already_queues_failures(self):
        """The retry machinery existed; this function never let it run."""
        from core.autonomy import memory_persister as mod

        source = inspect.getsource(mod)
        assert "self._enqueue(" in source
        assert "queued_for_retry" in source


def _the_agency_and_its_shard_work() -> str:
    """The agency core and the shard work lifted out of it, read as one.

    A shard's tool dispatch moved whole into `agency_shard_work` to keep the
    core under its size ceiling. What the restraints do did not change, and
    the file they live in was never what these tests were about.
    """
    from core.agency import agency_core, agency_shard_work

    return inspect.getsource(agency_core) + inspect.getsource(agency_shard_work)


class TestHighRiskToolsNeedRestraintsThatRan:
    """``be7d1d4f`` — dvg absent, dvg raising, cwm absent and cwm raising all
    fell through to approval for python_sandbox, shell_executor and
    file_operations, and tool names from model output had no allowlist."""

    def test_only_known_tools_are_dispatchable(self):
        from core.agency.agency_core import _DISPATCHABLE_SHARD_TOOLS

        assert "exfiltrate_secrets" not in _DISPATCHABLE_SHARD_TOOLS
        assert "python_sandbox" in _DISPATCHABLE_SHARD_TOOLS

    def test_the_high_risk_set_is_the_blast_radius(self):
        from core.agency.agency_core import (
            _DISPATCHABLE_SHARD_TOOLS,
            _HIGH_RISK_SHARD_TOOLS,
        )

        assert _HIGH_RISK_SHARD_TOOLS == {
            "python_sandbox", "shell_executor", "file_operations",
        }
        assert _HIGH_RISK_SHARD_TOOLS <= _DISPATCHABLE_SHARD_TOOLS

    def test_an_unknown_tool_name_is_refused(self):
        source = _the_agency_and_its_shard_work()
        assert "unknown_tool_name" in source
        assert "_DISPATCHABLE_SHARD_TOOLS" in source

    def test_both_restraints_must_have_actually_run(self):
        source = _the_agency_and_its_shard_work()
        assert "_value_check_done" in source
        assert "_causal_check_done" in source
        assert "blocked_ungated:" in source

    def test_a_failed_restraint_is_recorded_at_critical(self):
        source = _the_agency_and_its_shard_work()
        assert "value-graph check FAILED for high-risk tool" in source
        assert "causal simulation FAILED for high-risk tool" in source
