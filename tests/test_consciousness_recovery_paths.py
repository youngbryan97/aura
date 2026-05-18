from __future__ import annotations

import sys
import types
from collections import deque

import numpy as np

from core.consciousness import neurochemical_system, phi_core, substrate_authority
from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.consciousness.phi_core import PhiCore
from core.consciousness.substrate_authority import (
    ActionCategory,
    AuthorizationDecision,
    SubstrateAuthority,
    SubstrateVerdict,
)


class _FailingCallable:
    def __init__(self, message: str):
        self.message = message
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError(self.message)


def test_neurochemical_optional_links_emit_degradation_receipts(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        neurochemical_system,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    broken_container = types.SimpleNamespace(
        get=_FailingCallable("drive registry unavailable")
    )

    monkeypatch.setattr("core.container.ServiceContainer", broken_container)

    ncs = NeurochemicalSystem()
    ncs.on_success()
    ncs.on_novelty(0.4)

    assert recorded == [
        ("neurochemical_system", "RuntimeError"),
        ("neurochemical_system", "RuntimeError"),
    ]


def test_adaptive_mood_failures_fall_back_with_receipts(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        neurochemical_system,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    broken_adaptive_mood = types.SimpleNamespace(
        predict=_FailingCallable("mood predictor unavailable"),
        update_from_outcome=_FailingCallable("mood learner unavailable"),
    )

    monkeypatch.setattr(
        "core.consciousness.adaptive_mood.get_adaptive_mood",
        lambda: broken_adaptive_mood,
    )

    ncs = NeurochemicalSystem()

    mood = ncs.get_mood_vector()
    update = ncs.learn_mood_from_outcome({"valence": 0.2})

    assert {"valence", "arousal", "motivation", "sociality", "stress", "calm", "wakefulness"} <= set(mood)
    assert update == {}
    assert recorded == [
        ("neurochemical_system", "RuntimeError"),
        ("neurochemical_system", "RuntimeError"),
    ]


def test_substrate_authority_reader_and_audit_failures_are_visible(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        substrate_authority,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    authority = SubstrateAuthority()

    class _BrokenChemistry:
        def __init__(self):
            self.calls: list[str] = []

        @property
        def chemicals(self):
            self.calls.append("chemicals")
            raise RuntimeError("chemistry snapshot unavailable")

        def on_frustration(self, *_args, **_kwargs):
            self.calls.append("on_frustration")
            raise RuntimeError("feedback unavailable")

    broken_chemistry = _BrokenChemistry()

    authority._field_ref = types.SimpleNamespace(
        get_coherence=_FailingCallable("field offline")
    )
    authority._somatic_ref = types.SimpleNamespace(
        evaluate=_FailingCallable("somatic gate offline")
    )
    authority._neurochemical_ref = broken_chemistry

    assert authority._get_field_coherence() == 0.5
    assert authority._get_somatic_state("content", "source", 0.5) == (0.0, 0.0, True)
    assert authority._get_neurochemical_constraints(ActionCategory.RESPONSE) == ("normal", [])
    authority._neurochemical_feedback(AuthorizationDecision.BLOCK, ActionCategory.RESPONSE)

    import core.consciousness.authority_audit as authority_audit

    monkeypatch.setattr(
        authority_audit,
        "get_audit",
        _FailingCallable("audit unavailable"),
    )
    authority._record(
        SubstrateVerdict(
            decision=AuthorizationDecision.ALLOW,
            reason="test",
            field_coherence=0.6,
            somatic_approach=0.0,
            somatic_confidence=0.0,
            neurochemical_state="normal",
            body_budget_available=True,
            constraints=[],
        )
    )

    assert recorded == [
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
        ("substrate_authority", "RuntimeError"),
    ]


def test_phi_core_degradation_receipts_for_metric_and_surrogate_failures(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        phi_core,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    metrics_module = types.ModuleType("core.observability.metrics")
    metrics_module.get_metrics = _FailingCallable("metrics unavailable")
    monkeypatch.setitem(sys.modules, "core.observability.metrics", metrics_module)

    phi = PhiCore.__new__(PhiCore)
    assert phi._detect_disconnected_graph(np.zeros((3, 3))) == (False, 3, [1, 1, 1])

    phi._last_result = None
    phi._state_history = deque([0] * 20)
    phi.compute_surrogate_phi = _FailingCallable("surrogate unavailable")

    assert phi.get_live_phi() == 0.0
    assert recorded == [
        ("phi_core", "RuntimeError"),
        ("phi_core", "RuntimeError"),
    ]
