"""An immune system that cannot enforce must not look fully armed.

CP126 (critical), core/service_registration.py: "Immune system silently
ignores enforcement backend failure. Firewall, quarantine, process,
resource, and ARP backend activation errors are swallowed while the immune
decision layer is returned as available."

The whole defect was ``except (...): pass``. An immune system that can
DECIDE to quarantine but cannot ENFORCE it is not a degraded immune system,
it is a reporting one — and every caller asking ServiceContainer for
"immune_system" received an object that looked fully armed. Detection
without enforcement is the most dangerous shape a security control can take,
precisely because it is trusted.

The decision layer genuinely works without the backends, so activation stays
best-effort and the immune system is still returned. What changed is that
the gap is recorded as a degradation and readable off the object.
"""
from __future__ import annotations

import pytest

import core.service_registration as reg
from core.service_registration import install_immune_enforcement


class _Immune:
    """Stand-in decision layer."""


@pytest.fixture
def degradations(monkeypatch):
    recorded: list = []
    monkeypatch.setattr(reg, "record_degradation", lambda *a, **k: recorded.append(a))
    return recorded


def _backend(monkeypatch, *, fails: Exception | None):
    def _activate():
        if fails is not None:
            raise fails

    monkeypatch.setattr(
        "core.security.defensive_runtime.ensure_defensive_runtime_active",
        _activate,
    )


class TestAFailedBackendIsVisible:
    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("firewall backend refused"),
            OSError("cannot open packet filter"),
            ImportError("arp module missing"),
            AttributeError("quarantine backend has no activate"),
        ],
    )
    def test_activation_failure_is_recorded(self, monkeypatch, degradations, error):
        _backend(monkeypatch, fails=error)
        install_immune_enforcement(_Immune())
        assert degradations, "enforcement backend failure was swallowed"

    def test_the_object_reports_enforcement_is_missing(self, monkeypatch, degradations):
        _backend(monkeypatch, fails=RuntimeError("no firewall"))
        immune = install_immune_enforcement(_Immune())
        assert immune.enforcement_backends_active is False
        assert "RuntimeError" in immune.enforcement_backends_error

    def test_the_decision_layer_is_still_returned(self, monkeypatch, degradations):
        """Refusing to return an immune system because enforcement is
        missing would be worse — detection still has value."""
        _backend(monkeypatch, fails=RuntimeError("no firewall"))
        original = _Immune()
        assert install_immune_enforcement(original) is original


class TestASuccessfulInstallIsAlsoStated:
    def test_active_enforcement_is_marked(self, monkeypatch, degradations):
        _backend(monkeypatch, fails=None)
        immune = install_immune_enforcement(_Immune())
        assert immune.enforcement_backends_active is True
        assert immune.enforcement_backends_error == ""

    def test_a_healthy_install_records_nothing(self, monkeypatch, degradations):
        _backend(monkeypatch, fails=None)
        install_immune_enforcement(_Immune())
        assert degradations == []


class TestAnnotationFailureDoesNotBreakBoot:
    def test_an_unannotatable_object_still_returns(self, monkeypatch, degradations):
        """A slotted or frozen immune system must not take the boot down."""
        _backend(monkeypatch, fails=None)

        class _Slotted:
            __slots__ = ()

        obj = _Slotted()
        assert install_immune_enforcement(obj) is obj
        assert degradations, "the annotation failure was silent"


class TestTheFactoryUsesIt:
    def test_the_registered_factory_delegates_here(self):
        import inspect

        source = inspect.getsource(reg._register_critique_closure_and_runtime_organs)
        factory = source.split("def _create_immune_system():", 1)[1][:400]
        assert "install_immune_enforcement" in factory

    def test_the_silent_pass_is_gone(self):
        import inspect

        source = inspect.getsource(install_immune_enforcement)
        handler = source.split("except (ImportError", 1)[1][:400]
        assert "record_degradation" in handler
        assert "pass" not in handler
