################################################################################

"""
Unit tests for core.container.ServiceContainer.
"""
import pytest
from core.container import ServiceContainer, ServiceLifetime

@pytest.fixture
def clean_container():
    """Provides a fresh container for each test."""
    ServiceContainer.clear()
    return ServiceContainer


class TestServiceRegistration:
    def test_register_and_get(self, clean_container):
        """Services can be registered and retrieved."""
        clean_container.register("test_svc", lambda: {"status": "ok"})
        result = clean_container.get("test_svc")
        assert result == {"status": "ok"}

    def test_singleton_returns_same_instance(self, clean_container):
        """Singleton lifetime returns the same object on repeated gets."""
        clean_container.register(
            "counter", lambda: {"count": 0}, lifetime=ServiceLifetime.SINGLETON
        )
        a = clean_container.get("counter")
        b = clean_container.get("counter")
        assert a is b

    def test_transient_returns_different_instances(self, clean_container):
        """Transient lifetime creates a new object each time."""
        clean_container.register(
            "ephemeral", lambda: {"count": 0}, lifetime=ServiceLifetime.TRANSIENT
        )
        a = clean_container.get("ephemeral")
        b = clean_container.get("ephemeral")
        assert a is not b

    def test_register_instance(self, clean_container):
        """Pre-created instances can be registered directly."""
        obj = {"pre_made": True}
        clean_container.register_instance("preset", obj)
        assert clean_container.get("preset") is obj

    def test_register_normalizes_legacy_instance_input(self, clean_container):
        """Legacy callers that pass an instance to register() should still resolve cleanly."""
        obj = object()
        clean_container.register("legacy", obj)
        assert clean_container.get("legacy") is obj

    def test_get_missing_service_raises(self, clean_container):
        """Accessing an unregistered service raises ServiceNotFoundError."""
        from core.exceptions import ServiceNotFoundError
        with pytest.raises(ServiceNotFoundError, match="not found"):
            clean_container.get("nonexistent")

    def test_get_missing_with_default(self, clean_container):
        """Accessing missing service with default returns the default."""
        result = clean_container.get("nonexistent", default="fallback")
        assert result == "fallback"


class TestCircularDependency:
    def test_circular_dependency_detected(self, clean_container):
        """Circular dependencies raise CircularDependencyError."""
        from core.exceptions import CircularDependencyError
        clean_container.register("a", lambda b: None, dependencies=["b"])
        clean_container.register("b", lambda a: None, dependencies=["a"])
        with pytest.raises(CircularDependencyError, match="Circular dependency"):
            clean_container.get("a")


class TestValidation:
    def test_validate_success(self, clean_container):
        """Validation passes when all required services resolve."""
        clean_container.register("svc", lambda: "hello")
        ok, errors = clean_container.validate()
        assert ok is True
        assert errors == []

    def test_validate_missing_dependency(self, clean_container):
        """Validation reports missing dependency."""
        clean_container.register("svc", lambda dep: None, dependencies=["dep"])
        ok, errors = clean_container.validate()
        assert ok is False
        assert any("dep" in e for e in errors)

    def test_health_report(self, clean_container):
        """Health report includes registered services."""
        clean_container.register("test", lambda: "val")
        clean_container.get("test")  # Materialize it
        report = clean_container.get_health_report()
        assert "test" in report["services"]
        assert report["services"]["test"]["status"] == "online"

    def test_health_report_marks_invalid_sovereignty_seal_degraded(self, clean_container, tmp_path, monkeypatch):
        """Seal drift should surface in the health report."""
        seal_path = tmp_path / "sovereignty_seal.json"
        monkeypatch.setattr(ServiceContainer, "_seal_path", classmethod(lambda cls: seal_path))

        clean_container.register_instance("alpha", object())
        clean_container.write_sovereignty_seal()
        clean_container.register_instance("beta", object())

        report = clean_container.get_health_report()

        assert report["status"] == "degraded"
        assert report["sovereignty_seal"]["present"] is True
        assert report["sovereignty_seal"]["valid"] is False


##
