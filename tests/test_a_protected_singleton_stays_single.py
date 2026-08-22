"""A blocked registration still handed back the rejected instance.

108 "Protected core service overwrite blocked after lock:
'executive_authority'" in one sampled window. The noise was the small half.

get_executive_authority builds a new ExecutiveAuthority whenever the container
does not return one, then registers it. register_instance does NOT raise when
it refuses — after the registry is locked it logs and returns — so the
try/except around it never fired, and the function returned the very object the
container had just rejected.

Every caller taking that path was therefore deciding executive authority on a
throwaway object while the canonical one sat in the container. A protected
singleton was silently one instance per call, which is the opposite of what
protecting it was for.
"""
from __future__ import annotations

import warnings

import pytest

from core.consciousness.executive_authority import (
    ExecutiveAuthority,
    get_executive_authority,
)
from core.container import ServiceContainer

#: What a registry teardown is allowed to fail with. Anything else is a
#: defect in the container and should surface.
_TEARDOWN_ERRORS = (AttributeError, KeyError, RuntimeError, TypeError, ValueError)


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    try:
        ServiceContainer.register_instance(
            "executive_authority", None, required=False
        )
    except _TEARDOWN_ERRORS as teardown_exc:
        # Teardown must not mask a test failure, so this does not raise. It
        # does report, because a clean-up that has silently stopped working
        # leaks a registered singleton into every test that follows.
        warnings.warn(
            f"registry teardown did not clear executive_authority: {teardown_exc!r}",
            RuntimeWarning,
            stacklevel=2,
        )


def test_the_container_instance_wins_when_registration_is_refused(monkeypatch):
    """The exact live path: refusal is silent, so it must be re-checked."""
    canonical = ExecutiveAuthority(orchestrator=None)

    calls = {"n": 0}

    def fake_get(name, default=None):
        # First call (the early return) misses; the re-read after the blocked
        # registration finds the canonical one, exactly as the live container
        # behaves once locked.
        calls["n"] += 1
        if name != "executive_authority":
            return default
        return None if calls["n"] == 1 else canonical

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(fake_get))
    monkeypatch.setattr(
        ServiceContainer,
        "register_instance",
        staticmethod(lambda *a, **k: None),  # refuses silently, as when locked
    )

    result = get_executive_authority()

    assert result is canonical, "the rejected instance must not be returned"


def test_a_fresh_registry_keeps_the_instance_it_just_built(monkeypatch):
    """When registration is genuinely accepted, nothing is discarded."""
    stored: dict[str, object] = {}

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: stored.get(name, default)),
    )
    monkeypatch.setattr(
        ServiceContainer,
        "register_instance",
        staticmethod(lambda name, instance, **k: stored.__setitem__(name, instance)),
    )

    first = get_executive_authority()
    second = get_executive_authority()

    assert isinstance(first, ExecutiveAuthority)
    assert second is first, "a singleton must not be rebuilt per call"


def test_a_container_that_raises_does_not_break_the_caller(monkeypatch):
    """Authority must still be obtainable when the registry is unhappy."""

    def exploding(name, default=None):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(exploding))
    monkeypatch.setattr(
        ServiceContainer, "register_instance", staticmethod(lambda *a, **k: None)
    )

    assert isinstance(get_executive_authority(), ExecutiveAuthority)
