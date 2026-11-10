"""The app's feeling is read from her running state, and reading it never builds a service.

The health snapshot calls her_feeling, and from 22 October it reached ServiceContainer.get, which can construct
what it is asked for: tests/test_health_read_model.py failed on main from then on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from interface.routes import feeling

pytestmark = pytest.mark.unit


def test_her_feeling_comes_from_the_running_orchestrator_and_nothing_is_built(monkeypatch):
    affect = SimpleNamespace(emotions={"joy": 0.9, "fear": 0.1}, mood_baselines={"joy": 0.2, "fear": 0.1})
    state = SimpleNamespace(affect=affect, cognition=SimpleNamespace(), identity=SimpleNamespace())
    orchestrator = SimpleNamespace(state_repo=SimpleNamespace(_current=state))

    def built(*_args, **_kwargs):
        raise AssertionError("reading her feeling must not construct a service")

    monkeypatch.setattr(ServiceContainer, "get", built)
    monkeypatch.setattr(
        ServiceContainer, "peek", staticmethod(lambda name, default=None: orchestrator if name == "orchestrator" else default)
    )
    named = feeling.her_feeling()
    assert named and named[0]["name"] == "joy"


def test_with_nothing_running_she_names_no_feeling(monkeypatch):
    monkeypatch.setattr(ServiceContainer, "peek", staticmethod(lambda name, default=None: default))
    assert feeling.her_feeling() == []
