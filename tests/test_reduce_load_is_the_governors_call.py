"""reduce_load/global has a handler, and it sheds only what the resource governor's own tier names.

Live, 13 November 2026: her immune system asked for reduce_load/global (resource pressure over 0.78, which her
resident 27B holds most of the time) and autopoiesis had no handler for it.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.providers import self_maintenance_provider as provider
from core.resource import resource_governor


class _Governor:
    def __init__(self, tier):
        self.tier, self.shed = tier, []

    def sample(self):
        return SimpleNamespace(eviction_tier=self.tier)

    def execute_eviction(self, tier):
        self.shed.append(tier)
        return 1


def test_with_no_tier_named_nothing_is_shed(monkeypatch):
    governor = _Governor(resource_governor.EvictionTier.NONE)
    monkeypatch.setattr(resource_governor, "get_resource_governor", lambda: governor)
    assert provider._reduce_load_as_the_governor_judges() is True
    assert governor.shed == []


def test_the_tier_the_governor_names_is_shed(monkeypatch):
    governor = _Governor(resource_governor.EvictionTier.MODERATE)
    monkeypatch.setattr(resource_governor, "get_resource_governor", lambda: governor)
    assert provider._reduce_load_as_the_governor_judges() is True
    assert governor.shed == [resource_governor.EvictionTier.MODERATE]


def test_autopoiesis_is_built_with_the_handler():
    from core.cognitive.autopoiesis import RepairStrategy

    engine = provider._autopoiesis()
    assert engine._repair_handlers[RepairStrategy.REDUCE_LOAD]["global"] is provider._reduce_load_as_the_governor_judges
