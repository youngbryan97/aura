"""Her immune system asks to reduce load only when the resource governor, which owns load, names an eviction tier.

LIVE 13 November 2026: resource pressure over 0.78 is her resident 27B's ordinary footprint. Read as an antigen, it
asked autopoiesis to reduce_load/global every fifteen seconds, and once a handler existed for that request it ran 536
ineffective repairs in two hours, each fed back as a failure.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.adaptation import adaptive_immunity as immunity
from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneConfig,
    AdaptiveImmuneSystem,
    Antigen,
    CellKind,
    ImmuneCell,
)
from core.resource import resource_governor


def _loaded_antigen() -> Antigen:
    return Antigen(antigen_id="ag_load", subsystem="runtime_engine", vector=np.zeros(16, dtype=np.float32),
                   danger=0.3, subsystem_need=0.5, threat_probability=0.5, resource_pressure=0.82, error_load=0.1,
                   health_pressure=0.1, temporal_pressure=0.1, recurrence_pressure=0.1, error_signature="pressure")


def _kind_chosen(tmp_path, monkeypatch, tier) -> str | None:
    monkeypatch.setattr(resource_governor, "get_resource_governor",
                        lambda: SimpleNamespace(sample=lambda: SimpleNamespace(eviction_tier=tier)))
    immune = AdaptiveImmuneSystem(config=AdaptiveImmuneConfig(population_size=1, max_population=4,
                                                              dream_every_observations=100),
                                  state_dir=tmp_path, rng_seed=7)
    antigen = _loaded_antigen()
    cell = ImmuneCell(cell_id="b", lineage_id="b_lineage", kind=CellKind.B, receptor=antigen.vector.copy(),
                      subsystem_scope="runtime_engine", persistence=0.8, fitness=0.4)
    artifact = immune._emit_artifact(cell, antigen, affinity=1.0, activation=0.9)
    return artifact.kind.value if artifact is not None else None


def test_her_models_ordinary_footprint_is_not_load_to_reduce(tmp_path, monkeypatch):
    assert _kind_chosen(tmp_path, monkeypatch, resource_governor.EvictionTier.NONE) != "reduce_load"


def test_load_the_governor_names_is_still_reduced(tmp_path, monkeypatch):
    assert _kind_chosen(tmp_path, monkeypatch, resource_governor.EvictionTier.MODERATE) == "reduce_load"


def test_without_a_governor_reading_the_antigen_stands(monkeypatch):
    def unreadable():
        raise RuntimeError("no governor")

    monkeypatch.setattr(resource_governor, "get_resource_governor", unreadable)
    assert immunity._the_governor_names_load() is True
