import asyncio
from dataclasses import dataclass

import numpy as np

from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneConfig,
    AdaptiveImmuneSystem,
    Antigen,
    CellKind,
    ImmuneCell,
    TissueField,
)


def run(coro):
    return asyncio.run(coro)


@dataclass
class _RepairResult:
    success: bool = True
    governance_approved: bool = True


class _AutopoiesisStub:
    def __init__(self):
        self.calls = []

    async def request_repair(self, component, strategy):
        self.calls.append((component, getattr(strategy, "value", str(strategy))))
        return _RepairResult(success=True, governance_approved=True)


def test_tissue_field_diffuses_local_damage():
    field = TissueField(diffusion=0.35, decay=0.0)
    field.register_edge("state_repository", "continuity", 1.0)

    antigen = Antigen(
        antigen_id="ag_tissue",
        subsystem="state_repository",
        vector=np.zeros(16, dtype=np.float32),
        danger=0.9,
        subsystem_need=0.8,
        threat_probability=0.9,
        resource_pressure=0.7,
        error_load=0.6,
        health_pressure=0.4,
        temporal_pressure=0.2,
        protected=False,
    )
    field.ingest_antigen(antigen)

    snapshot = field.to_dict()
    assert snapshot["danger"]["state_repository"] > 0.0
    assert snapshot["danger"]["continuity"] > 0.0
    assert field.get_need("continuity") > 0.0


def test_regulatory_cells_suppress_autoimmune_effectors(tmp_path):
    immune = AdaptiveImmuneSystem(state_dir=tmp_path, rng_seed=1)
    antigen = immune.present_antigen(
        {
            "type": "exception",
            "text": "Identity coherence failure",
            "subsystem": "identity_core",
            "source": "identity_guard",
            "danger": 0.95,
            "error_count": 6,
            "resource_pressure": 0.4,
            "stack_trace": "X" * 800,
        }
    )

    killer = ImmuneCell(
        cell_id="killer",
        lineage_id="killer_lineage",
        kind=CellKind.CYTOTOXIC,
        receptor=antigen.vector.copy(),
        subsystem_scope="identity_core",
        persistence=0.6,
    )
    regulator = ImmuneCell(
        cell_id="regulator",
        lineage_id="reg_lineage",
        kind=CellKind.REGULATORY,
        receptor=antigen.vector.copy(),
        subsystem_scope="identity_core",
        persistence=0.9,
        regulatory_strength=1.4,
    )
    immune._cells = [killer, regulator]

    response, _ = immune._observe_core(antigen)
    assert response.suppression_applied > 0.0
    assert response.artifacts
    assert all(artifact.suppressed for artifact in response.artifacts)


def test_successful_lineage_clones_and_persists_as_memory(tmp_path):
    cfg = AdaptiveImmuneConfig(
        population_size=1,
        max_population=8,
        dream_every_observations=100,
    )
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=2)
    autopoiesis = _AutopoiesisStub()
    immune._get_service = lambda name: autopoiesis if name == "autopoiesis" else None

    antigen = immune.present_antigen(
        {
            "type": "error_signature",
            "text": "database is locked",
            "subsystem": "state_repository",
            "source": "signature",
            "danger": 0.9,
            "error_count": 5,
            "resource_pressure": 0.2,
        }
    )
    seed = ImmuneCell(
        cell_id="seed_b",
        lineage_id="seed_lineage",
        kind=CellKind.B,
        receptor=antigen.vector.copy(),
        subsystem_scope="state_repository",
        persistence=0.7,
        fitness=0.5,
    )
    immune._cells = [seed]

    event = {
        "type": "error_signature",
        "text": "database is locked",
        "subsystem": "state_repository",
        "source": "signature",
        "danger": 0.9,
        "error_count": 5,
        "resource_pressure": 0.2,
    }
    run(immune.observe_event(event))
    run(immune.observe_event(event))

    assert autopoiesis.calls
    assert len(immune._cells) > 1

    immune.dream_consolidate()
    assert any(cell.kind == CellKind.MEMORY for cell in immune._cells)

    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=2)
    assert any(cell.kind == CellKind.MEMORY for cell in reloaded._cells)


def test_species_assignment_preserves_multiple_niches(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=6, max_population=12)
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)
    immune._cells = []

    low_cluster = np.full(16, 0.1, dtype=np.float32)
    high_cluster = np.full(16, 0.9, dtype=np.float32)

    for idx in range(3):
        immune._cells.append(
            ImmuneCell(
                cell_id=f"low_{idx}",
                lineage_id=f"low_lineage_{idx}",
                kind=CellKind.B,
                receptor=low_cluster.copy(),
                subsystem_scope="state_repository",
            )
        )
    for idx in range(3):
        immune._cells.append(
            ImmuneCell(
                cell_id=f"high_{idx}",
                lineage_id=f"high_lineage_{idx}",
                kind=CellKind.CYTOTOXIC,
                receptor=high_cluster.copy(),
                subsystem_scope="llm_router",
            )
        )

    immune._assign_species()
    assert immune._species_count >= 2
    assert len({cell.species_id for cell in immune._cells}) >= 2
