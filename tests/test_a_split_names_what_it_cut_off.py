"""A partition names the cells it cut off, not only how many.

LIVE 2026-09-19: "population split into 7 pieces: 44,1,1,1,1,1,1", held red
for 828 s, and nothing on any surface said which six cells nothing binds to.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.morphogenesis import runtime as morph_runtime


def test_the_event_names_the_cells_outside_the_largest_piece(monkeypatch):
    published: list[dict] = []
    monkeypatch.setattr(
        "core.morphogenesis.telemetry.publish", lambda status: published.append(dict(status))
    )
    monkeypatch.setattr("core.morphogenesis.telemetry.publish_motifs", lambda _status: None)
    def cell(cell_id, name, subsystem):
        return SimpleNamespace(
            cell_id=cell_id, manifest=SimpleNamespace(name=name, subsystem=subsystem)
        )

    fake = SimpleNamespace(
        governor=SimpleNamespace(status=lambda: {}),
        graph=SimpleNamespace(
            components=lambda: [{"lonely"}, {"a", "b", "c"}, {"stray"}]
        ),
        motifs=SimpleNamespace(status=lambda: {}),
        registry=SimpleNamespace(
            active_cells=lambda: [
                cell("lonely", "curiosity", "drives"),
                cell("a", "recall", "memory"),
            ]
        ),
    )
    morph_runtime.MorphogeneticRuntime._publish_telemetry(fake)
    sizes = published[0]["component_sizes"]
    assert sizes.startswith("3,1,1")
    # Named where she declared them, and by id where nothing declares them.
    assert "cut off from the rest: curiosity (drives), stray" in sizes
