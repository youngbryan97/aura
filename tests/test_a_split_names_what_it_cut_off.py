"""A partition names the cells it cut off, not only how many.

LIVE 2026-09-19: "population split into 7 pieces: 44,1,1,1,1,1,1", held red
for 828 s, and nothing on any surface said which six cells nothing binds to.
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

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
    fake._cells_alone_in_their_subsystem = MethodType(
        morph_runtime.MorphogeneticRuntime._cells_alone_in_their_subsystem, fake
    )
    morph_runtime.MorphogeneticRuntime._publish_telemetry(fake)
    sizes = published[0]["component_sizes"]
    assert sizes.startswith("3,1,1")
    # Named where she declared them, and by id where nothing declares them.
    assert "cut off from the rest: curiosity (drives), stray" in sizes


def test_a_cell_alone_in_its_subsystem_does_not_read_as_a_split(monkeypatch):
    """It is left unbound on purpose, so the piece it makes is explained.

    LIVE: morphogenesis.components sat red_high at 7 for the life of every
    process — six cells, each the only one in its subsystem, each its own
    piece — and the channel that means "the population came apart" never
    meant it.
    """
    published: list[dict] = []
    monkeypatch.setattr(
        "core.morphogenesis.telemetry.publish", lambda status: published.append(dict(status))
    )
    monkeypatch.setattr("core.morphogenesis.telemetry.publish_motifs", lambda _status: None)

    def cell(cell_id, subsystem):
        return SimpleNamespace(
            cell_id=cell_id,
            manifest=SimpleNamespace(name=cell_id, subsystem=subsystem),
        )

    cells = [
        cell("a", "memory"), cell("b", "memory"),
        cell("lonely", "drives"), cell("stray", "weather"),
    ]
    fake = SimpleNamespace(
        governor=SimpleNamespace(status=lambda: {}),
        graph=SimpleNamespace(components=lambda: [{"a", "b"}, {"lonely"}, {"stray"}]),
        motifs=SimpleNamespace(status=lambda: {}),
        registry=SimpleNamespace(active_cells=lambda: cells),
    )
    fake._cells_alone_in_their_subsystem = MethodType(
        morph_runtime.MorphogeneticRuntime._cells_alone_in_their_subsystem, fake
    )
    morph_runtime.MorphogeneticRuntime._publish_telemetry(fake)
    assert published[0]["unexplained_pieces"] == 1


def test_a_piece_that_nothing_explains_is_counted(monkeypatch):
    published: list[dict] = []
    monkeypatch.setattr(
        "core.morphogenesis.telemetry.publish", lambda status: published.append(dict(status))
    )
    monkeypatch.setattr("core.morphogenesis.telemetry.publish_motifs", lambda _status: None)

    def cell(cell_id, subsystem):
        return SimpleNamespace(
            cell_id=cell_id, manifest=SimpleNamespace(name=cell_id, subsystem=subsystem)
        )

    cells = [cell("a", "memory"), cell("b", "memory"), cell("c", "memory")]
    fake = SimpleNamespace(
        governor=SimpleNamespace(status=lambda: {}),
        # Two cells of one subsystem in one piece, the third cut off from them.
        graph=SimpleNamespace(components=lambda: [{"a", "b"}, {"c"}]),
        motifs=SimpleNamespace(status=lambda: {}),
        registry=SimpleNamespace(active_cells=lambda: cells),
    )
    fake._cells_alone_in_their_subsystem = MethodType(
        morph_runtime.MorphogeneticRuntime._cells_alone_in_their_subsystem, fake
    )
    morph_runtime.MorphogeneticRuntime._publish_telemetry(fake)
    assert published[0]["unexplained_pieces"] == 2
