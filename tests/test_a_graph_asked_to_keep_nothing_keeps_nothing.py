"""The in-memory sentinel, on a store that is JSON over a path.

`BeliefGraph(persist_path=":memory:")` is what two tests wrote to say "no disk
behind this". The store took the sentinel as a relative path and wrote the
graph to a file literally named `:memory:` in whatever directory the process
started in — one sat in the repository root from 2026-09-18, holding a real
belief graph with Bryan in it, and made `reqproof capture` refuse every proof
for a dirty tree.
"""
from __future__ import annotations

import os
from pathlib import Path

from core.world_model.belief_graph import BeliefGraph


def _graph_with_a_belief(**paths: str) -> BeliefGraph:
    graph = BeliefGraph(**paths)
    graph.update_belief("AURA_SELF", "knows", "Bryan", confidence_score=0.9)
    return graph


def test_the_sentinel_writes_no_file_at_all(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    graph = _graph_with_a_belief(persist_path=":memory:", causal_path=":memory:")

    graph._save(force=True)
    graph.flush()
    graph._save_causal(force=True)

    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_an_ordinary_path_still_persists(tmp_path: Path, monkeypatch) -> None:
    """The null: the guard must not have turned saving off for everyone."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "world.json"
    graph = _graph_with_a_belief(
        persist_path=str(target), causal_path=str(tmp_path / "causal.json")
    )

    graph._save(force=True)
    graph.flush()

    assert target.exists()
    assert "AURA_SELF" in target.read_text()


def test_a_sentinel_graph_reloads_as_empty(tmp_path: Path, monkeypatch) -> None:
    """Nothing kept means nothing found, rather than reading a stray file."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ":memory:").write_text('{"nodes": {"LEFTOVER": {}}, "edges": []}')

    graph = BeliefGraph(persist_path=":memory:", causal_path=":memory:")

    assert "LEFTOVER" not in graph.graph.nodes
    assert os.path.exists(tmp_path / ":memory:")
