"""Every column must move when the thing it is named for moves, and only then.

The state schema pairs each column with the attribute it reads. That pairing is
a claim, and until this test existed nothing checked it: nine of the world
model's seventeen columns were one place out from `model_hidden_norm` onward,
so `causal_nodes` was carrying the forward model's last surprise and the hidden
norm was carrying a count of available facets. Every value was real. Every name
was wrong, and every finding about the world model named the wrong quantity.

The check writes a distinguishable value into each declared source that is a
plain path on `AuraState`, reads the whole state vector, and requires that the
column named for that source is among the ones that moved. It cannot cover the
columns that read an organ rather than the state — those are listed here rather
than skipped silently, so the list is a statement of what is unchecked.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.state.aura_state import AuraState
from core.subject.state import DOMAINS, _SCHEMAS, feature_names, read_core_state

#: Columns whose source is an organ, a hash of several fields, or a derived
#: quantity with no single writable path. Named rather than skipped.
UNCHECKABLE_PREFIXES = ("A.", "G.", "C.", "S.", "N.")

#: Sources that are not a path on the state object.
def _is_state_path(source: str) -> bool:
    if not source or "[" in source or "(" in source:
        return False
    head = source.split(".", 1)[0]
    return head in {"world", "cognition", "soma", "motivation", "identity", "cold", "affect"}


def _write(state: AuraState, path: str, value: object) -> bool:
    node = state
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, dict) else getattr(node, part, None)
        if node is None:
            return False
    name = parts[-1]
    if isinstance(node, dict):
        node[name] = value
        return True
    if not hasattr(node, name):
        return False
    setattr(node, name, value)
    return True


def _sample_for(current: object) -> object:
    if isinstance(current, dict):
        return {f"probe_{i}": {"seen": i} for i in range(5)}
    if isinstance(current, list):
        return [{"probe": i, "content": f"probe {i}", "timestamp": 1.0} for i in range(5)]
    if isinstance(current, bool):
        return not current
    if isinstance(current, (int, float)):
        return float(current) + 7.0
    if isinstance(current, str) or current is None:
        return "a distinguishable probe string"
    return None


def _pairs() -> list[tuple[str, str, str]]:
    out = []
    for domain in DOMAINS:
        schema = _SCHEMAS[domain]
        for name, source in zip(schema.features, schema.sources, strict=True):
            if _is_state_path(source):
                out.append((domain, name, source))
    return out


PAIRS = _pairs()


def test_there_are_columns_to_check() -> None:
    assert len(PAIRS) >= 20, f"the check has nothing to check: {len(PAIRS)}"


@pytest.mark.parametrize("domain,name,source", PAIRS, ids=[f"{d}.{n}" for d, n, _ in PAIRS])
def test_a_column_moves_when_its_own_source_moves(domain: str, name: str, source: str) -> None:
    columns = feature_names()
    key = f"{domain}.{name}"
    assert key in columns
    index = columns.index(key)

    before_state = AuraState.default()
    before = read_core_state(before_state).vector()

    after_state = AuraState.default()
    node = after_state
    for part in source.split(".")[:-1]:
        node = node.get(part) if isinstance(node, dict) else getattr(node, part, None)
        if node is None:
            pytest.skip(f"{source} is not reachable on a default state")
    leaf = source.split(".")[-1]
    current = node.get(leaf) if isinstance(node, dict) else getattr(node, leaf, None)
    sample = _sample_for(current)
    if sample is None or not _write(after_state, source, sample):
        pytest.skip(f"{source} cannot be written on a default state")

    after = read_core_state(after_state).vector()
    moved = np.flatnonzero(np.abs(after - before) > 1e-12)
    if moved.size == 0:
        pytest.skip(f"writing {source} moved nothing, so the pairing cannot be checked")
    assert index in moved, (
        f"{key} says it reads {source}; writing it moved "
        f"{[columns[i] for i in moved]} and not {key}"
    )


# ── the organ half, where the misalignment actually was ──────────────────


class _StubWorldModel:
    """A world model whose every reported quantity can be set one at a time."""

    def __init__(self, **values: float) -> None:
        self.values = {
            "surprise": 0.1,
            "hidden_norm": 0.2,
            "mean_surprise": 0.3,
            "last_surprise": 0.4,
            "step_count": 5.0,
            "train_steps": 6.0,
            "nodes": 7.0,
            "edges": 8.0,
            "causal_edges": 1.0,
            **values,
        }

    def surprise(self) -> float:
        return float(self.values["surprise"])

    def status(self) -> dict:
        return {
            "facets": {
                "learned": {
                    "available": True,
                    "detail": {
                        "hidden_norm": self.values["hidden_norm"],
                        "mean_surprise": self.values["mean_surprise"],
                        "last_surprise": self.values["last_surprise"],
                        "step_count": self.values["step_count"],
                        "train_steps": self.values["train_steps"],
                    },
                },
                "causal": {
                    "available": True,
                    "detail": {
                        "nodes": self.values["nodes"],
                        "edges": self.values["edges"],
                        "causal_edges": self.values["causal_edges"],
                    },
                },
            }
        }


#: leaf the stub reports -> the column the schema says carries it
WORLD_MODEL_COLUMNS = {
    "surprise": "W.model_surprise",
    "hidden_norm": "W.model_hidden_norm",
    "mean_surprise": "W.model_mean_surprise",
    "last_surprise": "W.model_last_surprise",
    "step_count": "W.model_steps",
    "nodes": "W.causal_nodes",
    "edges": "W.causal_edges",
    "causal_edges": "W.causal_confirmed",
    "train_steps": "W.model_train_steps",
}


@pytest.mark.parametrize("leaf,column", sorted(WORLD_MODEL_COLUMNS.items()))
def test_a_world_model_column_carries_the_quantity_it_is_named_for(leaf: str, column: str) -> None:
    from core.subject.state import Organs

    columns = feature_names()
    index = columns.index(column)
    state = AuraState.default()

    base = read_core_state(state, organs=Organs(world_model=_StubWorldModel())).vector()
    changed = read_core_state(
        state, organs=Organs(world_model=_StubWorldModel(**{leaf: 0.77}))
    ).vector()

    moved = np.flatnonzero(np.abs(changed - base) > 1e-12)
    assert moved.size, f"changing {leaf} moved nothing at all"
    assert index in moved, (
        f"{column} says it carries {leaf}; changing {leaf} moved "
        f"{[columns[i] for i in moved]}"
    )
