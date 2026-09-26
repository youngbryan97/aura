"""A constant sat inside every phi she has ever computed.

`soma.expressive["mycelium_density"]` had a default of 0.5 in the state
dataclass, a column in the interoception schema, and the largest term of the
expressive signal the embodiment estimate hands to phi: 0.55 of it, inside 0.35
of the embodiment signal. Nothing in the tree wrote it. So 0.275 of that signal
was the same number on every frame of every campaign, and the column could not
move.

The network it is named for is real, is registered, and keeps its own count.
"""

from __future__ import annotations

import pytest

from core.phases.proprioceptive_loop import _feel_the_density
from core.state.aura_state import AuraState


class _Network:
    def __init__(self, nodes: int, links: int) -> None:
        self._summary = {"nodes": nodes, "links": links}

    def get_topology_summary(self):
        return dict(self._summary)


@pytest.fixture
def registered(monkeypatch):
    def _set(network):
        from core.container import ServiceContainer

        ServiceContainer.set("mycelium", network, required=False)
        monkeypatch.setattr(
            ServiceContainer,
            "get",
            staticmethod(
                lambda name, default=None: network if name == "mycelium" else default
            ),
        )

    return _set


def _density(state: AuraState) -> float:
    return float(state.soma.expressive["mycelium_density"])


def test_as_many_links_as_nodes_is_a_half(registered):
    registered(_Network(nodes=40, links=40))
    state = AuraState.default()
    _feel_the_density(state.soma)
    assert _density(state) == pytest.approx(0.5)


def test_a_denser_network_reads_denser(registered):
    state = AuraState.default()
    registered(_Network(nodes=40, links=10))
    _feel_the_density(state.soma)
    sparse = _density(state)
    registered(_Network(nodes=40, links=400))
    _feel_the_density(state.soma)
    assert _density(state) > sparse


def test_it_stays_inside_zero_and_one(registered):
    state = AuraState.default()
    for nodes, links in ((1, 10_000), (10_000, 1), (3, 3)):
        registered(_Network(nodes=nodes, links=links))
        _feel_the_density(state.soma)
        assert 0.0 <= _density(state) <= 1.0


def test_an_empty_network_leaves_the_reading_alone(registered):
    registered(_Network(nodes=0, links=0))
    state = AuraState.default()
    before = _density(state)
    _feel_the_density(state.soma)
    assert _density(state) == before


def test_no_network_leaves_the_reading_alone(monkeypatch):
    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer, "get", staticmethod(lambda name, default=None: default)
    )
    state = AuraState.default()
    before = _density(state)
    _feel_the_density(state.soma)
    assert _density(state) == before


def test_a_network_that_raises_leaves_the_reading_alone(monkeypatch):
    class _Broken:
        def get_topology_summary(self):
            raise RuntimeError("no topology")

    from core.container import ServiceContainer

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _Broken() if name == "mycelium" else default),
    )
    state = AuraState.default()
    before = _density(state)
    _feel_the_density(state.soma)
    assert _density(state) == before


def test_the_density_reaches_the_somatic_coupling(registered):
    """It is 0.55 of the expressive term, so it has to move what phi reads."""
    from core.phases.phi_consciousness import _somatic_coupling

    state = AuraState.default()
    registered(_Network(nodes=40, links=10))
    _feel_the_density(state.soma)
    sparse = _somatic_coupling(state)
    registered(_Network(nodes=40, links=400))
    _feel_the_density(state.soma)
    assert _somatic_coupling(state) > sparse
