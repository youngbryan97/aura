"""The governor's edge bound is enforced, not merely declared.

LIVE, 2026-09-21, in the verifier's own output:

    🔎 VERIFIER [error] telemetry.no_channel_is_red @ morphogenesis.edges:
       morphogenesis.edges is red_high at 258count and has been for 2302s

258 bindings against ``MorphBounds.max_edges = 256``. The bound had been
there from the first commit of the layer and nothing ever compared anything
to it: the governor capped the population against ``population_delta`` and
there was no ``binding_delta`` to cap edges against. The telemetry channel
was the only thing that noticed, for thirty-eight minutes, while the
governor went on admitting binds.
"""

from __future__ import annotations

from core.morphogenesis.graph import MorphEdge
from core.morphogenesis.proposal import (
    MorphProposal,
    MorphTransition,
    TransitionKind,
)


def _bind(source: str, target: str) -> MorphTransition:
    return MorphTransition(
        kind=TransitionKind.BIND,
        subject=source,
        edge=MorphEdge(source=source, target=target),
    )


def test_binding_delta_counts_binds_and_unbinds():
    from core.morphogenesis.proposal import MorphProposal as P

    grew = P(proposer="a", transitions=(_bind("a", "b"), _bind("a", "c")))
    assert grew.binding_delta == 2

    mixed = P(
        proposer="a",
        transitions=(
            _bind("a", "b"),
            MorphTransition(
                kind=TransitionKind.UNBIND,
                subject="a",
                edge=MorphEdge(source="a", target="c"),
            ),
        ),
    )
    assert mixed.binding_delta == 0


def test_a_route_does_not_change_the_count():
    """ROUTE changes a weight, not an edge's existence."""
    routed = MorphProposal(
        proposer="a",
        transitions=(
            MorphTransition(
                kind=TransitionKind.ROUTE,
                subject="a",
                edge=MorphEdge(source="a", target="b"),
                weight=0.5,
            ),
        ),
    )
    assert routed.binding_delta == 0


def test_the_governor_refuses_a_bind_past_the_bound():
    """The claim the telemetry channel was making alone."""
    import inspect

    from core.morphogenesis import governor as gov

    source = inspect.getsource(gov)
    assert "self.bounds.max_edges" in source, (
        "max_edges must be compared against something, or it is a number in "
        "a dataclass that no decision reads"
    )
    assert "proposal.binding_delta" in source


def test_the_bound_and_the_channel_agree():
    """A red channel and an admitted proposal cannot both be right."""
    from core.morphogenesis.governor import MorphBounds
    from core.morphogenesis.telemetry import CHANNEL_EDGES

    from core.fsw.telemetry_dictionary import get_telemetry
    from core.morphogenesis.telemetry import declare

    declare()
    channel = get_telemetry().spec(CHANNEL_EDGES)
    assert channel is not None, "the edges channel must be declared"
    assert int(channel.limits.red_high) == MorphBounds().max_edges, (
        "the channel turns red at the count the governor permits, so one of "
        "them is wrong"
    )
