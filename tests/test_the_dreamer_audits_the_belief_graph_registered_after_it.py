"""Her sleep cycle's integrity audit reads the belief graph that boot registered after the dreamer was built.

The dreamer is a singleton built when first asked for, and the belief graph is registered late in boot, so a dreamer
built first kept None and logged "No belief graph available — skipping audit" on every sleep cycle of day run 4's
fifth start (10 November 2026).
"""
from __future__ import annotations

from core.container import ServiceContainer
from core.sleep.dreamer_v2 import DreamerV2


def test_a_dreamer_built_before_the_graph_reads_it_when_it_audits(monkeypatch):
    dreamer = DreamerV2(brain=object(), knowledge_graph=object(), belief_graph=None)
    graph = object()
    original = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get",
                        staticmethod(lambda name, default=None: graph if name == "belief_graph"
                                     else original(name, default=default)))
    assert dreamer._the_belief_graph() is graph
    assert dreamer.belief_graph is graph, "kept once found"


def test_a_graph_given_at_construction_is_the_one_audited():
    given = object()
    assert DreamerV2(brain=object(), knowledge_graph=object(), belief_graph=given)._the_belief_graph() is given
