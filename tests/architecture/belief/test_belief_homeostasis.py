from core.environment.belief_graph import BeliefEdge, BeliefNode, EnvironmentBeliefGraph
from core.environment.homeostasis import Homeostasis
from core.environment.ontology import EntityState, ObjectState, ResourceState
from core.environment.parsed_state import ParsedState


def test_belief_graph_preserves_unobserved_and_decays_confidence():
    graph = EnvironmentBeliefGraph()
    state = ParsedState(
        environment_id="env",
        context_id="ctx",
        sequence_id=1,
        entities=[EntityState(entity_id="self", kind="self", label="self", context_id="ctx", last_seen_seq=1)],
        objects=[ObjectState(object_id="stairs", kind="transition", label=">", context_id="ctx", last_seen_seq=1)],
    )
    graph.update_from_parsed_state(state)
    before = graph.stable_hash()
    graph.decay_unobserved(300)
    assert "stairs" in graph.nodes
    assert graph.nodes["stairs"].confidence < 1.0
    assert graph.stable_hash() != before


def test_belief_graph_records_edges_frontiers_hazards_and_blocked_edges():
    graph = EnvironmentBeliefGraph()
    graph.upsert_node(BeliefNode("a", "tile", "a", "ctx"))
    graph.upsert_node(BeliefNode("b", "tile", "b", "ctx"))
    graph.upsert_edge(BeliefEdge("a", "b", "adjacent"))
    assert graph.shortest_safe_path("a", "b") == ["a", "b"]
    graph.record_blocked_edge("a", "b", reason="wall")
    assert graph.shortest_safe_path("a", "b") == []


def test_homeostasis_critical_resource_raises_stabilization():
    parsed = ParsedState(
        environment_id="env",
        resources={"health": ResourceState(name="health", value=2, max_value=10, critical_below=0.3, trend=-0.1)},
    )
    homeostasis = Homeostasis()
    assessment = homeostasis.assess(homeostasis.extract(parsed))
    assert "health" in assessment.critical_resources
    assert assessment.recommended_goal == "stabilize_resource"
